#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import regex
import sys
import copy
import time
import uuid
import numpy
import argparse
import pickle
import random
import threading
import multiprocessing
import concurrent.futures
import traceback
import queue
import gc
import tempfile
from scipy import stats
from random import randint
from collections import Counter, defaultdict, deque
from datetime import datetime

global debug_mode
global smask

tm001=0.0
tm002=0.0
tm003=0.0
tm004=0.0
tm005=0.0
tm006=0.0
tm007=0.0
tm008=0.0
tm009=0.0
tm010=0.0
tm011=0.0

# KS-test boundaries from Section 4.2.5.
UNIFORM_THRESHOLD=0.98
UNIFORM_EPSILON=0.08
STAR_THRESHOLD=25
MISSING_TOKEN="<MISSING>"



seqnum=1


# Thread-pool branch backend (--parallel-backend thread) support. The sequential and
# process-pool backends keep using the plain module globals above directly (each process
# has its own memory, so that adapter is safe there); a thread-pool worker instead reads
# and writes the thread-local copies below so concurrent branch threads cannot stomp on
# each other's discovered_patterns/seqnum/random state. See _get_seqnum() etc. below.
_thread_local_discovery = threading.local()


def _thread_local_discovery_active():
    return getattr(_thread_local_discovery, "active", False)


def _get_discovered_patterns():
    if _thread_local_discovery_active():
        return _thread_local_discovery.discovered_patterns
    return discovered_patterns


def _get_seqnum():
    if _thread_local_discovery_active():
        return _thread_local_discovery.seqnum
    return seqnum


def _set_seqnum(value):
    if _thread_local_discovery_active():
        _thread_local_discovery.seqnum = value
    else:
        global seqnum
        seqnum = value


def _next_randint(a, b):
    if _thread_local_discovery_active():
        return _thread_local_discovery.rng.randint(a, b)
    return randint(a, b)



MOD_FACTOR = 32

MIN_NUM_OF_TERMS=1 # minimum required number of terms in the term band, initially I introduced this to filter out one word term band.

RANDOM_SAMPLE_SIZE=4096
# frequent words must be above this cut-off line
CUTOFF_COUNT=20
CC_THRESHOLD=0.1 # select only the word-term pairs that have correlation above this value

WILDCARD_THRESHOLD=0.9 # if almost all filter vectors are filled (above this threshold), then the rest of them will be * regardless of value distribution.
FILTER_THRESHOLD=70.0 # (percentage) above this max runlength proportion, it will be added to the filter

PATTERN_THRESHOLD=3 # there must be more than this number of values to say if there is any pattern among string values

def sanitize_id(id):
    return id.strip().replace(" ", "")

(_ADD, _DELETE, _INSERT) = list(range(3))
(_ROOT, _DEPTH, _WIDTH) = list(range(3))

class LogDataset:
    """Read-only logs and sampling indexes shared by discovery branches.

    logs[i] is one preprocessed log, log_scores[i] is its score, and log_score_indices[score] lists every matching log index i.
    """

    def __init__(self, log_count, logs, log_scores, log_score_indices):
        self.log_count = log_count
        self.logs = logs
        self.log_scores = log_scores
        self.log_score_indices = log_score_indices


class TokenizedLogFileStore:
    """Store all tokenized logs in a temporary file and load requested logs by index."""

    def __init__(self, filename, offsets):
        self.filename = filename
        self.offsets = offsets
        self.file_handle = None

    @classmethod
    def write(cls, all_tlogs, directory):
        filename = os.path.join(directory, "tokenized-logs.pickle")
        offsets = []
        with open(filename, "wb") as file_handle:
            for tlog in all_tlogs:
                offsets.append(file_handle.tell())
                pickle.dump(tlog, file_handle, pickle.HIGHEST_PROTOCOL)
        return cls(filename, offsets)

    def open(self):
        if self.file_handle is None:
            self.file_handle = open(self.filename, "rb")

    def read(self, log_index):
        self.open()
        self.file_handle.seek(self.offsets[log_index])
        return pickle.load(self.file_handle)

    def close(self):
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

    def __getstate__(self):
        return {"filename":self.filename, "offsets":self.offsets, "file_handle":None}

class Node:

    def __init__(self, name, log_count, identifier=None, expanded=True):
        self.__identifier = (str(uuid.uuid1()) if identifier is None else
                sanitize_id(str(identifier)))
        self.name = name
        self.expanded = expanded
        self.__bpointer = None
        self.__fpointer = []
        # custom states
        self.log_templates = []
        self.rep_logs = []
        self.all_vect = [-1]*log_count
        self.score_counts = None

    @property
    def identifier(self):
        return self.__identifier

    @property
    def bpointer(self):
        return self.__bpointer

    @bpointer.setter
    def bpointer(self, value):
        if value is not None:
            self.__bpointer = sanitize_id(value)

    @property
    def fpointer(self):
        return self.__fpointer

    def update_fpointer(self, identifier, mode=_ADD):
        if mode is _ADD:
            self.__fpointer.append(sanitize_id(identifier))
        elif mode is _DELETE:
            self.__fpointer.remove(sanitize_id(identifier))
        elif mode is _INSERT:
            self.__fpointer = [sanitize_id(identifier)]

    def is_leaf_node(self):
        if len(self.fpointer)==0:
            return True
        return False

    def print_node(self):
        print("* Node name:", self.name)
        print("     log template count:", len(self.log_templates))
        print("     rep_logs count:", len(self.rep_logs))
        print("     all_vect valid count:", sum(1 for x in self.all_vect if x>0))

class Tree:

    def __init__(self):
        self.nodes = []
        self.serial = 0
        self.coterm_set = set()

    def get_index(self, position):
        for index, node in enumerate(self.nodes):
            if node.identifier == position:
                break
        return index

    def create_node(self, name, log_count, identifier=None, parent=None):

        node = Node(name, log_count, identifier)
        self.nodes.append(node)
        self.__update_fpointer(parent, node.identifier, _ADD)
        node.bpointer = parent
        return node

    def find_inprogress_node(self):
        for node in self.nodes:
            if len(node.fpointer)==0 and -1 in node.all_vect: # if there is no child and -1 is in the all_vect, it is the unfinished leaf node.
                return node
        return None

    def find_leaf_node(self):
        for node in self.nodes:
            if len(node.fpointer)==0:
                return node
        return None

    def find_node(self, identifier):
        for node in self.nodes:
            if node.identifier==identifier:
                return node
        return None


    def show(self, position, level=_ROOT):
        queue = self[position].fpointer
        if level == _ROOT:
            print(("{0} [{1}] all_vect(-1)={2} len(log_templates)={3}".format(self[position].name, self[position].identifier, self[position].all_vect.count(-1), len(self[position].log_templates))))
        else:
            print((" "*level*10, "{0} [{1}] all_vect(-1)={2} len(log_templates)={3}".format(self[position].name, self[position].identifier, self[position].all_vect.count(-1), len(self[position].log_templates))))
        if self[position].expanded:
            level += 1
            for element in queue:
                self.show(element, level)  # recursive call

    # Get the list of terms up the parents
    def linage(self, position):
        term_list = []
        term_list.append(self[position].term)

        element = self[position].bpointer
        while element is not None:
            term_list.append(self[element].term)
            element = self[element].bpointer
        return list(reversed(term_list))

    def traverse_leaf(self, position, level=_ROOT):
        queue = self[position].fpointer
        if len(queue)==0:
            data_item = sorted(self.linage(position)[1:])
            if len(data_item)>=MIN_NUM_OF_TERMS:
                self.coterm_set.add(str(data_item))
            return
        level += 1
        for element in queue:
            self.traverse_leaf(element, level)  # recursive call

    def expand_tree(self, position, mode=_DEPTH):
        # Python generator. Loosly based on an algorithm from 'Essential LISP' by
        # John R. Anderson, Albert T. Corbett, and Brian J. Reiser, page 239-241
        yield position
        queue = self[position].fpointer
        while queue:
            yield queue[0]
            expansion = self[queue[0]].fpointer
            if mode is _DEPTH:
                queue = expansion + queue[1:]  # depth-first
            elif mode is _WIDTH:
                queue = queue[1:] + expansion  # width-first

    def is_branch(self, position):
        return self[position].fpointer

    def __update_fpointer(self, position, identifier, mode):
        if position is None:
            return
        else:
            self[position].update_fpointer(identifier, mode)

    def __update_bpointer(self, position, identifier):
        self[position].bpointer = identifier

    def __getitem__(self, key):
        return self.nodes[self.get_index(key)]

    def __setitem__(self, key, item):
        self.nodes[self.get_index(key)] = item

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, identifier):
        return [node.identifier for node in self.nodes if node.identifier is identifier]


delimiter_list = [ ' ', ',', ';', '|', ':', '<', '>', '=', '/','@'] # < and > are added because of '->'. By adding > as delimiter it will be broken up into '-' and the right side string.
keyval_pattern = { "pattern":"[^\s~]+", "label": "~300~" }

def split_by_delimiter(lvl, str_data, delim):
    global seqnum

    #if debug_mode:
    #    print " "*lvl+"\033[0;46m"+"Entering split_by_delimiter"+"\033[0m", "->"+str_data+"<-", "->"+delim+"<-"

    # if input string is just one delimiter character, just return it
    if str_data==delim:
        #if debug_mode:
        #    print " "*lvl+"\033[0;47m"+"Leaving split_by_delimiter"+"\033[0m", "->"+str_data+"<-", "->"+delim+"<-"
        return [delim]

    # TODO: I may need to convert all known patterns first before splitting by delimters.
    #s = detect_all_patterns(lvl+4,s)

    # If I detect key-value pattern, convert the value part into *
    if delim=='=' and '=' in str_data:
        s = ' '+str_data+' '
        # pad front and back with space
        found = True
        while found:
            diff = 0
            found = False
            matched = regex.finditer("[\t ](\S+[=])"+keyval_pattern["pattern"]+"[\t ]", s)
            for m in matched:
                label_str = "~KV"+format(seqnum,'09d')+"~"
                seqnum += 1
                seqnum = seqnum % MOD_FACTOR 
                s = s[0:m.start()+1-diff]+m.group(1)+label_str+s[(m.end()-1)-diff:]
                diff = diff + ((m.end()-1)-(m.start()+1)) - len(m.group(1)+label_str)
                found = True
        #print "::::::::::->"+str_data+"<----->"+s+"<-"
        str_data = s[1:len(s)-1] # remove spaces at the front and back

    # split the input string by delimiter
    tokenized = regex.split("("+regex.escape(delim)+")",str_data) # keep the delimiter within the list

    # remove empty token
    removed = True
    while removed:
        removed = False
        if "" in tokenized:
            tokenized.remove("")
            removed = True
    #if debug_mode:
    #    print " "*lvl+"Removed empty token"+"\033[93;44m"+str(tokenized)+"\033[0m"

    # process each token
    for i in reversed(list(range(0,len(tokenized)))):

        tok = tokenized[i]
        if tok==delim:
            continue

        if debug_mode and len(tok)==0:
            print(" "*lvl+"\033[0;31m"+"WARNING 742: zero length token detected -"+"\033[0m", str_data)
            print(" "*(lvl+4), tokenized)
            sys.exit(0)
            continue

        # if any delimiter in lower priority than current one exists, recursively call the split_by_delimiter 
        low_delim_found = False
        cur = delimiter_list.index(delim)
        for j in range(cur+1,len(delimiter_list)):
            if delimiter_list[j] in tok:

                #if debug_mode:
                #    print " "*lvl+"\033[0;42m"+"[split_by_delimiter("+delim+")] CALLING itself("+delimiter_list[j]+")"+"\033[0m ->"+tokenized[i]+"<-"
                #    print " "*lvl, "****",tok, i
                #    print " "*lvl, "****",tokenized
                #    print " "*lvl, "****",tokenized[0:i]
                #    print " "*lvl, "****",tokenized[i+1:]

                tokenized = tokenized[:i] + split_by_delimiter(lvl+4, tok, delimiter_list[j]) + tokenized[i+1:]

                #if debug_mode:
                #    print " "*lvl+"\033[0;42m"+"[split_by_delimiter("+delim+")] Returned"+"\033[0m ->"+str(tokenized)+"<-"
                #    print " "*lvl, "****",tokenized

                low_delim_found = True
                break
        #if not low_delim_found:
        #    if debug_mode:
        #        print " "*lvl+"\033[0;33m"+"[split_by_delimiter("+delim+")] CALLING detect_all_patterns()"+"\033[0m ->"+tokenized[i]+"<-"
        #    tokenized[i] = detect_all_patterns(lvl+4,tokenized[i])

    #if debug_mode:
    #    print " "*lvl+"\033[0;47m"+"Leaving split_by_delimiter"+"\033[0m", tokenized
    return tokenized


def get_bracket_char(log):
    pos = len(log)
    bkt_open = None
    bkt_close = None
    # quote and double-quote are treated like parentheses, but since there is no left and right, it is handled differently in the if statement in Custom_split.
    bkt_pair = {"(":")","{":"}","[":"]","<":">","'":"'","\"":"\""}

    for c in "({[<'\"":
        loc = log.find(c)
        if loc>=0 and loc<pos:
            pos = loc
            bkt_open = c
            bkt_close = bkt_pair[c]
    return bkt_open, bkt_close, pos

def custom_split(log):

    tokenized = []
    # Determine the first occuring parentheses
    bracket_open, bracket_close, pos = get_bracket_char(log)
    if bracket_open==None: # no parentheses found
        return split_by_delimiter(8,log,' ')

    # Extract the content within the brackets pair and call recursively
    pstack = [] # for storing index
    qstack = [] # for storing char in case of ' or "
    for i,c in enumerate(log):
        if (c==bracket_open and c not in ["'","\""]) or (c=="'" and "'" not in qstack) or (c=="\"" and "\"" not in qstack):
            pstack.append(i)
            if c in ["'","\""]:
                qstack.append(c)
        elif (c==bracket_close and c not in ["'","\""]) or (c=="'" and "'" in qstack) or (c=="\"" and "\"" in qstack):
            if len(pstack)>0:
                spos = pstack.pop() # start and end position of parentheses segment
                epos = i
            # There are cases where > exists without opening <.
            # Ex) ...org.apache.hadoop.mapreduce.v2.app.MRAppMaster 1><LOG_DIR>/stdout 2><LOG_DIR>/stderr
            else:
                continue
            if len(qstack)>0 and c in ["'","\""]:
                schar = qstack.pop()
                if c!=schar:
                    print("ERROR 531", schar)
                    print(log)
                    sys.exit(0)
            if len(pstack)==0: # if stack becomes empty
                middle_part = custom_split(log[spos+1:epos])
                ending_part = custom_split(log[epos+1:])

                #if debug_mode:
                #    print " "*4+"Whole:",log
                #    print " "*4+"spos=:",spos
                #    print " "*4+"epos=:",epos
                #    print " "*4+"Begin:",log[0:spos]
                #    print " "*4+"Middle:",middle_part
                #    print " "*4+"Ending:",ending_part

                tokenized = split_by_delimiter(8,log[0:spos],' ') + [bracket_open] + middle_part + [bracket_close] + ending_part

                #tokenized = split_by_delimiter(8,log[0:spos],' ') + [bracket_open]
                #if len(middle_part)>0:
                #    tokenized = tokenized + middle_part
                #tokenized = tokenized + [bracket_close]
                #if len(ending_part)>0:
                #    tokenized = tokenized + ending_part
                break

    # Unclosed parentheses may exist. In such case, there are unprocessed strings.
    # Ex) ucState = COMMITTED, replication# = 0 < minimum = 0
    if len(pstack)>0:
        spos = pstack[0] # take the first parentheses in the pstack and do recursive call
        epos = len(log) # assume that there is imaginary closing parenthesis
        middle_part = custom_split(log[spos+1:epos])
        tokenized = split_by_delimiter(4,log[0:spos],' ') + [bracket_open] + middle_part

    #if debug_mode:
    #    print "\033[0;35mcustom_split() returning: "+str(tokenized)+"\033[0m"
    return tokenized


def is_number(s):
    return s.lstrip('-').replace('.','',1).isdigit()

def are_all_numbers(numlist):
    for n in numlist:
        if not(is_number(n)):
            return False
    return True

# included code 2024-03-20
def is_include_percentage(tok):

    if regex.match(r'[\d\w\W\s]*\d+(?:\.\d+)?%[\d\w\W\s]*', tok):
        return True
    return False

# included code 2024-03-20
def are_all_include_percentage(tok_list):
    for tok in tok_list:
        if not(is_include_percentage(tok)):
            return False
    return True

def is_hexa(s):
    try:
        int(s, 16)
        return True
    except ValueError:
        return False

def are_all_hexa(numlist):
    for n in numlist:
        if not(is_hexa(n)):
            return False
    return True


def is_all_integer(numlist):
    for n in numlist:
        # sign check of the first character
        if not(ord(n[0]) in range(ord('0'),ord('9')+1) or n[0] in ['+','-']):
            return False

        for c in n[1:]:
            if ord(c)<48: # '0' is 48
                return False
            if ord(c)>57: # '9' is 57
                return False
    return True


def is_all_floatingpoint(numlist):
    for n in numlist:
        # sign check of the first character
        if not(ord(n[0]) in range(ord('0'),ord('9')+1) or n[0] in ['+','-']):
            return False

        point_detected = False
        for c in n[1:]:
            if not point_detected and c=='.':
                point_detected = True
                continue
            if ord(c)<48: # '0' is 48
                return False
            if ord(c)>57: # '9' is 57
                return False
        if n[-1]=='.':
            return False
        if not point_detected:
            return False
    return True



def follows_format(klist):

    #if debug_mode:
    #    print "        [follows_format()] Entering. len=",len(klist)

    # if there are less than 3 values, it is too less to say whether there is a pattern or not
    if len(klist)<PATTERN_THRESHOLD:
        return False

    # check for string length
    slen = len(klist[0])
    for w in klist[1:]:
        if slen!=len(w):

    #        if debug_mode:
    #            print "        [follows_format()] Various string length!"

            return False
    
    # check if all entries are alphabet only
    alphabet_only = True
    for w in klist:
        if not w.isalpha():
            alphabet_only = False
    if alphabet_only: # if only alphabet, do not add as new pattern
        return False

    # eliminate any word in the form of ~100~ since these are special token I inserted
    keylist = []
    for w in klist:
        if "~" not in w:
            keylist.append(w)
    if len(keylist)<2:
        return False

    # Build a regex that preserves the shared character structure.
    seedkey = keylist[0]
    smask = ""
    fixed_count = 0
    for c in range(0,slen):
        pos_fixed = True
        all_digits = seedkey[c].isdigit()
        all_alpha = seedkey[c].isalpha()
        for w in keylist[1:]:
            if w[c]!=seedkey[c]:
                pos_fixed = False
            if not w[c].isdigit():
                all_digits = False
            if not w[c].isalpha():
                all_alpha = False
        if pos_fixed:
            fixed_count += 1
            smask = smask + regex.escape(seedkey[c])
        elif all_digits:
            smask = smask + "\\d"
        elif all_alpha:
            smask = smask + "[A-Za-z]"
        else:
            return False

    if fixed_count==0:
        return False


    patterns = _get_discovered_patterns()
    pat = { "pattern": smask, "label":"~"+str(400+len(patterns))+"~"}

    pattern_exists = False
    for p in patterns:
        if p["pattern"]==smask:
            pattern_exists = True
            pat = p
            break
    if not pattern_exists:
        patterns.append(pat)

    return True


def all_terms_exist(s, keywords_list):
    for term in keywords_list:
        #if term not in custom_split(s):
        if term not in s:
            return False
    return True


# kws: keyword set
def multiple_term_inclusion_count(lines, kws):
    cnt = 0
    for s in lines:
        if all_terms_exist(s, kws):
            cnt += 1
    return cnt


def print_correlation(bow,v,w1,w2):
    for i in range(0,len(bow)):

        if bow[i]!=w1:
            continue

        for j in range(0,len(bow)):
            if bow[j]!=w2:
                continue

            val = numpy.corrcoef(v[i], v[j])[0][1]
            return w1+":"+w2,"{0:.3f}".format(val)
 

standalone_patterns = [
#    {   "pattern":"\-?\d+",          # integer
#        "label": "~100~",
#        "matcher": None },
#    {   "pattern":"\-?\d+(ms|msec|millisec|s|sec|second|seconds|us|microsec|KiB|GiB|MB|KB|GB|%)", # millisec, seconds, microsec ... in integer value
#        "label": "~101~",
#        "matcher": None },
#    {   "pattern": "\-?\d+\.\d+",     # FP num
#        "label": "~102~",
#        "matcher": None },
#    {   "pattern":"\-?\d+\.\d+(ms|msec|millisec|s|sec|second|seconds|us|microsec|KiB|GiB|MB|KB|GB|%)", # millisec, seconds, microsec ... in FP value
#        "label": "~103~",
#        "matcher": None },
#    {   "pattern": "\-?\d+\.\d+%",     # FP percent
#        "label": "~104~",
#        "matcher": None },
#    {   "pattern":"\d+\^\d+",         # exponent
#        "label": "~105~",
#        "matcher": None },
#    {   "pattern": "0x[\da-f]+",   # hexa num
#        "label": "~106~",
#        "matcher": None },
#    {   "pattern": "155\.230\.91\.\\d{3}(:\\d)?",   # IP and port
#        "label": "~107~",
#        "matcher": None },

    {   "pattern":"[\da-zA-Z]{8}\-[\da-zA-Z]{4}\-[\da-zA-Z]{4}\-[\da-zA-Z]{4}\-[\da-zA-Z]{12}", # UUID format
        "label": "~108~",
        "matcher": None },
#    {'pattern': 'container_\\d{13}_\\d{4}_\\d{2}_\\d{6}', 'label': '~109~', 'matcher': None},
#    {'pattern': 'blk_\\d{10}_\\d{4}', 'label': '~110~', 'matcher': None},
#    {'pattern': 'application_\\d{13}_\\d{4}', 'label': '~111~', 'matcher': None},
#    {'pattern': 'DFSClient_NONMAPREDUCE_\-?\\d+_\\d', 'label': '~112~', 'matcher': None},
#    {'pattern': 'DFSClient_attempt_\\d+_\\d{4}_._000000_0_\\d+_1', 'label': '~113~', 'matcher': None},
#    {'pattern': 'fsimage.ckpt_\\d{19}', 'label': '~114~', 'matcher': None},
#    {'pattern': 'BP\-\\d{9}\-\\d+.\\d+.\\d+.\\d+\-\\d{13}', 'label': '~115~', 'matcher': None},
#    {'pattern': 'appattempt_\\d{13}_\\d{4}_\\d{6}', 'label': '~116~', 'matcher': None},
#    {'pattern': 'job_\\d{13}_\\d{4}', 'label': '~117~', 'matcher': None},
#
    {'pattern': 'req\-[a-z0-9]{8}\-[a-z0-9]{4}\-[a-z0-9]{4}\-[a-z0-9]{4}\-[a-z0-9]{12}', 'label': '~115~', 'matcher': None},

#    {'pattern': 'DFSClient_NONMAPREDUCE_\-\\d{9}_\\d', 'label': '~111~', 'matcher': None},
#    {'pattern': 'DFSClient_attempt_\\d{13}_\\d{4}_r_\\d{6}_\\d_\\d{9}_\\d', 'label': '~112~', 'matcher': None},
#    {'pattern': 'edits_tmp_\\d{19}-\\d{19}_\\d{19}', 'label': '~113~', 'matcher': None},
#    {'pattern': 'application_\\d{13}_\\d{4}', 'label': '~114~', 'matcher': None},
#    {'pattern': 'appattempt_\\d\\d\\d\\d\\d........_\\d\\d\\d\\d_\\d\\d\\d\\d\\d\\d', 'label': '~116~', 'matcher': None},
#    {'pattern': 'deimos\\d.', 'label': '~117~', 'matcher': None},
#    {'pattern': 'job_\\d\\d\\d\\d........._\\d\\d\\d\\d', 'label': '~119~', 'matcher': None},
#    {'pattern': '\\d\\d\\d.\\d\\d\\d.\\d\\d.\\d\\d.', 'label': '~121~', 'matcher': None},
#    {'pattern': '#\\d\\d....', 'label': '~122~', 'matcher': None},
#    {'pattern': '\\d.\\d.s', 'label': '~123~', 'matcher': None},
#    {'pattern': 'DS-........-....-\\d...-....-............', 'label': '~124~', 'matcher': None},
#    {'pattern': 'masterappattempt_\\d\\d\\d\\d........._\\d\\d\\d\\d_\\d\\d\\d\\d\\d\\d', 'label': '~125~', 'matcher': None},
]

# These patterns are checked for every token in each sampled log.
for pattern in standalone_patterns:
    pattern["matcher"] = regex.compile("^"+pattern["pattern"]+"$")

date_matcher = regex.compile("^(\\d{4})\-(\\d{2})\-(\\d{2})$")

# This list grows as we learn more patterns.
discovered_patterns = [
#    {   "pattern":"\-?\d+ms",
#        "label": "~105~" },
#    {'pattern': 'container_\\d\\d\\d\\d........._\\d\\d\\d\\d_\\d\\d_\\d\\d\\d...', 'label': '~106~'},
#    {'pattern': 'appattempt_\\d\\d\\d\\d\\d........_\\d\\d\\d\\d_\\d\\d\\d\\d\\d\\d', 'label': '~107~'},
#    {'pattern': 'deimos\\d.', 'label': '~108~'},
#    {'pattern': 'application_\\d\\d\\d\\d........._\\d\\d\\d\\d', 'label': '~109~'},
#    {'pattern': 'job_\\d\\d\\d\\d........._\\d\\d\\d\\d', 'label': '~110~'},
#    {'pattern': 'blk_\\d\\d\\d\\d\\d\\d\\d..._....', 'label': '~111~'},
#    {'pattern': '\\d\\d\\d.\\d\\d\\d.\\d\\d.\\d\\d.', 'label': '~112~'},
#    {'pattern': '#\\d\\d....', 'label': '~113~'},
#    {'pattern': '\\d.\\d.s', 'label': '~114~'},
#    {'pattern': 'DS-........-....-\\d...-....-............', 'label': '~115~'},
#    {'pattern': 'masterappattempt_\\d\\d\\d\\d........._\\d\\d\\d\\d_\\d\\d\\d\\d\\d\\d', 'label': '~116~'},
]


common_patterns = [
#    {   "pattern":"<=",
#        "label": "~201~" },
#    {   "pattern":"=>",
#        "label": "~202~" },
#    {   "pattern":"<-",
#        "label": "~203~" },
#    {   "pattern":"->",
#        "label": "~204~" },

#    {   "pattern":"don't",
#        "label": "~205~" },
#    {   "pattern":"won't",
#        "label": "~206~" },
#    {   "pattern":"shouldn't",
#        "label": "~207~" },
#    {   "pattern":"couldn't",
#        "label": "~208~" },
#    {   "pattern":"it's",
#        "label": "~209~" },
#    {   "pattern":"It's",
#        "label": "~210~" },
#    {   "pattern":"Didn't",
#        "label": "~211~" },
#    {   "pattern":"didn't",
#        "label": "~212~" },
#    {   "pattern":"wasn't",
#        "label": "~213~" },
#    {   "pattern":"Wasn't",
#        "label": "~214~" },

    {   "pattern": "hdfs://([a-zA-Z0-9_\-\.\*:\+]+/)+([a-zA-Z0-9_\-\.\*:\+]*(\?[a-zA-Z0-9_\-:\+]+=[a-zA-Z0-9_\-:\+]+(&[a-zA-Z0-9_\-:\+]+=[a-zA-Z0-9_\-:\+]+)*)?)",
        "serial": "1",
        "prefix":"hdfs_url" },

    {   "pattern": "hdfs://([a-zA-Z0-9_\-\.]+):([0-9]+)",
        "serial": "1",
        "prefix":"hdfs_url" },

    {   "pattern": "http://([a-zA-Z0-9_\-\.\*:\+]+/)+([a-zA-Z0-9_\-\.\*:\+]*(\?[a-zA-Z0-9_\-:\+]+=[a-zA-Z0-9_\-:\+]+(&[a-zA-Z0-9_\-:\+]+=[a-zA-Z0-9_\-:\+]+)*)?)",
        "serial": "1",
        "prefix":"http_url" },

    {   "pattern": "http://([a-zA-Z0-9_\-\.]+):([0-9]+)",
        "serial": "1",
        "prefix":"http_url" },

    {   "pattern": "https://([a-zA-Z0-9_\-\.\*:\+]+/)+([a-zA-Z0-9_\-\.\*:\+]*(\?[a-zA-Z0-9_\-:\+]+=[a-zA-Z0-9_\-:\+]+(&[a-zA-Z0-9_\-:\+]+=[a-zA-Z0-9_\-:\+]+)*)?)",
        "serial": "1",
        "prefix":"https_url" },

    {   "pattern": "https://([a-zA-Z0-9_\-\.]+):([0-9]+)",
        "serial": "1",
        "prefix":"https_url" },

    {   "pattern": "[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+",
        "serial": "1",
        "prefix":"email" },

    {   "pattern": "http://([a-zA-Z0-9_\-\.]+):([0-9]+)(/[a-zA-Z0-9_\-\.\*\+\-]+)+",
        "serial": "1",
        "prefix":"http_url" },

    {   "pattern": "/(var|tmp|home|usr|home|etc|opt|gogo|airwordcount|wikimean|wikimedian|wikistandarddeviation|cluster|jobhistory|node|ws)(/[a-zA-Z0-9_\-\.\*\+\-]+)+",
        "serial": "1",
        "prefix":"file_path" },

    {   "pattern": "(?<![a-zA-Z0-9_\-\.\*\+])/(?:[a-zA-Z0-9_\-\.\*\+]+/)*[a-zA-Z0-9_\-\.\*\+]+",
        "serial": "1",
        "prefix":"file_path" },

    {   "pattern": "\d+\.\d+\.\d+\.\d+(:\d+)?", # IP address and port number
        "serial": "1",
        "prefix":"ipaddr_port" },

    {   "pattern":"\-?\d+\.\d+ (KB|GB|MB)", # 21.5 MB
        "serial": "1",
        "prefix":"data_size_float" },

    {   "pattern":"\-?\d+ (KB|GB|MB)", # 5 GB
        "serial": "1",
        "prefix":"data_size_int" },
]

def preprocess_known_patterns(logs):
    processed = []
    for i in range(0,len(logs)):
        log = logs[i]
        for item in common_patterns:

            found = True
            while found:
                diff = 0
                found = False
                matched = regex.finditer(item["pattern"], log)
                for m in matched:
                    label_str = item["prefix"]+"_"+format(int(item["serial"]),'09d')
                    item["serial"]=str(int(item["serial"])+1)
                    log = log[0:m.start()-diff]+label_str+log[(m.end())-diff:]
                    diff = diff + ((m.end())-(m.start())) - len(label_str)
                    found = True
        processed.append(log)
    return processed


def replace_known_patterns(tlogs):
    for i in range(0,len(tlogs)):
        tlog = tlogs[i]

        for j in range(0,len(tlog)):

            w = tlog[j]
            if w in [' ',':',',','=','<','>','@']: 
                continue

            # TODO: If I enable this, "assigned" and "released" are not converted into ~10?~ format.
            #if w.isalpha():
            #    continue
            #if not any(i.isdigit() for i in w):
            #    continue

            # preprocess any troublesome characters to special marker
            if '*' in w:
                tlogs[i][j] = regex.sub("\*","~200~",w)
                w = tlogs[i][j]

            for pat in standalone_patterns:
                matched = pat["matcher"].match(w)
                if matched!=None:

                    #tlogs[i][j] = pat["label"]
                    tlogs[i][j] = "~AB"+format(_get_seqnum(),'09d')+"~"
                    _set_seqnum((_get_seqnum()+1) % MOD_FACTOR)

                    #print "\033[0;35m"+matched.group(0)+"\033[0m -->", tlogs[i][j]

            is_date = True
            matched = date_matcher.match(w)
            if matched!=None:
                year = matched.group(1)
                month = matched.group(2)
                day = matched.group(3)
                if int(year)>datetime.today().year:
                    is_date = False
                if int(month)>12:
                    is_date = False
                if int(day)>31:
                    is_date = False

                if is_date:
                    tlogs[i][j] = "~date_"+format(_get_seqnum(),'04d')+"~"
                    _set_seqnum((_get_seqnum()+1) % 1000)
                    #print "Date format reassigned as:",tlogs[i][j]


number_patterns = [
#    {   "pattern": "\d+\.\d+\.\d+\.\d+(:\d)?", # IP address and port number
#        "type":"ipaddrport",
#        "increment":"1",
#        "serial":"1",
#        "matcher": None},

    {   "pattern":"\-?\d+",          # integer
        "type":"int",
        "increment":"1",
        "serial": "1",
        "matcher": None },

    {   "pattern": "\-?\d+\.\d+",     # FP num
        "type":"float",
        "increment":"1",
        "serial": "1",
        "matcher": None },

    {   "pattern":"\-?\d+(ms|msec|millisec|s|sec|second|seconds|us|microsec|KiB|GiB|MB|KB|GB|%)", # millisec, seconds, microsec ... in integer value
        "type":"int_time",
        "increment":"1",
        "serial": "1",
        "matcher": None },

    {   "pattern":"\-?\d+\.\d+(ms|msec|millisec|s|sec|second|seconds|us|microsec|KiB|GiB|MB|KB|GB|%)", # millisec, seconds, microsec ... in FP value
        "type":"float_time",
        "increment":"1",
        "serial": "1",
        "matcher": None },

    {   "pattern":"\-?\d+\^\d+", # exponent
        "type":"exponent",
        "increment":"1",
        "serial": "1",
        "matcher": None },

    {   "pattern": "0x[\da-fA-F]+", # hexa num
        "type":"hexa1", 
        "increment":"1", # not used
        "serial": "1", # not used
        "matcher": None },

    {   "pattern": "[\da-fA-F]+",   # hexa num
        "type":"hexa2", 
        "increment":"1", # not used
        "serial": "1", # not used
        "matcher": None },

#    {   "pattern": "155\.230\.91\.\\d{3}(:\\d)?",   # IP and port
#        "label": "~107~",
#        "matcher": None },
#    {   "pattern":"[\da-zA-Z]{8}\-[\da-zA-Z]{4}\-[\da-zA-Z]{4}\-[\da-zA-Z]{4}\-[\da-zA-Z]{12}", # UUID format
#        "label": "~108~",
#        "matcher": None },
]

# Compile once: this matching loop runs for every token in the input corpus.
for pattern in number_patterns:
    pattern["matcher"] = regex.compile("^"+pattern["pattern"]+"$")


def uniquify_numbers(tlogs):
    for i in range(0,len(tlogs)):
        tlog = tlogs[i]
        for j in range(0,len(tlog)):
            w = tlog[j]
            if w in [' ',':',',','=','<','>']:
                continue
            if '*' in w:
                tlogs[i][j] = regex.sub("\*","~200~",w)
                w = tlogs[i][j]

            found = False
            for p in number_patterns:
                matched = p["matcher"].match(w)
                if matched!=None:
                    found = True
                    break
            if found:
                if p["type"]=="int":
                    tlogs[i][j] = p["serial"] 
                    p["serial"] = str(int(p["serial"])+int(p["increment"])) #
                elif p["type"]=="float":
                    # Keep a decimal representation without binary float drift.
                    serial = int(p["serial"])
                    tlogs[i][j] = str(serial//10)+"."+str(serial%10)
                    p["serial"] = str(serial+int(p["increment"]))
                elif p["type"]=="int_time": 
                    tlogs[i][j] = p["serial"]+matched.group(1)
                    p["serial"] = str(int(p["serial"])+int(p["increment"]))
                elif p["type"]=="float_time":
                    serial = int(p["serial"])
                    tlogs[i][j] = str(serial//10)+"."+str(serial%10)+matched.group(1)
                    p["serial"] = str(serial+int(p["increment"]))
                elif p["type"]=="exponent":
                    tlogs[i][j] = p["serial"]
                    p["serial"] = str(int(p["serial"])+int(p["increment"]))
                elif p["type"]=="hexa1":
                    val = list(matched.group(0)[2:]) 
                    for k in range(0,len(val)):
                        c = val[k]
                        if c in ['0','1','2','3','4','5','6','7','8','9']:
                            val[k]=str(randint(0,9))
                        elif c in ['a','b','c','d','e','f']:
                            val[k]=['a','b','c','d','e','f'][randint(0,5)]
                        elif c in ['A','B','C','D','E','F']:
                            val[k]=['A','B','C','D','E','F'][randint(0,5)]
                        else:
                            print("ERROR 235 c=",c)
                            sys.exit(0)
                    tlogs[i][j] = "0x"+"".join(val)
                elif p["type"]=="hexa2": 
                    '''
                    if any(i.isdigit() for i in matched.group(0)):
                        val = list(matched.group(0))
                        for k in range(0,len(val)):
                            c = val[k]
                            if c in ['0','1','2','3','4','5','6','7','8','9']:
                                val[k]=str(randint(0,9))
                            elif c in ['a','b','c','d','e','f']:
                                val[k]=['a','b','c','d','e','f'][randint(0,5)]
                            elif c in ['A','B','C','D','E','F']:
                                val[k]=['A','B','C','D','E','F'][randint(0,5)]
                            else:
                                print "ERROR 835 c=",c
                                sys.exit(0)
                        tlogs[i][j] = "".join(val)
                    '''
                    pass
                else:
                    print("ERROR 283")
                    sys.exit(0)


def apply_all_patterns(tlogs):
    global tm001

    # Convert known patterns in the token to a marker
    #print "    [apply_all_patterns()] Converting tokens of known patterns to markers using newly discovered pattern ..."
    #print "    [apply_all_patterns()] number of tokenized logs:", len(tlogs)
    tm_checkpt = time.time()
    for i in range(0,len(tlogs)):
        tlog = tlogs[i]
        for j in range(0,len(tlog)):
            word = tlog[j]

            # Two performance optimization
            if word in [' ',':',',','=','<','>']:
                continue

            # TODO: If I enable this, "assigned" and "released" are not converted into ~10?~ format.
            #if word.isalpha():
            #    continue

            #if not any(i.isdigit() for i in word):
            #    continue

            # preprocess any troublesome characters to special marker
            if '*' in word:
                tlogs[i][j] = regex.sub("\*","~200~",word)
                word = tlogs[i][j]

            for p in standalone_patterns:
                matched = p["matcher"].match(word)
                if matched!=None:
                    tlogs[i][j] = p["label"]
                    #print "\033[0;35m"+matched.group(0)+"\033[0m -->", tlogs[i][j]

    elapsed = time.time() - tm_checkpt
    tm001 += elapsed
    #print "    [apply_all_patterns()] Done converting. It took", elapsed, "seconds"


# Apply custom patterns learned since next_pattern_index to the tokenized logs.
def apply_new_patterns(tlogs, next_pattern_index):
    global tm002
    global seqnum

    if next_pattern_index >= len(discovered_patterns):
        return next_pattern_index

    compiled_patterns = [regex.compile("^"+p["pattern"]+"$") for p in discovered_patterns[next_pattern_index:]]
    tm_checkpt = time.time()
    for i in range(0,len(tlogs)):
        tlog = tlogs[i]
        for j in range(0,len(tlog)):
            word = tlog[j]

            # Two performance optimization
            if word in [' ',':',',','=','<','>']:
                continue

            # TODO: If I enable this, "assigned" and "released" are not converted into ~10?~ format.
            #if word.isalpha():
            #    continue

            #if not any(i.isdigit() for i in word):
            #    continue

            # preprocess any troublesome characters to special marker
#            if '*' in word:
#                tlogs[i][j] = re.sub("\*","~200~",word)
#                word = tlogs[i][j]

            for p in compiled_patterns:
                matched = p.match(word)
                if matched!=None:
                    tlogs[i][j] = "~CP"+format(seqnum,'09d')+"~"
                    seqnum += 1
                    seqnum = seqnum % MOD_FACTOR
                    break
    elapsed = time.time() - tm_checkpt
    tm002 += elapsed
    #print "    [apply_new_patterns()] Done converting. It took", elapsed, "seconds"
    return len(discovered_patterns)


# TODO: Combine this with apply_new_patterns() when both discovery paths use the same token-log storage.
def apply_new_patterns_multiprocessing(tlogs):
    global tm002

    compiled_patterns = [regex.compile("^"+p["pattern"]+"$") for p in _get_discovered_patterns()]
    tm_checkpt = time.time()
    for i in range(0,len(tlogs)):
        tlog = tlogs[i]
        for j in range(0,len(tlog)):
            word = tlog[j]
            if word in [' ',':',',','=','<','>']:
                continue
            for pattern in compiled_patterns:
                if pattern.match(word) is not None:
                    tlogs[i][j] = "~CP"+format(_get_seqnum(),'09d')+"~"
                    _set_seqnum((_get_seqnum()+1) % MOD_FACTOR)
                    break
    tm002 += time.time()-tm_checkpt


def Determine_runlen_filter_word(token_d, tlogs, ftwords):

    global new_pattern_added

    # if there is only one word, just add it to the filter_words
    if len(token_d)==1:
        return list(token_d.keys())[0]
    else:
        if is_all_integer(list(token_d.keys())):
            #print "    All integer keys detected!"
            return "*"
        elif is_all_floatingpoint(list(token_d.keys())):
            #print "    All floating number keys detected!"
            return "*"
        elif follows_format(list(token_d.keys())):
#            #print "    Keys follow certain format!", token_d.keys()[0]
#            # Insert newly detected custom pattern to the standalone_patterns dictionary
#            pat = { "pattern": smask, "label":"~"+str(100+len(standalone_patterns))+"~"}
#            pattern_exists = False
#            for p in standalone_patterns:
#                if p["pattern"]==smask:
#                    pattern_exists = True
#                    pat = p
#            if not pattern_exists:
#                standalone_patterns.append(pat)
#                new_pattern_added = True
#                if debug_mode:
#                    print "    [Determine_runlen_filter_word] \033[31;46m New pattern added \033[0m - ", pat
#                    print "    [Determine_runlen_filter_word] \033[37;41m Current filter words:\033[0m - ", ftwords
#                print "\033[31;46m New pattern added \033[0m - ", pat
#
#                # Update all the tokenized logs
#                apply_new_patterns(tlogs)
#                apply_new_patterns([ftwords]) # Update all the tokens in the filter words as well. Input should be list of list.
#
#                if debug_mode:
#                    print "    [Determine_runlen_filter_word] \033[38;42m Updated filter words:\033[0m - ", ftwords
#            else:
#                if debug_mode:
#                    print "    [Determine_runlen_filter_word] Pattern already exists - ", pat, str(ftwords)
            return "*"
        return sorted(token_d, key=lambda k: token_d[k], reverse=True)[0]


def determine_filter_word(token_d, tlen, fillup_ratio):

    pv = compute_uniformity_pvalue(token_d)
    cr = 100.0*float(len(token_d))/float(tlen) # cardinality
    if debug_mode:
        print("    \033[34;46mpv:"+str(pv)+"\033[0m \033[34;42mcr:"+str(cr)+"\033[0m")

    # if there is only one word, just add it to the filter_words
    if len(token_d)==1:
        if debug_mode:
            print("    \033[43;5m"+"STATIC STRING because there is only one value in the dictionary."+"\033[0m")
        #print "\033[43;5m"+"STATIC STRING because there is only one value in the dictionary."+"\033[0m"
        return list(token_d.keys())[0], pv, cr

    #if debug_mode:
    #    print "    Tokens in the dictionary:",token_d.keys()

    if any(w in token_d for w in [" ", "@","<",">","=","(",")"]):
        if debug_mode:
            print("    \033[43;5m"+"STATIC STRING because special char (including space) is in the token keys."+"\033[0m")
        #print "\033[43;5m"+"STATIC STRING because special char (including space) is in the token keys."+"\033[0m"
        return sorted(token_d, key=lambda k: token_d[k], reverse=True)[0], pv, cr

    if are_all_numbers(list(token_d.keys())):
        if debug_mode:
            print("    \033[43;5m"+"WILDCARD because they are all numbers."+"\033[0m")
        #print "\033[43;5m"+"WILDCARD because they are all numbers."+"\033[0m"
        return '*', pv, cr

    is_custom_pattern_marker = all(token.startswith('~CP') and token.endswith('~') for token in token_d)
    if not is_custom_pattern_marker and follows_format(list(token_d.keys())):
        if debug_mode:
            print("    \033[43;5m"+"WILDCARD because new pattern is detected."+"\033[0m")
        #print "\033[43;5m"+"WILDCARD because new pattern is detected."+"\033[0m"
        return '*', pv, cr
    if pv>(1.0 + UNIFORM_THRESHOLD) / 2.0:
        if debug_mode:
            print("    \033[43;5m"+"WILDCARD because it is a uniform distribution."+"\033[0m")
        #print "\033[43;5m"+"WILDCARD because it is a uniform distribution."+"\033[0m"
        return '*', pv, cr

    if are_all_include_percentage(list(token_d.keys())):
        return '*', pv, cr

    
    if are_all_hexa(list(token_d.keys())):
        if debug_mode:
            print("    \033[43;5m"+"WILDCARD because they are all hexadecimal numbers."+"\033[0m")
        #print "\033[43;5m"+"WILDCARD because they are all hexadecimal numbers."+"\033[0m"
        return '*', pv, cr

    token =  sorted(token_d, key=lambda k: token_d[k], reverse=True)[0]
    # A custom-pattern marker is converted to a wildcard during final template generation.
    if '~' in token.replace('~200~', '') and not token.startswith('~CP'):
        if debug_mode:
            print("    \033[43;5m"+"WILDCARD because it is a known pattern."+"\033[0m")
        #print "\033[43;5m"+"WILDCARD because it is a known pattern."+"\033[0m"
        return '*', pv, cr

    #if fillup_ratio!=None and fillup_ratio>WILDCARD_THRESHOLD:
    #    print "\033[43;5m"+"WILDCARD because fill-up ratio is reached."+"\033[0m", fillup_ratio
    #    return "*", pv, cr

    if debug_mode:
        print("    \033[43;5m"+"STATIC STRING because it did not meet any condition for the wildcard."+"\033[0m")
    #print "\033[43;5m"+"STATIC STRING because it did not meet any condition for the wildcard."+"\033[0m"

    return token, pv, cr


def read_log_files(flist, filter_str):
    logs = []
    for i in range(0,len(flist)):
        for log in flist[i]:
            if len(log.strip())==0:
                continue

            if filter_str != None:
                if filter_str not in log:
                    continue
                print(log.strip())

            log = " ".join(log.split())
            logs.append(log)


    print("Total number of logs loaded:",len(logs))
    return logs


def match_and_remove(tmpl,logs):

    global tm011
    tm_checkpt = time.time()

    # first, build a list of index to delete
    to_delete = []
    match_count = 0
    for i in range(0,len(logs)):
        log = logs[i]
        matched = regex.match("^"+tmpl+"$",log)

        if matched!=None:
            match_count = match_count + 1
            to_delete.append(i)
            #if debug_mode:
            #    print "DEL:",log

    # delete matched logs
    before_removal = len(logs)
    to_delete = sorted(to_delete)
    for i in reversed(sorted(to_delete)):
        del logs[i]
    del_count = before_removal - len(logs)
    if match_count!=del_count:
        print("Error 263: match count and delete count mismatch!")
        sys.exit(0)

    elapsed = time.time() - tm_checkpt
    tm011+=elapsed
    return del_count


def exist_match(log_template, logs):
    template_matcher = regex.compile("^"+log_template+"$")
    for i in range(0,len(logs)):
        if logs[i]==None: 
            continue
        matched = template_matcher.match(logs[i])
        if matched!=None:
            if debug_mode:
                print("\033[0;32mMatch found at "+str(i)+":", logs[i], "\033[0m ")
            return i
    return -1


def test_multiple_match(rlogs, vect, log_template):

    cnt = 0
    to_delete = []
    for j in range(0,len(rlogs)):
        if vect[j]==-1:
            continue
        log = rlogs[j]
        matched = regex.match("^"+log_template+"$", "".join(log))
        if matched!=None:
            #print "      ", "".join(log)
            to_delete.append(j)
            cnt += 1
    return cnt, to_delete


def mark_matched_logs(logs, vect, rlogs, log_template, i, score_counts=None, log_scores=None):
#    print('Here is mark_matched_logs')
    # very long spark log have trouble at this function, so we must check.
#    print('logs: ')
#    print(logs)
#    print('log_template: ')
#    print(log_template)

    #print "Entering mark_matched_logs() Star_count:",log_template.count(".*")
    marked = 0 # how many logs match to the log template?
    replog_selected = False

    template_matcher = regex.compile("^"+log_template+"$")

    for j in range(0,len(logs)):

        if vect[j]>-1: # skip logs already matched by previous templates
            continue

        log = logs[j]

        matched = template_matcher.match(log)
        if matched!=None:
            vect[j] = i
            marked += 1

            if score_counts is not None:
                score = log_scores[j]
                score_counts[score] -= 1
                if score_counts[score] == 0:
                    del score_counts[score]

            if not replog_selected:
                replog_selected = True
                rlogs.append(log)

#    if sum(1 for x in vect if x>0) != len(rlogs):
#        print "ERROR: vect("+str(sum(1 for x in vect if x>0))+") and rlogs length("+str(len(rlogs))+") does not match!!"
#        sys.exit(0)

    #print "Leaving mark_matched_logs()"
    return marked


def remove_log_template_matches(logs, logtem):
    sample_logs = []
    # Remove matching logs using pre-filled log templates
    for i in range(0,len(logtem)):
        log_template = logtem[i]
        # Remove any matched logs from the logs.
        before_removal = len(logs)
        alog = None
        for j in reversed(list(range(0,len(logs)))):
            matched = regex.match("^"+log_template+"$",logs[j])
            if matched!=None:
                alog = logs[j]
                del logs[j]
        removed_logs = before_removal - len(logs)

        if removed_logs==0:
            print("ERROR: no matching logs found from the given template.")
            print("template:", log_template)
            sys.exit(0)

        #print "\033[0;31m"+"["+format(i,'3d')+"]",format(len(logs),'5d'),format(removed_logs,'4d'),"\033[0m","\033[0;32m\""+logtem[i]+"\",\033[0m"
        print("["+format(i,'3d')+"]",format(len(logs),'5d'),format(removed_logs,'4d'), logtem[i])

        sample_logs.append(alog)
    print("Done removing logs using pre-populated log templates. Remaining logs:", len(logs))
    return sample_logs


def build_random_index(data_len, sample_len):
    # create random index list
    if data_len<=sample_len:
        numbers = list(range(0,data_len))
    else:
        numbers = set()
        #while len(numbers)<sample_len:
        #    numbers.add(randint(0,data_len-1))

        # TODO deterministically generate the number list
        num = 0
        while len(numbers)<sample_len:
            numbers.add(num)
            num += 1

    return numbers


def sample_by_length(logs, vect, ssize):
    global tm006
    tm_checkpt = time.time()

    tlen_d = defaultdict()
    for log in logs:
        toklen = len(log.split())
        if toklen not in tlen_d:
            tlen_d[toklen]= []
        tlen_d[toklen].append(log)
    most_popular = sorted(tlen_d, key=lambda k: len(tlen_d[k]), reverse=True)[0]


    for x in tlen_d:
        print(x, len(tlen_d[x]))

    sys.exit(0)

    selected=[]
    for log in logs:
        if len(log.split())==most_popular:
            selected.append(log)
        if len(selected)>=ssize:
            elapsed = time.time() - tm_checkpt
            #print "{0:.3f}".format(elapsed), "Random log selection"
            tm006+=elapsed
            return selected
    sys.exit(0)


def build_log_scores(logs):
    scores = []
    for log in logs:
        token_count = len(log.split())
        space_count = log.count(' ') + log.count('\t') + log.count(',') + log.count(':') + log.count(';') + log.count(',')
        scores.append(token_count*1000 + space_count)
    return scores


def build_log_score_index(log_scores):
    score_indices = defaultdict(list)
    for i, score in enumerate(log_scores):
        score_indices[score].append(i)
    return score_indices


def sample_by_token_length_and_space_count(logs, tlogs, vect, log_scores, score_counts, score_indices):
    global tm006
    tm_checkpt = time.time()

    most_popular = max(score_counts, key=score_counts.get)

    if debug_mode:
        print("** Summary of log groups using characters **")
        for score in sorted(score_counts, key=score_counts.get, reverse=True)[:20]:
            print("  For the key of",format(score,'5d')+",", format(score_counts[score], '5d'),"logs are grouped.")
        print("    ...")

    selected = []
    tselected = []
    if score_indices is not None:
        indices = score_indices[most_popular]
    else:
        indices = range(len(log_scores))
    for i in indices:
        if vect[i]==-1 and log_scores[i]==most_popular:
            selected.append(logs[i])
            tselected.append(tlogs[i])
            if len(selected)>=1000:
                break

    elapsed = time.time() - tm_checkpt
    #print "{0:.3f}".format(elapsed), "Random log selection"
    tm006+=elapsed

    return selected, tselected


def sample_by_term_correlation(logs, ssize):
    global tm008
    tm_checkpt = time.time()

    numset = build_random_index(len(logs), ssize)

    rand_logs = random_sample_logs(logs, ssize)
    toke_logs = do_tokenization(rand_logs)
    bow,bow_list = select_significant_terms(toke_logs)
    all_vect = build_term_vectors(bow_list,toke_logs)
    corr_d = compute_term_correlation(all_vect, bow, bow_list, max(len(x) for x in list(bow.keys())))
    while corr_d==None:
        rand_logs = random_sample_logs(logs, ssize)
        toke_logs = do_tokenization(rand_logs)
        bow,bow_list = select_significant_terms(toke_logs)
        all_vect = build_term_vectors(bow_list,toke_logs)
        corr_d = compute_term_correlation(all_vect, bow, bow_list, max(len(x) for x in list(bow.keys())))
        print("corr_d None. Looping one more time.")

    term_groups = determine_term_groups(corr_d,bow,rand_logs)
    tb = sorted(term_groups, key=len, reverse=True)[0]

    selected=[]
    tselected=[]
    for log in logs:
        if all_terms_exist(log, tb):
            selected.append(log)
            tselected.append(custom_split(log))
        if len(selected)>=ssize:
            elapsed = time.time() - tm_checkpt
            #print "{0:.3f}".format(elapsed), "Term correlation-based selection"
            tm008+=elapsed
            return selected,tselected

    elapsed = time.time() - tm_checkpt
    #print "{0:.3f}".format(elapsed), "Term correlation-based selection"
    tm008+=elapsed

    return selected,tselected


def sample_by_signature(logs, ssize):
    global tm007
    tm_checkpt = time.time()

    numset = build_random_index(len(logs), ssize)

    # dictionary of signature
    tlen_d = defaultdict()
    for n in numset:
        signature = regex.sub("[\d\s\w]","",logs[n])
        if signature not in tlen_d:
            tlen_d[signature]=0
        tlen_d[signature] += 1
    most_popular = sorted(tlen_d, key=lambda k: tlen_d[k], reverse=True)[0]

    #for x in sorted(tlen_d, key=lambda k: tlen_d[k], reverse=True):
    #    print x, tlen_d[x]
    #sys.exit(0)

    selected=[]
    for log in logs:
        if regex.sub("[\d\s\w]","",log)==most_popular:
            selected.append(log)
        if len(selected)>=ssize:

            elapsed = time.time() - tm_checkpt
            #print "{0:.3f}".format(elapsed), "Random log selection"
            tm007+=elapsed
            return selected

    elapsed = time.time() - tm_checkpt
    #print "{0:.3f}".format(elapsed), "Random log selection"
    tm007+=elapsed

    return selected

def random_sample_logs(logs, n):
    global tm003
    # Randomly select RANDOM_SAMPLE_SIZE logs from the original set
    # To avoid selecting the same log, I first collect the set of RANDOM_SAMPLE_SIZE index and then create a log list.
    if len(logs)<=n:
        return logs
    '''
    tm_checkpt = time.time()
    numset = build_random_index(len(logs), ssize)
    selected = []
    for pos in numset:
        selected.append(logs[pos])
    elapsed = time.time() - tm_checkpt
    #print "{0:.3f}".format(elapsed), "Random log selection"
    '''

    tm_checkpt = time.time()
    selected = []
    step_size = len(logs) / RANDOM_SAMPLE_SIZE
    for log in logs[::step_size]:
        if len(selected)>=RANDOM_SAMPLE_SIZE:
            break
        selected.append(log)
    elapsed = time.time() - tm_checkpt
    #print "{0:.3f}".format(elapsed), "Random log selection"

    tm003+=elapsed
    return selected


def do_tokenization(logs):
    global tm004
    tm_checkpt = time.time()
    tlogs = []
    for i in range(0,len(logs)):
        log = logs[i]
        tlogs.append(custom_split(log))
    elapsed = time.time() - tm_checkpt
    #print "{0:.3f}".format(elapsed), "Tokenizing"
    tm004+=elapsed
    return tlogs


def select_significant_terms(tlog_data):
    # Build bag-of-words with their frequencies
    tm_checkpt = time.time()
    wd = defaultdict()
    for tlog in tlog_data:
        for w in tlog:
            if not w.isalpha():
                continue
            if len(w)<=2:
                continue
            if w not in wd:
                wd[w] = 0
            wd[w] += 1
    # Delete infrequent words
    for w in wd:
        if wd[w] < CUTOFF_COUNT:
            del wd[w]
    wl = sorted(wd, key=lambda k: wd[k], reverse=True)
    elapsed = time.time() - tm_checkpt
    #print "{0:.3f}".format(elapsed),"Creating bag-of-words took"
    if debug_mode:
        print("The length of bag-of-words list:",len(wl))
    return wd, wl


def build_term_vectors(wl,tlog_data):
    # Build word vectors for all the words
    # One vector indicates whether the word exists in that log line
    # Thus, the vector length is equal to the length of tlog_data
    tm_checkpt = time.time()
    vectors = []
    for i in range(0,len(wl)):
        w = wl[i]
        ivect = [0]*len(tlog_data)
        for j in range(0,len(tlog_data)):
            tlog = " "+" ".join(tlog_data[j])+" "
            if " "+w+" " in tlog:
                ivect[j]=1
        #print str(bow[w]),w+" "*(40-len(w)),"".join(str(x) for x in ivect)
        vectors.append(ivect)
    elapsed = time.time() - tm_checkpt
    #print "{0:.3f}".format(elapsed),"Creating indicator vectors for all words"
    if debug_mode:
        print("Length of vectors:",len(vectors))
    return vectors


def compute_term_correlation(vect, bow, bow_list, max_word_length):
    # Compute the correlations of all word pairs into correlation_dict dictionary
    tm_checkpt = time.time()
    correlation_dict = defaultdict()
    for i in range(0,len(vect)):
        word = bow_list[i]
        if word not in correlation_dict:
            correlation_dict[word] = defaultdict()
        for j in range(0,len(vect)):
            term = bow_list[j]
            if term not in correlation_dict:
                correlation_dict[term] = defaultdict()
            if word==term:
                correlation_dict[word][term]=1.0
                correlation_dict[term][word]=1.0
                continue

            if term not in correlation_dict[word] and word not in correlation_dict[term]:
                val = numpy.corrcoef(vect[i], vect[j])[0][1]
            elif term not in correlation_dict[word] and word in correlation_dict[term]:
                val = correlation_dict[term][word]
            elif term in correlation_dict[word] and word not in correlation_dict[term]:
                val = correlation_dict[word][term]
            else:
                val = correlation_dict[term][word]

            if val>CC_THRESHOLD: # or val<-1.0*CC_THRESHOLD:
                #print bow_list[j]+" "*(50-len(bow_list[j])),"\t","{0:.3f}".format(val)
                #corr_list.append("{0:.3f}".format(val)+":"+bow_list[j])
                correlation_dict[word][term] = val
                correlation_dict[term][word] = val
    # Print word correlations
#    for word in sorted(correlation_dict, key=lambda k: bow[k], reverse=True):
#        cc_list="["
#        for term in sorted(correlation_dict[word], key=lambda k: correlation_dict[word][k], reverse=True):
#            if word==term:
#                continue
#            cc_list = cc_list+term+":"+"{0:.3f}".format(correlation_dict[word][term])+", "
#        cc_list += "]" 
#        print "*",format(bow[word],'4d'),word+" "*(max_word_length-len(word)), cc_list
    elapsed = time.time() - tm_checkpt
    #print "{0:.3f}".format(elapsed), "Correlation computation"

    # check if correlation_dict is empty or not
    clen = 0
    for word in sorted(correlation_dict, key=lambda k: bow[k], reverse=True):
        clen += len(correlation_dict[word])
        #print "        [compute_term_correlation] length of "+word, len(correlation_dict[word]), correlation_dict[word]
    if clen==len(correlation_dict):
        print("WARNING 219: correlation_dict is empty")
        #for t in log_templates:
        #    print "\033[0;43m"+str(t["count"])+"\033[0m",t["template"]
        print("Total number of templates:",len(log_templates))
        return None

    #print print_correlation(bow_list,vect,"Stopping","Exit")
    return correlation_dict


def display_term_groups(correlation_dict, word_dict):
    # Determine the correlated term group
    tm_checkpt = time.time()
    tgrp = [] # term group
    circ = []
    max_word_length = max(len(x) for x in list(word_dict.keys()))
    for word in sorted(correlation_dict, key=lambda k: word_dict[k], reverse=True):
        # if word is already part of any of the previous groups, skip it
        is_member = False
        for g in tgrp:
            if word in g:
                is_member = True
        if is_member:
            continue
        for term in sorted(correlation_dict[word], key=lambda k: correlation_dict[word][k], reverse=True):
            # append new term only if its correlation coefficient to all the members are above threshold
            eligible = True
            for member in circ:
                #print "member:",member," term:",term, " member keys:", correlation_dict[member].keys()
                if term not in correlation_dict[member]:
                    eligible = False
                    break
                val = correlation_dict[member][term]
                if val<CC_THRESHOLD:
                    eligible = False
                    break
            if eligible: # the word itself is added automatically as the first member here because correlation_dict has itself
                circ.append(term)
        if len(circ)>=2:
            print(word, " "*(max_word_length-len(word)), format(len(correlation_dict[word]),'3d'), format(len(circ),'3d'), circ)
            tgrp.append(sorted(circ))
        # circle members determined at this point
        circ = []
    elapsed = time.time() - tm_checkpt
    if debug_mode:
        print("Displaying term groups took", elapsed, "seconds")
    return


def determine_term_groups(correlation_dict, word_dict, logs):
    # Determine the correlated term group
    tm_checkpt = time.time()
    tgrp = [] # term group
    circ = []
    max_word_length = max(len(x) for x in list(word_dict.keys()))
    for word in sorted(correlation_dict, key=lambda k: word_dict[k], reverse=True):
        # if word is already part of any of the previous groups, skip it
        is_member = False
        for g in tgrp:
            if word in g:
                is_member = True
        if is_member:
            continue
        for term in sorted(correlation_dict[word], key=lambda k: correlation_dict[word][k], reverse=True):
            # append new term only if its correlation coefficient to all the members are above threshold
            eligible = True
            for member in circ:
                #print "member:",member," term:",term, " member keys:", correlation_dict[member].keys()
                if term not in correlation_dict[member]:
                    eligible = False
                    break
                val = correlation_dict[member][term]
                if val<CC_THRESHOLD:
                    eligible = False
                    break
            if eligible: # the word itself is added automatically as the first member here because correlation_dict has itself
                circ.append(term)
        if len(circ)>=2:
            # counting inclusion of all terms is very costly
            #print word, " "*(max_word_length-len(word)), format(len(correlation_dict[word]),'3d'), format(len(circ),'3d'), format(multiple_term_inclusion_count(logs,circ),'5d'), circ
            if debug_mode:
                print(word, " "*(max_word_length-len(word)), format(len(correlation_dict[word]),'3d'), format(len(circ),'3d'), circ)
            tgrp.append(sorted(circ))
        # circle members determined at this point
        circ = []
    elapsed = time.time() - tm_checkpt
    #print "{0:.3f}".format(elapsed), "Determining term groups"
    return tgrp 


def do_filtering(tlogs, valid_vect, flt_words, flt_valid_vect):

    # Filter logs based on the flt_words
    filtered = []
    for i in range(0,len(tlogs)):

        if valid_vect[i]==0:
            continue

        tlog = tlogs[i]

        # go through the tokens for each log and select only the ones that match the filter words in active positions
        add_ok = True
        for j in range(0,len(flt_valid_vect)):

            if flt_valid_vect[j]==0:
                continue
            if flt_words[j]=='*':
                continue

            # Match a missing filter word only when this column is absent.
            if flt_words[j]==MISSING_TOKEN:
                if j>=len(tlog):
                    continue
                else:
                    add_ok = False
                    break

            if j>len(tlog)-1: 
                add_ok = False
                break
            if tlog[j] != flt_words[j]: 
                add_ok = False
                break

        if add_ok:
            filtered.append(tlog)
        else:
            valid_vect[i] = 0

    # TODO: if there is 0 filtered logs, we need to drop some filter words.

    return filtered
 

def Generate_log_template(fwords):

    # Transform the discovered log template into a python-ready form
    log_template = "".join(fwords).strip()

    log_template = regex.sub("\\\\","\\\\\\\\",log_template )
    log_template = regex.sub("\-","\-",log_template)
    log_template = regex.sub("\[","\[",log_template)
    log_template = regex.sub("\]","\]",log_template)
    log_template = regex.sub("\(","\(",log_template)
    log_template = regex.sub("\)","\)",log_template)
    log_template = regex.sub("\$","\$",log_template)
    log_template = regex.sub("\?","\?",log_template)
    log_template = regex.sub("\+","\+",log_template)
    log_template = regex.sub("\|","\|",log_template)
    #log_template = re.sub("\\\\","~201~",log_template)

    log_template = regex.sub("\*","\S+",log_template)

    for it in standalone_patterns:
        log_template = regex.sub(it["label"],"\S+",log_template)
    
    log_template = regex.sub("~200~","\*",log_template)

    # recover special common word patterns
    #for n in range(0,len(common_patterns)):
    #    if common_patterns[n]["label"]=="~324~" or common_patterns[n]["label"]=="~325~":
    #        log_template = re.sub(common_patterns[n]["label"], "\S+ \S+", log_template)
    #    else:
    #        log_template = re.sub(common_patterns[n]["label"], "\S+", log_template)
    #log_template = re.sub("~300~","\S+",log_template)


    log_template = regex.sub("\\S+~","\S+",log_template)

    log_template = regex.sub("\\\\\[\\\\\]","\\[\S*\\]",log_template)
    log_template = regex.sub("{}","{\S*}",log_template)

    final_template = []
    for t in log_template.split():
        if "\S+" in t and "=\S+" not in t and ":\S+" not in t: 
            final_template.append("\S+")
        else:
            final_template.append(t)
    log_template = " ".join(final_template)

    found = True
    while found:
        diff = 0
        found = False
        matched = regex.finditer("(\\\\S\+)[:]\\\\S\+", log_template)
        for m in matched:
            log_template = log_template[0:m.start()-diff]+"\S+"+log_template[m.end()-diff:]
            diff = diff + 4
            found = True

    while "\S+\S+" in log_template:
        log_template = regex.sub("\\\\S\+\\\\S\+", "\S+", log_template)


    #log_template = re.sub("\\\\S\+ \\\\S\+",".*",log_template)
    #log_template = re.sub("\\\\S\+,\\\\S\+",".*",log_template)
    #log_template = re.sub("\\\\S\+:\\\\S\+",".*",log_template)

    #print "Log Sample:  ", "\033[0;31m"+log_sample+"\033[0m"
    #print "Log template:", "\033[0;47m"+log_template+"\033[0m"
    return log_template


# realcall: True if the call is for the final log_template generation, not just for testing the current log_template
def generate_log_template_star(fwords,realcall):

    search_patterns = [
                        { "pattern":"( |_|:|,|<|\[|\(|\"|/)(\-?\d+)( |_|:|>|,|\]|\)|\(|/|\"|$)",       "type":"int" },  # int

#                        { "pattern":"_\-?\d+",       "type":"int" },  # int
#                        { "pattern":"\-?\d+_",       "type":"int" },  # int
#                        { "pattern":"_\-?\d+_",       "type":"int" },  # int
#                        { "pattern":" \-?\d+$",       "type":"int" },  # int
#                        { "pattern":" \-?\d+ ",       "type":"int" },  # int
#                        { "pattern":" \-?\d+,",       "type":"int" },  # int
#                        { "pattern":" \-?\d+\)",       "type":"int" },  # int
#                        { "pattern":"\(\-?\d+\)",       "type":"int" },  # int
#                        { "pattern":":\-?\d+,",       "type":"int" },  # int
#                        { "pattern":":\-?\d+$",       "type":"int" },  # int
#                        { "pattern":":\-?\d+ ",       "type":"int" },  # int
#                        { "pattern":":\-?\d+>",       "type":"int" },  # int
#                        { "pattern":":\-?\d+:",       "type":"int" },  # int

                        { "pattern":"( |_|:|,|<|\[|/|\"|\()(\-?\d+\.\d+)( |_|:|>|,|\]|\)|/|\"|$)",       "type":"int" },  # FP
#                        { "pattern":" \-?\d+\.\d+$",  "type":"float" }, # FP

                        { "pattern":"( |_|:|,|<|\[|\(|\"|/)(\-?\d+) ?(ms|msec|millisec|s|sec|second|seconds|us|microsec|KiB|GiB|MB|KB|GB|%)", "type":"int_time" }, # millisec, seconds, microsec ... in integer value
                        { "pattern":"( |_|:|,|<|\[|\(|\"|/)(\-?\d+\.\d+) ?(ms|msec|millisec|s|sec|second|seconds|us|microsec|KiB|GiB|MB|KB|GB|%)", "type":"float_time" }, # FP value with unit
                        #{ "pattern":"\-?\d+\^\d+", "type":"exponent", "increment":"1" }, # exponent
                        #{ "pattern": "0x[\da-fA-F]+", "type":"hexa1" },
                        #{ "pattern": "[\da-fA-F]+", "type":"hexa2"}
                    ]
 
    # Transform the discovered log template into a python-ready form
    log_template = "".join(fwords).strip()

    for item in search_patterns:

        found = True
        while found:
            diff = 0
            found = False
            matched = regex.finditer(item["pattern"], log_template)
            for m in matched:

                found = True

                #if found and realcall:
                #    print "   \033[1;91m<(1)", item["pattern"], "\033[0m"
                #    print "   ", fwords
                #    print "   \033[0;103m<(2)", log_template, "\033[0m"

                # m.start(2) is the int part. I just want to replace that to .*, leaving characters front and back intact.
                log_template = log_template[0:m.start(2)-diff]+".*"+log_template[(m.end(2))-diff:]
                diff = diff + ((m.end(2))-(m.start(2))) - len(".*")

                #if found and realcall:
                #    print "   \033[0;102m<(3)", log_template, "\033[0m"

    log_template = regex.sub("\\\\","\\\\\\\\",log_template )
    log_template = regex.sub("\-","\-",log_template)
    log_template = regex.sub("\[","\[",log_template)
    log_template = regex.sub("\]","\]",log_template)
    log_template = regex.sub("\(","\(",log_template)
    log_template = regex.sub("\)","\)",log_template)
    log_template = regex.sub("\$","\$",log_template)
    log_template = regex.sub("\?","\?",log_template)
    log_template = regex.sub("\+","\+",log_template)
    log_template = regex.sub("\|","\|",log_template)
    log_template = regex.sub(r"\{", r"\\{", log_template)
    log_template = regex.sub(r"\}", r"\\}", log_template)
    #log_template = re.sub("\\\\","~201~",log_template)

    #log_template = re.sub("\*","\S+",log_template)
    log_template = regex.sub("\*",".*",log_template)

    for it in standalone_patterns:
        #log_template = re.sub(it["label"],"\S+",log_template)
        log_template = regex.sub(it["label"],".*",log_template)
    
    log_template = regex.sub("~200~","\*",log_template)

    # recover special common word patterns
    #for n in range(0,len(common_patterns)):
    #    log_template = re.sub(common_patterns[n]["label"], ".*", log_template)
    #log_template = re.sub("~300~",".*",log_template)

    log_template = regex.sub("~\\S+~",".*",log_template)

    for p in common_patterns:
        marker = p["prefix"]+"_"+"\\d{9}"
        log_template = regex.sub(marker,".*",log_template)

    log_template = regex.sub("\\\\\[\\\\\]","\\[.*\\]",log_template)
    log_template = regex.sub("{}","{.*}",log_template)

    final_template = []
    for t in log_template.split():
        if ".*" in t and "=.*" not in t and ":.*" not in t and regex.match("^\.+\*(ms|msec|millisec|s|sec|second|seconds|us|microsec|KiB|GiB|MB|KB|GB|%)$", t)==None:
            final_template.append(".*")
        else:
            final_template.append(t)
    log_template = " ".join(final_template)

    found = True
    while found:
        diff = 0
        found = False
        matched = regex.finditer("(\.\*)[:]\.\*", log_template)
        for m in matched:
            log_template = log_template[0:m.start()+1-diff]+log_template[(m.end()-1)-diff:]
            diff += 3
            found = True

    while ".* .*" in log_template:
        log_template = regex.sub("\.\* \.\*", ".*", log_template)

    while "..*" in log_template:
        log_template = regex.sub("\.\.\*", ".*", log_template)

    while ".*.*.*.*" in log_template:
        log_template = regex.sub("\.\*\.\*\.\*\.\*", ".*", log_template)
    while ".*.*.*" in log_template:
        log_template = regex.sub("\.\*\.\*\.\*", ".*", log_template)
    while ".*.*" in log_template:
        log_template = regex.sub("\.\*\.\*", ".*", log_template)

    if log_template.count(".*") > STAR_THRESHOLD:
        parts=log_template.split(".*")
        log_template=""
        for i in range(STAR_THRESHOLD):
            log_template+=parts[i]
            log_template+=".*"

    #if realcall:
    #    print "   \033[33;35m<(4)", log_template, "\033[0m"

    return log_template


def postprocess_raw_template(fwords):
    logtem= "".join(fwords).strip()
    for it in standalone_patterns:
        logtem= regex.sub(it["label"],"*",logtem)
    logtem = regex.sub("~200~","*",logtem)
    logtem = regex.sub("~300~","*",logtem)
    return logtem 


def compute_uniformity_pvalue(val_dict):
    tensor=[] 
    for n in sorted(val_dict, key=lambda k: val_dict[k], reverse=False):
        if len(tensor)==0:
            cumul = 0
        else:
            cumul = tensor[-1]
        tensor.append(cumul+val_dict[n]) 
    pval = stats.kstest(tensor,'uniform',args=(tensor[0],tensor[-1])).pvalue
    return pval


def exist_partial_match2(rlogs, fwords, fmask):
    survived = []
    for i in range(0,len(rlogs)):
        tlog = rlogs[i]
        # go through the tokens for each log and select only the ones that match the filter words in active positions
        add_ok = True
        for j in range(0,len(fmask)):
            if fmask[j]==0:
                continue
            if j>len(tlog)-1: 
                add_ok = False
                break
            if fwords[j]=='*':
                continue
            if '~' in fwords[j]:
                continue
            if tlog[j] != fwords[j]:
                add_ok = False
                break
        if add_ok:
            survived.append(tlog)
    print("    ",len(survived))
    return survived
 
def exist_partial_match(rlogs, rvect, fwords, fmask):
    print("Entering exist_partial_match.")
    print("Filter words:",fwords)
    print("Filter masks:",fmask)

    for k in range(0,len(rlogs)):
        if rvect[k]==-1: 
            continue
        tokenized = rlogs[k] 

        #print "Repre log:",tokenized

        match_found = True
        for m in range(0,len(fmask)): 
            if fmask[m]==0:
                continue
            if m >= len(tokenized):
                #print "Difference found!! Length too short."
                match_found = False
                break
            #print "fwords[m]="+fwords[m], "m="+str(m), "len(tokenized)="+str(len(tokenized)), "match_found="+str(match_found)

            if '~' in fwords[m]:
                continue

            if fwords[m]!=tokenized[m]:
                #print "Difference found!!", fwords[m], tokenized[m]
                match_found = False
                break

        if match_found:
            #print "Leaving exist_partial_match() with True! Checked", k, "logs."
            return True
        #    print "===> Match found"
        #    print rlogs[k]
        #    print "".join(fwords)
        #    print "".join(str(x) for x in fmask)
    #print "Leaving exist_partial_match() with False! Checked", k, "logs."
    return False

def finalize_filter_with_star(fword,fmask):
    tfword = copy.deepcopy(fword)
    tfmask = copy.deepcopy(fmask)
    for i in range(0,len(tfmask)):
        if tfmask[i]==0:
            tfword[i]="*"
            tfmask[i]=1
    return tfword, tfmask


def construct_candidate_log_templates(input_logs, rep_logs):

    global tm005
    global debug_mode

    valid_mask = [1]*len(input_logs)

    column_cnt = max(len(x) for x in input_logs)
    filter_words = [""]*column_cnt
    filter_mask = [0]*column_cnt

    tm_checkpt = time.time()
    token_added_order = []
    count_added_order = [] 
    candidate_set = [] 
    add_candidate = False
    while 0 in filter_mask: # only when filter selection vector is not full

        filtered_logs = do_filtering(input_logs, valid_mask, filter_words, filter_mask)

        if debug_mode:
            print("XXXXXXXX check 001")
            input("\033[0;35m->Press ENTER to continue filtering ...\033[0m")

        if debug_mode:
            print("Updating column_cnt from", column_cnt,"to",max(len(x) for x in filtered_logs))
            #for x in filtered_logs:
            #    print "    ",x
        column_cnt = max(len(x) for x in filtered_logs)
        filter_words = filter_words[:column_cnt]
        filter_mask = filter_mask[:column_cnt]

        if 0 not in filter_mask: 
            break

        max_runlen = 0
        max_runlen_pct = 0.0
        max_runlen_pos = -1
        max_runlen_positions = []

        all_column_dict = defaultdict() # dict of dict, column_dict per tpos is saved here
        for tpos in range(0,column_cnt):

            # Skip the token position if it has been added to the filter words already
            if filter_mask[tpos]==1:
                continue

            column_dict = defaultdict()
            for tlog in filtered_logs: # tlog is the filtered and tokenized log lines
                # Missing positions participate in the column frequency.
                if len(tlog) > tpos:
                    tok = tlog[tpos]
                else:
                    tok = MISSING_TOKEN
                if tok not in column_dict: # if key is not yet created, make one
                    column_dict[tok] = 0
                column_dict[tok] += 1 # increment count
            all_column_dict[tpos] = column_dict

            if len(column_dict)==0: 
                print("ERROR 832: No token values collected into the dictionary. Perhaps log's length ran out.")
                sys.exit(0)

            runlength_token = sorted(column_dict, key=lambda k: column_dict[k], reverse=True)[0]
            runlength = column_dict[runlength_token] 

            runlen_percent = float(runlength*100.0)/float(len(filtered_logs))
            if runlen_percent<100.0 and runlen_percent>max_runlen_pct:
                max_runlen_pct = runlen_percent
                max_runlen_positions = [tpos]
            elif runlen_percent<100.0 and runlen_percent==max_runlen_pct:
                max_runlen_positions.append(tpos)
            #print tpos, max_runlen_pos, "->"+runlength_token+"<-", runlen_percent

            if runlength==len(filtered_logs):

                found = False
                for p in number_patterns: 
                    matched = p["matcher"].match(runlength_token)
                    if matched!=None:
                        found = True
                if found:
                    #print runlength_token, "WILDCARD"
                    runlength_token='*'

                filter_mask[tpos] = 1
                filter_words[tpos] = runlength_token

                if debug_mode:
                    print("\033[0;36mAdding single-valued column to the filter\033[0m [tpos:"+str(tpos)+"]", "->"+runlength_token+"<-", runlength)
                #print "\033[0;36mAdding single-valued column to the filter\033[0m [tpos:"+str(tpos)+"]", "->"+runlength_token+"<-", runlength

                token_added_order.append(runlength_token)
                count_added_order.append(1) 

        if 0 not in filter_mask: 
            if debug_mode:
                print("Exiting loop since all filters are determined.", filter_mask)
                print("->filter_words:", filter_words)
            break

        if len(max_runlen_positions)>1:
            max_runlen_pos = max_runlen_positions[_next_randint(0,len(max_runlen_positions)-1)]
        elif len(max_runlen_positions)==1:
            max_runlen_pos = max_runlen_positions[0]

        if max_runlen_pos==-1: 
            print("ERROR: max column not selected!!!")
            sys.exit(0)

        if max_runlen_pos>=0: 

            if debug_mode:
                print("max_column:"+str(max_runlen_pos)+",\033[35;47mCalling determine_filter_word ...\033[0m")
            target_dict = all_column_dict[max_runlen_pos] 
            filled = float(sum(filter_mask))/float(len(filter_mask)) 
            new_fword, pv, cr = determine_filter_word(target_dict, len(filtered_logs), filled)
            #print "{0:.5f}".format(pv), "{0:.5f}".format(cr), "---->"+new_fword

            if debug_mode:
                print("max_column:"+str(max_runlen_pos)+",\033[35;47mdetermine_filter_word returned:\033[0m", "->"+new_fword+"<-")
                print("*** Max token from each column ***")
                for h in all_column_dict: # h is a column position
                    d = all_column_dict[h]
                    for n in sorted(d, key=lambda k: d[k], reverse=True):
                        print("["+str(h)+"]",d[n], "\t","->"+n+"<-")
                        break
            if debug_mode and ' ' not in target_dict and '=' not in target_dict:
                print("------------------------------------------------------------")
                print("[Column:"+str(max_runlen_pos)+"]", "\033[1;91mpval:",pv,"\033[0m", "Cardinality:", "{0:.2f}".format(cr),"%")
                print("------------------------------------------------------------")
                print("num  |   count   |       token      ")
                print("------------------------------------------------------------")
                # print each line
                cnt = 1
                for n in sorted(target_dict, key=lambda k: target_dict[k], reverse=True):
                    print("["+str(cnt)+"]\t",target_dict[n], "\t\t","->"+n+"<-")
                    cnt += 1
                    if cnt>40:
                        print("...")
                        break

# we want to see new fword
#            print('new fword: ' + new_fword)
#            print("pv: " + str(pv))

            if new_fword=="*":
                # Handle wildcards not caused by the p-value test.
                if pv<(1.0 + UNIFORM_THRESHOLD) / 2.0:
                    filter_mask[max_runlen_pos] = 1
                    filter_words[max_runlen_pos] = "*"
                    token_added_order.append("*")
                    count_added_order.append(len(target_dict))
                else:
                    tw,tm = finalize_filter_with_star(filter_words,filter_mask)
                    log_template = generate_log_template_star(tw,False)
                    #print "log template:", log_template
                    if exist_match(log_template, rep_logs)>=0:
                        new_fword = sorted(target_dict, key=lambda k: target_dict[k], reverse=True)[0]
                    else:
                        filter_words,filter_mask = finalize_filter_with_star(filter_words,filter_mask)

            else:
                if pv > UNIFORM_THRESHOLD - UNIFORM_EPSILON and pv <= (1.0 + UNIFORM_THRESHOLD) / 2.0:
                    add_candidate = True
                else:
                    add_candidate = False

                if add_candidate:
                    tw,tm = finalize_filter_with_star(filter_words,filter_mask)
                    log_template = generate_log_template_star(tw,False)
                    #print "log template:", log_template
                    #print "\033[1;95mCandidate:", "\033[0m \033[90;102m", "".join(tw), "\033[0m "

                    if exist_match(log_template, rep_logs)<0:
                        candidate_set.append(log_template)
                        #print "\033[1;94mCandidate ACCEPTED", "\033[0m ", "\033[1;95mCandidate:", "\033[0m \033[90;102m", "".join(tw), "\033[0m "
                    else:
                        #print "\033[1;91mCandidate REJECTED", "\033[0m ","\033[1;95mCandidate:", "\033[0m \033[90;102m", "".join(tw), "\033[0m "
                        pass

            if new_fword!="*": 
                filter_mask[max_runlen_pos] = 1
                filter_words[max_runlen_pos] = new_fword
                token_added_order.append(new_fword)
                count_added_order.append(len(target_dict))
            #print "\033[0;35mNew filter word\033[0m [tpos:"+str(max_runlen_pos)+"]", "=>"+new_fword+"<="

        if debug_mode:
            print("Current column_cnt:",column_cnt)
            print("Max runlength percent:", "{0:.2f}".format(max_runlen_pct),"%")
            print("Max runlength percent position:", max_runlen_pos)
            print("\033[1;94mMax runlength percent word :", "->"+filter_words[max_runlen_pos]+"<-\033[0m")
            print("sum of valid_mask:", sum(valid_mask))
            print("filter_mask(sum:"+str(sum(filter_mask))+"/"+str(len(filter_mask))+"):","".join(str(x) for x in filter_mask))
            print("Filtering logs using filter_words:",filter_words, len(filter_words))

            #print "Added order:", "\033[1;95m|\033[0m".join(token_added_order)
            #print "Cardinality order:", "\033[1;95m|\033[0m".join(count_added_order)
            print("\033[1;95mToken list in the added order:(The number is the count of unique tokens.)\033[0m")
            for w in range(0,len(token_added_order)):
                print("        ", format(count_added_order[w],'3d'),token_added_order[w])

            input("\033[0;35m->Press ENTER to continue filtering ...\033[0m")
            print(" ")

        #print "filter_vect(sum:"+str(sum(filter_mask))+"/"+str(len(filter_mask))+"):","".join(str(x) for x in filter_mask)
        if debug_mode:
            print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    # END of inner while loop

    elapsed = time.time() - tm_checkpt
    tm005 += elapsed
    #print "{0:.3f}".format(elapsed),"Filter construction going through all columns"

    # All filter_mask is filled now.
    # Generate a log template
    #print "Callling generate_log_template_star() 3"
    template_words = filter_words
    if MISSING_TOKEN in template_words:
        template_words = template_words[:template_words.index(MISSING_TOKEN)]
    log_template = generate_log_template_star(template_words,True)

    candidate_set.append(log_template)

    # Identical templates represent the same Tree branch.
    return list(dict.fromkeys(candidate_set))


# Longest common subsequence
def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])
    return lcs_set


def _pair_lcs_length_sum(template_a, template_b, pair_cache):
    key_a = tuple(template_a)
    key_b = tuple(template_b)
    # Use the same cache key regardless of template order.
    if key_b < key_a:
        cache_key = (key_b, key_a)
    else:
        cache_key = (key_a, key_b)

    if cache_key in pair_cache:
        return pair_cache[cache_key]

    token_d = {}
    n = 0
    for tok in key_a + key_b:
        if tok in token_d:
            continue
        token_d[tok] = chr(n)
        n += 1

    str_a = "".join(token_d[tok] for tok in key_a)
    str_b = "".join(token_d[tok] for tok in key_b)
    result = sum(len(item) for item in lcs(str_a,str_b))

    pair_cache[cache_key] = result
    return result


# logtem is a list of 'count','template' dict with tokenized static templates
def compute_slcpl(logtem, pair_cache):
    selected = []
    for t in logtem:
        selected.append({
            "count": t["count"],
            "template": [s for s in t["template"] if s != " "],
        })

    SL_sum = 0
    total_log_count = 0 
    for t in selected:
        static_length = len(t['template'])
        #print "  \033[1;94m", static_length, "\033[0m", "\033[33;36m","".join(t['template']), "\033[0m"
        SL_sum += (static_length * t['count'])
        total_log_count += t['count']
    if total_log_count==0:
        print("Error: total_log_count 0")
        sys.exit(0)
    SL = float(SL_sum)/float(total_log_count) 
    print("\033[0;32mAverage weighted SL:", SL, "\033[0m")

    total_cpl = 0
    weighted_cpl_sum = 0.0
    for i in range(0,len(selected)):
        set_len_sum = 0
        for j in range(0,len(selected)):
            if i==j: 
                continue

            set_len_sum += _pair_lcs_length_sum(
                selected[i]['template'],
                selected[j]['template'],
                pair_cache,
            )

        set_len_sum = float(set_len_sum)/float(len(selected))

#        set_len_sum = float(set_len_sum)/float(len(logtem)) # divide by the log template count because template i is compared with all the rest

        #print "\n",i,"set_len_sum=",set_len_sum
        #print "    weighted:", float(set_len_sum*selected[i]['count'])/float(total_log_count)
        weighted_cpl_sum += float(set_len_sum*selected[i]['count'])/float(total_log_count) # weighted sum of cpl

        sys.stdout.write('\r'+"Processed "+"{0:.1f}".format(100.0*float(i)/float(len(selected)))+"%")
        sys.stdout.flush()

    print(" ")
    if len(selected)>1:
        CPL = float(total_cpl)/float(len(selected))/float(len(selected)-1)
    else:
        CPL=0.0
    #print "\033[1;95mAverage CPL:", CPL, "\033[0m"
    #print "\033[0;32mInverse Average max CPL:", 1.0/(1.0+CPL), "\033[0m"
    #print "\033[0;103mScore:", SL/(1.0+CPL), "\033[0m"

    return SL,weighted_cpl_sum



def tokenize_template_for_slcl(template):
    static_template = template.replace(".*","")
    unescaped_template = regex.sub("\\\\","",static_template)
    return unescaped_template.split()


def _get_template_coverage(template,logs,coverage_cache):
    # Return this final regex's full-corpus match bitmap and raw match count.
    if template in coverage_cache:
        return coverage_cache[template]
    matcher = regex.compile("^"+template+"$")
    coverage_bytes = bytearray((len(logs)+7)//8)
    match_count = 0
    for log_index,log in enumerate(logs):
        if matcher.match(log) is None:
            continue
        coverage_bytes[log_index//8] |= 1 << (log_index%8)
        match_count += 1
    result = (int.from_bytes(coverage_bytes,byteorder="little"),match_count)
    coverage_cache[template] = result
    return result


def _build_effective_template_set(log_templates,logs,coverage_cache):
    candidates = []
    for t in log_templates:
        if t is None:
            continue
        template = str(t["template"])
        coverage,match_count = _get_template_coverage(template,logs,coverage_cache)
        if match_count>0:
            candidates.append({"template":template,"coverage":coverage,"match_count":match_count})
    candidates.sort(key=lambda candidate:(candidate["match_count"],candidate["template"]),reverse=True)
    selected = []
    covered = 0
    matched_count = 0
    for candidate in candidates:
        newly_covered = candidate["coverage"] & ~covered
        newly_matched_count = bin(newly_covered).count("1")
        if newly_matched_count==0:
            continue
        selected.append({"count":newly_matched_count,"template":candidate["template"]})
        covered |= candidate["coverage"]
        matched_count += newly_matched_count
    return selected,matched_count


class LeafScore:

    def __init__(self,node,selected,sum_matched,log_count,sl,cpl,score):
        self.node = node
        self.selected = selected
        self.sum_matched = sum_matched
        self.log_count = log_count
        self.sl = sl
        self.cpl = cpl
        self.score = score

    @property
    def remaining_count(self):
        return self.log_count-self.sum_matched


def _evaluate_leaf(leaf_node, logs, score_cache, pair_cache, scoring_token_cache, coverage_cache):
    # Rebuild the effective set from full-corpus regex coverage.
    selected,sum_matched = _build_effective_template_set(leaf_node.log_templates,logs,coverage_cache)

    if len(selected)==0:
        return None

    score_key = tuple(sorted((t["count"],t["template"]) for t in selected))
    if score_cache is not None and score_key in score_cache:
        SL,CPL = score_cache[score_key]
    else:
        # Tokenize static template parts for SL/CPL scoring.
        scoring_templates = []
        for t in selected:
            template = t['template']
            if template not in scoring_token_cache:
                scoring_token_cache[template] = tuple(tokenize_template_for_slcl(template))
            scoring_templates.append({
                "count":t["count"],
                "template":scoring_token_cache[template],
            })

        # Compute the SL and CPL scores for this leaf.
        SL,CPL = compute_slcpl(scoring_templates,pair_cache)
        if score_cache is not None:
            score_cache[score_key] = (SL,CPL)
    return LeafScore(leaf_node,selected,sum_matched,len(logs),SL,CPL,SL-CPL)


def _select_best_leaf(tree, logs):
    leaf_nodes = []
    for leaf_node in tree.nodes:
        if not leaf_node.is_leaf_node():
            continue
        if -1 in leaf_node.all_vect:
            print("Skipping unfinished leaf:", leaf_node.name)
            continue
        leaf_nodes.append(leaf_node)

    print("Completed leaf count:",len(leaf_nodes))

    best_result = None
    score_cache = {}
    pair_cache = {}
    scoring_token_cache = {}
    coverage_cache = {}
    for leaf_index, leaf_node in enumerate(leaf_nodes):

        print(
            "Evaluating leaf",
            str(leaf_index+1)+"/"+str(len(leaf_nodes))+":"
        )
        leaf_node.print_node()
        result = _evaluate_leaf(leaf_node,logs,score_cache,pair_cache,scoring_token_cache,coverage_cache)
        if result is None:
            print("Skipping leaf with no effective templates:", leaf_node.name)
            continue

        print("    Sum of matched logs:", result.sum_matched)
        print("   ",result.remaining_count,"logs remaining.")
        print("    Initial template count:", len(leaf_node.log_templates))
        print("    Selected template count:", len(result.selected))
        print("    SL= "+str(result.sl))
        print("    CPL= "+str(result.cpl))
        print("    score= "+str(result.score))

        if best_result is None or result.score>best_result.score:
            best_result = result

    print("Cached leaf score sets:",len(score_cache))
    print("Cached template-pair scores:",len(pair_cache))
    print("Cached template coverages:",len(coverage_cache))

    return best_result


template_token_cache = {}


def tokenize_log_template(s):
    if s in template_token_cache:
        return list(template_token_cache[s])

    tok = custom_split(regex.sub("\\\\","",s))
    for y in range(0,len(tok)):
        if '~' in tok[y]:
            tok[y]=".*"
    template_token_cache[s] = tok
    return list(template_token_cache[s])



########################## ########################## ########################## ##########################
########################## ########################## ########################## ##########################

log_templates = []

prepopulated_log_templates = [
"DEBUG ovsdbapp.backend.ovs_idl.vlog \[\-\] \[POLLIN\] on fd .* {{\(pid=.*\) __log_wakeup .*}}",
"DEBUG neutron.plugins.ml2.drivers.openvswitch.agent.ovs_neutron_agent \[None .* None None\] Agent rpc_loop \- iteration:.* started {{\(pid=.*\) rpc_loop .*}}",
"DEBUG neutron.plugins.ml2.drivers.openvswitch.agent.ovs_neutron_agent \[None .* None None\] Agent rpc_loop \- iteration:.* completed. Processed ports statistics: {'regular': {'updated': .* 'added': .* 'removed': .* Elapsed:.* {{\(pid=.*\) loop_count_and_wait .*}}",
"DEBUG neutron.plugins.ml2.drivers.openvswitch.agent.openflow.native.ofswitch \[None .* None None\] ofctl request version=.*,msg_type=.*,msg_len=.*,xid=.*,OFPFlowStatsRequest\(cookie=.*,cookie_mask=.*,flags=.*,match=.*\(oxm_fields={.*}\),out_group=.*,out_port=.*,table_id=.*,type=.*\) result \[OFPFlowStatsReply\(body=\[OFPFlowStats\(byte_count=.*,cookie=.*,duration_nsec=.*,duration_sec=.*,flags=.*,hard_timeout=.*,idle_timeout=.*,instructions=\[.*\],length=.*,match=.*\(oxm_fields={.*",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task L3NATAgentWithStateReport.periodic_sync_routers_task {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task VolumeManager._publish_service_capabilities {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._check_instance_build_time {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG neutron_lib.callbacks.manager \[None .* None None\] Notify callbacks \['neutron.services.segments.db._update_segment_host_mapping_for_agent\-\-9223363269425746616'\] for agent, after_update {{\(pid=.*\) _notify_loop .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager.update_available_resource {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task VolumeManager._report_driver_status {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._poll_rebooting_instances {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG keystone.policy.backends.rules \[None .* None None\] enforce identity:validate_token: {'is_delegated_auth': False, 'access_token_id': None, 'user_id': u'aa24c04c8a1649ad8b3edf176357288d', 'roles': \[u'service', u'admin'\], 'user_domain_id': u'default', 'consumer_id': None, 'trustee_id': None, 'is_domain': False, 'is_admin_project': True, 'trustor_id': None, 'token': <KeystoneToken \(audit_id=.*, audit_chain_id=.*\) at .* 'project_id': u'fca0da50f8ad44d3a24cb1e4e6dd1478', 'trust_id': None, 'project_domain_id': u'default'} {{\(pid=.*\) enforce .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._instance_usage_audit {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._poll_unconfirmed_resizes {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task SchedulerManager._expire_reservations {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._reclaim_queued_deletes {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_concurrency.lockutils \[\-\] Lock \"_check_child_processes\" acquired by \"neutron.agent.linux.external_process._check_child_processes\" :: waited .* {{\(pid=.*\) inner .*}}",
"DEBUG nova.api.openstack.placement.requestlog \[None .* service placement\] Starting request: .* \"GET .* {{\(pid=.*\) __call__ .*}}",
"DEBUG keystone.middleware.auth \[None .* None None\] Authenticating user token {{\(pid=.*\) process_request .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task SchedulerManager._run_periodic_tasks {{\(pid=.*\) run_periodic_tasks .*}}",
"INFO keystone.common.wsgi \[None .* None None\] GET .*",
"INFO nova.api.openstack.placement.requestlog \[None .* service placement\] .* \"GET .* status: .* len: .* microversion: .*",
"DEBUG keystone.middleware.auth \[None .* None None\] RBAC: auth_context: {'is_delegated_auth': False, 'access_token_id': None, 'user_id': u'aa24c04c8a1649ad8b3edf176357288d', 'roles': \[u'service', u'admin'\], 'user_domain_id': u'default', 'consumer_id': None, 'trustee_id': None, 'is_domain': False, 'is_admin_project': True, 'trustor_id': None, 'token': <KeystoneToken \(audit_id=.*, audit_chain_id=.*\) at .* 'project_id': u'fca0da50f8ad44d3a24cb1e4e6dd1478', 'trust_id': None, 'project_domain_id': u'default'} {{\(pid=.*\) fill_context .*}}",
"DEBUG keystone.common.authorization \[None .* None None\] RBAC: Authorization granted {{\(pid=.*\) check_policy .*}}",
"DEBUG keystone.common.authorization \[None .* None None\] RBAC: Authorizing identity:validate_token\(\) {{\(pid=.*\) _build_policy_check_credentials .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._poll_rescued_instances {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._heal_instance_info_cache {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG nova.scheduler.client.report \[None .* None None\] Refreshing aggregate associations for resource provider .* {{\(pid=.*\) _ensure_resource_provider .*}}",
"DEBUG oslo_concurrency.lockutils \[None .* None None\] Lock \"compute_resources\" acquired by \"nova.compute.resource_tracker._update_available_resource\" :: waited .* {{\(pid=.*\) inner .*}}",
"DEBUG oslo_concurrency.lockutils \[\-\] Lock \"_check_child_processes\" released by \"neutron.agent.linux.external_process._check_child_processes\" :: held .* {{\(pid=.*\) inner .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._poll_volume_usage {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_concurrency.lockutils \[None .* None None\] Lock \"compute_resources\" released by \"nova.compute.resource_tracker._update_available_resource\" :: held .* {{\(pid=.*\) inner .*}}",
"DEBUG cinder.manager \[None .* None None\] Notifying Schedulers of capabilities ... {{\(pid=.*\) _publish_service_capabilities .*}}",
"DEBUG nova.compute.manager \[None .* None None\] CONF.reclaim_instance_interval <= .* skipping... {{\(pid=.*\) _reclaim_queued_deletes .*}}",
"DEBUG neutron.db.agents_db \[None .* None None\] Agent healthcheck: found .* active agents {{\(pid=.*\) agent_health_check .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._sync_scheduler_instance_info {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG nova.compute.manager \[None .* None None\] Rebuilding the list of instances to heal {{\(pid=.*\) _heal_instance_info_cache .*}}",
"DEBUG oslo_concurrency.lockutils \[None .* None None\] Lock \"host_instance\" acquired by \"nova.scheduler.host_manager.sync_instance_info\" :: waited .* {{\(pid=.*\) inner .*}}",
"DEBUG oslo_concurrency.processutils \[None .* None None\] CMD \"sudo cinder\-rootwrap .* env LC_ALL=.* vgs \-\-noheadings \-\-unit=.* \-o name,size,free,lv_count,uuid \-\-separator : \-\-nosuffix stack\-volumes\-lvmdriver\-1\" returned: .* in .* {{\(pid=.*\) execute .*}}",
"DEBUG oslo_concurrency.processutils \[None .* None None\] CMD \"sudo cinder\-rootwrap .* env LC_ALL=.* lvs \-\-noheadings \-\-unit=.* \-o vg_name,name,size \-\-nosuffix stack\-volumes\-lvmdriver\-1\" returned: .* in .* {{\(pid=.*\) execute .*}}",
"DEBUG oslo_concurrency.processutils \[None .* None None\] Running cmd \(subprocess\): sudo cinder\-rootwrap .* env LC_ALL=.* vgs \-\-noheadings \-\-unit=.* \-o name,size,free,lv_count,uuid \-\-separator : \-\-nosuffix stack\-volumes\-lvmdriver\-1 {{\(pid=.*\) execute .*}}",
"DEBUG nova.compute.resource_tracker \[None .* None None\] Compute_service record updated for triton5:triton5 {{\(pid=.*\) _update_available_resource .*}}",
"DEBUG cinder.scheduler.host_manager \[None .* None None\] Received volume service update from triton5@lvmdriver\-1: {u'filter_function': None, u'goodness_function': None, u'volume_backend_name': u'lvmdriver\-1', u'driver_version': u'3.0.0', u'sparse_copy_volume': False, u'pools': \[{u'pool_name': u'lvmdriver\-1', u'filter_function': None, u'goodness_function': None, u'total_volumes': .* u'multiattach': False, u'provisioned_capacity_gb': .* u'allocated_capacity_gb': .* u'thin_provisioning_support': False, u'free_capacity_gb': .* u'location_info': u'LVMVolumeDriver:triton5:stack\-volumes\-lvmdriver\-1:default:.*', u'total_capacity_gb': .* u'thick_provisioning_support': True, u'reserved_percentage': .* u'QoS_support': False, u'max_over_subscription_ratio': .* u'vendor_name': u'Open Source', u'storage_protocol': u'iSCSI'} {{\(pid=.*\) update_service_capabilities .*}}",
"DEBUG oslo_concurrency.processutils \[None .* None None\] Running cmd \(subprocess\): sudo cinder\-rootwrap .* env LC_ALL=.* lvs \-\-noheadings \-\-unit=.* \-o vg_name,name,size \-\-nosuffix stack\-volumes\-lvmdriver\-1 {{\(pid=.*\) execute .*}}",
"DEBUG cinder.volume.drivers.lvm \[None .* None None\] Updating volume stats {{\(pid=.*\) _update_volume_stats .*}}",
"DEBUG nova.compute.resource_tracker \[None .* None None\] Hypervisor/Node resource view: name=.* free_ram=.* free_disk=.* free_vcpus=.* pci_devices=\[{\"dev_id\": .* \"product_id\": .* \"dev_type\": \"type\-PCI\", \"numa_node\": null, \"vendor_id\": .* \"label\": .* \"address\": \".*\"}, {\"dev_id\": .* \"product_id\": .* \"dev_type\": \"type\-PCI\", \"numa_node\": null, \"vendor_id\": .* \"label\": .* \"address\": \".*\"}, {\"dev_id\": .* \"product_id\": .* \"dev_type\": \"type\-PCI\", \"numa_node\": null, \"vendor_id\": .* \"label\": .* \"address\": \".*\"}, {\"dev_id\": .* \"product_id\": .* \"dev_type\": \"type\-PCI\", \"numa_node\": null, \"vendor_id\": .* \"label\": .* \"address\": \".*",
"DEBUG nova.compute.manager \[None .* None None\] Didn't find any instances for network info cache update. {{\(pid=.*\) _heal_instance_info_cache .*}}",
"DEBUG nova.compute.resource_tracker \[None .* None None\] Auditing locally available compute resources for triton5 \(node: triton5\) {{\(pid=.*\) update_available_resource .*}}",
"DEBUG nova.compute.resource_tracker \[None .* None None\] Hypervisor: free VCPUs: .* {{\(pid=.*\) _report_hypervisor_resource_view .*}}",
"INFO nova.compute.resource_tracker \[None .* None None\] Final resource view: name=.* phys_ram=.* used_ram=.* phys_disk=.* used_disk=.* total_vcpus=.* used_vcpus=.*",
"DEBUG nova.compute.resource_tracker \[None .* None None\] Total usable vcpus: .* total allocated vcpus: .* {{\(pid=.*\) _report_final_resource_view .*}}",
"DEBUG nova.compute.manager \[None .* None None\] Starting heal instance info cache {{\(pid=.*\) _heal_instance_info_cache .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._run_pending_deletes {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._cleanup_incomplete_migrations {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_concurrency.lockutils \[None .* None None\] Lock \"host_instance\" released by \"nova.scheduler.host_manager.sync_instance_info\" :: held .* {{\(pid=.*\) inner .*}}",
"INFO nova.scheduler.host_manager \[None .* None None\] Successfully synced instances from host 'triton5'.",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._poll_bandwidth_usage {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG nova.compute.manager \[None .* None None\] Cleaning up deleted instances {{\(pid=.*\) _run_pending_deletes .*}}",
"DEBUG keystone.common.fernet_utils \[None .* None None\] Loaded .* Fernet keys from .* but `\[fernet_tokens\] max_active_keys = 3`; perhaps there have not been enough key rotations to reach `max_active_keys` yet\? {{\(pid=.*\) load_keys .*}}",
"DEBUG nova.compute.manager \[None .* None None\] There are .* instances to clean {{\(pid=.*\) _run_pending_deletes .*}}",
"DEBUG nova.compute.manager \[None .* None None\] Cleaning up deleted instances with incomplete migration {{\(pid=.*\) _cleanup_incomplete_migrations .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._sync_power_states {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_concurrency.lockutils \[None .* None None\] Lock \"storage\-registry\-lock\" acquired by \"nova.virt.storage_users.do_get_storage_users\" :: waited .* {{\(pid=.*\) inner .*}}",
"DEBUG oslo_concurrency.lockutils \[None .* None None\] Lock \"storage\-registry\-lock\" released by \"nova.virt.storage_users.do_get_storage_users\" :: held .* {{\(pid=.*\) inner .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._cleanup_running_deleted_instances {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG oslo_concurrency.lockutils \[None .* None None\] Lock \"storage\-registry\-lock\" acquired by \"nova.virt.storage_users.do_register_storage_use\" :: waited .* {{\(pid=.*\) inner .*}}",
"INFO keystone.common.wsgi \[None .* None None\] POST .*",
"DEBUG keystone.middleware.auth \[None .* None None\] There is either no auth token in the request or the certificate issuer is not trusted. No auth context will be set. {{\(pid=.*\) fill_context .*}}",
"DEBUG keystone.auth.core \[None .* None None\] MFA Rules not processed for user `aa24c04c8a1649ad8b3edf176357288d`. Rule list: .* \(Enabled: `True`\). {{\(pid=.*\) check_auth_methods_against_rules .*}}",
"DEBUG ovsdbapp.backend.ovs_idl.vlog \[\-\] tcp:.*: entering ACTIVE {{\(pid=.*\) _transition .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._run_image_cache_manager_pass {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG ovsdbapp.backend.ovs_idl.vlog \[\-\] 0\-ms timeout {{\(pid=.*\) __log_wakeup .*}}",
"DEBUG nova.virt.libvirt.imagecache \[None .* None None\] Skipping verification, no base directory at .* {{\(pid=.*\) _get_base .*}}",
"DEBUG oslo_concurrency.lockutils \[None .* None None\] Lock \"storage\-registry\-lock\" released by \"nova.virt.storage_users.do_register_storage_use\" :: held .* {{\(pid=.*\) inner .*}}",
"DEBUG ovsdbapp.backend.ovs_idl.vlog \[\-\] tcp:.*: entering IDLE {{\(pid=.*\) _transition .*}}",
"DEBUG ovsdbapp.backend.ovs_idl.vlog \[\-\] tcp:.*: idle .* ms, sending inactivity probe {{\(pid=.*\) run .*}}",
"DEBUG oslo_service.periodic_task \[None .* None None\] Running periodic task ComputeManager._poll_shelved_instances {{\(pid=.*\) run_periodic_tasks .*}}",
"DEBUG keystone.auth.core \[None .* None None\] MFA Rules not processed for user `65190d3514a143c291095441cfc4d3a8`. Rule list: .* \(Enabled: `True`\). {{\(pid=.*\) check_auth_methods_against_rules .*}}",
"DEBUG ovsdbapp.backend.ovs_idl.vlog \[\-\] 4995\-ms timeout {{\(pid=.*\) __log_wakeup .*",
"DEBUG ovsdbapp.backend.ovs_idl.vlog \[\-\] 4996\-ms timeout {{\(pid=.*\) __log_wakeup .*",
]

prepopulated_log_templates = []
#rep_logs = [] 

reuse_filename="CODE50_REUSE.p"
reuse_logfilename="CODE50_REUSE_LOG.p"

# Discovery execution backend.  Keep multiprocessing coordination in this
# section so the CLI, candidate construction, matching, and leaf scoring stay
# independent of the execution strategy.
def run_tree_discovery(tree, log_dataset, all_tlogs, linear_mode, worker_count=1, parallel_backend="process"):
    """Run template discovery with the selected execution strategy.

    The original Tree backend remains the default execution path.
    """
    if worker_count > 1 and not linear_mode:
        if parallel_backend == "thread":
            return _run_parallel_tree_discovery_thread(tree, log_dataset, all_tlogs, worker_count)
        else:
            return _run_parallel_tree_discovery(tree, log_dataset, all_tlogs, worker_count)
    if worker_count > 1:
        print("--linear has no candidate branches; using the sequential discovery backend.")
    return _run_sequential_tree_discovery(tree, log_dataset, all_tlogs, linear_mode)


def _run_sequential_tree_discovery(tree, log_dataset, all_tlogs, linear_mode):
    """Run the existing Tree-based Sequential Covering search to completion."""
    cur_node = tree.find_inprogress_node()
    next_pattern_index = len(discovered_patterns)
    runtime_checkpt = time.time()

    while -1 in cur_node.all_vect:

        sampled_logs, tokenized_logs = sample_by_token_length_and_space_count(log_dataset.logs, all_tlogs, cur_node.all_vect, log_dataset.log_scores, cur_node.score_counts, log_dataset.log_score_indices)
        #sampled_logs = random_sample_logs(all_logs, RANDOM_SAMPLE_SIZE)
        #sampled_logs = sample_by_length(all_logs, all_vect, RANDOM_SAMPLE_SIZE)
        #sampled_logs = sample_by_signature(all_logs, RANDOM_SAMPLE_SIZE)
        if len(sampled_logs)==0:
            break
        if debug_mode:
            print("\033[93;100mSampled_logs size:", len(sampled_logs),"\033[0m ")
            for i in range(0,len(sampled_logs)):
                print("    [Sampled]", sampled_logs[i])
                print("             ", "".join(tokenized_logs[i]))

        #tokenized_logs = do_tokenization(sampled_logs)
        #sampled_logs,tokenized_logs = sample_by_term_correlation(all_logs, RANDOM_SAMPLE_SIZE)

        if debug_mode:
            print("Number of logs:",len(sampled_logs))
            print("Number of tokenized logs:",len(tokenized_logs))

        #apply_all_patterns(tokenized_logs)
        replace_known_patterns(tokenized_logs)


        #if len(cur_node.log_templates)==167:
        #    debug_mode = True

    
        candidate_set = construct_candidate_log_templates(tokenized_logs,cur_node.rep_logs) # returns a list of candidate log templates
        next_pattern_index = apply_new_patterns(all_tlogs, next_pattern_index)

#        log_template = candidate_set[0]
#        removed_count = mark_matched_logs(all_logs, cur_node.all_vect, cur_node.rep_logs, log_template, len(cur_node.log_templates))
#        print "["+format(len(cur_node.log_templates),'3d')+"]", format(cur_node.all_vect.count(-1),'5d'), format(removed_count,'4d'), log_template
#        cur_node.log_templates.append({"count":removed_count,"template":log_template})
#        #raw_templates.append(postprocess_raw_template(filter_words))

        if (linear_mode):
            tval = 999999
        else:
            tval = 1

        if len(candidate_set)>tval:
            for log_template in candidate_set:
                print("\033[93;100mCandidate:"+"\033[0m \033[90;103m", log_template, "\033[0m ")
            
                conflicted = exist_match(log_template, cur_node.rep_logs)
                if conflicted>=0:
                    print("ERROR<1>: log_template overlaps with one of the already-found log templates.")
                    sys.exit(0)

                new_name = 'n'+str(tree.serial)
                new_identifier  = 'i'+str(tree.serial)
                tree.create_node( new_name, log_dataset.log_count, new_identifier, parent=cur_node.identifier)
                tree.serial += 1
    
                new_node = tree.find_node(new_identifier)
                # copy internal state
                new_node.log_templates = list(cur_node.log_templates)
                new_node.rep_logs = list(cur_node.rep_logs)
                new_node.all_vect = list(cur_node.all_vect)
                new_node.score_counts = dict(cur_node.score_counts)

                new_node.print_node()
    
                # Mark matched logs from all_logs using new log template. 
                removed_count = mark_matched_logs(log_dataset.logs, new_node.all_vect, new_node.rep_logs, log_template, len(new_node.log_templates), new_node.score_counts, log_dataset.log_scores)
                if removed_count==0:
                    print("\n\033[1;94mWARNING[1]:\033[0m No logs removed from the template!")
                    print("TEMPLATE->",log_template)
                    print("Remaining logs:", log_dataset.log_count-sum(1 for x in cur_node.all_vect if x>0))
                    print("Log template count:", len(cur_node.log_templates))
                    for p in standalone_patterns:
                        print(p['label'],p['pattern'])
                    sys.exit(0)
                    break
                print("["+format(len(new_node.log_templates),'3d')+"]", format(new_node.all_vect.count(-1),'5d'), format(removed_count,'4d'), log_template)
                # attach new candidate log templates
                new_node.log_templates.append({"count":removed_count,"template":log_template})
                print("\033[1;34m",tree.show("top"),"...\033[0m")
        else: 

            log_template = candidate_set[0]

            tok_candi = tokenize_log_template(log_template) # tok_candi: tokenized candidate
            merged_template_indices = []
            printed = False
            for i in range(0,len(cur_node.log_templates)):
                if cur_node.log_templates[i]==None:
                    continue
                tok_logtm = tokenize_log_template(cur_node.log_templates[i]["template"]) # tok_lt: tokenized log template

                # Compare token by token to see if they have only 1 difference
                if len(tok_candi)==len(tok_logtm):
                    diff_count = 0
                    diff_loc = -1
                    for j in range(0,len(tok_candi)):
                        if tok_candi[j]!=tok_logtm[j]:
                            diff_count+=1
                            diff_loc = j
                            if diff_count==2:
                                break
                    if diff_count==1:
                        if not printed:
                            print("**\033[0;32m", "".join(tok_candi), "\033[0m")
                            printed = True
                        print("**\033[0;35m", "".join(tok_logtm), "\033[0m")

                        print("   Token to update:", tok_candi[diff_loc])
                        print("   Token to update:", tok_logtm[diff_loc])
                        tok_logtm[diff_loc]=".*"
                        log_template = generate_log_template_star(tok_logtm,True)
                        tok_candi = tokenize_log_template(log_template)
                        print("   New log_template:", log_template)
                        merged_template_indices.append(i)
                        cur_node.log_templates[i]=None
                        cur_node.rep_logs[i]=None

            removed_count = mark_matched_logs(log_dataset.logs, cur_node.all_vect, cur_node.rep_logs, log_template, len(cur_node.log_templates), cur_node.score_counts, log_dataset.log_scores)
            if removed_count==0:
                print("\n\033[1;95mWARNING[2]:\033[0m \033[1;31mNo logs removed from the template!", "\033[0m")
                print("   *TEMPLATE->",log_template)
                print("   *Remaining logs:", log_dataset.log_count-sum(1 for x in cur_node.all_vect if x>0))
                print("   *Log template count:", len(cur_node.log_templates))
                for p in standalone_patterns:
                    print("   ",p['label'],p['pattern'])
                sys.exit(0)
                continue
                #break

            #print "\n\033[0;32m"+t['template']+"\033[0m"
            #if "Memory usage of ProcessTree" in log_template:
            #if "addStoredBlock: blockMap updated:" in log_template:
            #if "INFO org.mortbay.log:" in log_template:
            #    print "["+format(len(cur_node.log_templates),'3d')+"]", format(cur_node.all_vect.count(-1),'5d'), format(removed_count,'4d'), "\033[0;32m"+log_template+"\033[0m"
            #else:
            #    print "["+format(len(cur_node.log_templates),'3d')+"]", format(cur_node.all_vect.count(-1),'5d'), format(removed_count,'4d'), log_template
            #print "["+format(len(cur_node.log_templates),'3d')+"]", format(cur_node.all_vect.count(-1),'5d'), format(removed_count,'4d'), log_template
            #print "\""+re.sub("\"","\\\"",log_template)+"\","
            # Transfer logs from merged templates to the new template index.
            previously_matched_count = 0
            for j in range(0,len(cur_node.all_vect)):
                if cur_node.all_vect[j] in merged_template_indices:
                    cur_node.all_vect[j] = len(cur_node.log_templates)
                    previously_matched_count += 1

            removed_count += previously_matched_count
            print(len(cur_node.log_templates), removed_count, "\033[0;34m"+log_template+"\033[0m")

            #sys.exit(0)

            if removed_count>0:
                cur_node.log_templates.append({"count":removed_count,"template":log_template})
            #raw_templates.append(postprocess_raw_template(filter_words))

        if cur_node.is_leaf_node() and -1 in cur_node.all_vect: 
            pass
        else:
            cur_node = tree.find_inprogress_node()
            if cur_node==None:
                break
            print("\033[1;94m=> Switching to ",cur_node.name,"\033[0m")

        if debug_mode:
            input("\033[1;94m->Press ENTER to continue ...\033[0m")

        #print "============================================================================================================="
        #raw_input("\033[1;94m->Press ENTER to continue ...\033[0m")

    print("Final custom pattern count:", len(discovered_patterns))
    return time.time() - runtime_checkpt


# Branch-level multiprocessing backend. This is reached only from the dispatcher
# above; the original sequential Tree backend remains untouched.
class BranchState:

    def __init__(self, branch_path=()):
        self.branch_path = branch_path
        self.discovered_patterns = []
        self.seqnum = 1
        self.random_state = None


class BranchResult:
    """Outcome returned by one branch task run in a multiprocessing Pool worker."""

    def __init__(self, branch_status, node=None, branch_state=None, candidate_set=None, failure_traceback=None, timers=None):
        self.branch_status = branch_status
        self.node = node
        self.branch_state = branch_state
        self.candidate_set = candidate_set
        self.failure_traceback = failure_traceback
        self.timers = timers


def _reset_discovery_timers():
    global tm001, tm002, tm003, tm004, tm005, tm006, tm007, tm008, tm009, tm010, tm011
    tm001 = tm002 = tm003 = tm004 = tm005 = tm006 = tm007 = tm008 = tm009 = tm010 = tm011 = 0.0


def _discovery_timer_snapshot():
    return (tm001, tm002, tm003, tm004, tm005, tm006, tm007, tm008, tm009, tm010, tm011)


def _add_discovery_timers(totals, values):
    for index, value in enumerate(values):
        totals[index] += value


# TODO: Remove this global-state adapter after legacy discovery functions accept branch_state directly.
def _activate_branch_state(branch_state):
    global discovered_patterns, seqnum
    discovered_patterns = copy.deepcopy(branch_state.discovered_patterns)
    seqnum = branch_state.seqnum
    random.setstate(branch_state.random_state)


# TODO: Remove with _activate_branch_state after legacy discovery functions accept branch_state directly.
def _capture_branch_state(branch_state):
    branch_state.discovered_patterns = copy.deepcopy(discovered_patterns)
    branch_state.seqnum = seqnum
    branch_state.random_state = random.getstate()


def _read_tokenized_log_sample_from_store(tokenized_log_store, sample_indices):
    """Load only the selected tokenized logs from the temporary file by original log index."""
    global tm006
    tm_checkpt = time.time()
    tokenized_logs = []
    for log_index in sample_indices:
        tokenized_logs.append(tokenized_log_store.read(log_index))
    elapsed = time.time() - tm_checkpt
    tm006 += elapsed
    return tokenized_logs


# TODO: Combine this with sample_by_token_length_and_space_count() when both paths use the same token-log storage.
def sample_by_token_length_and_space_count_multiprocessing(log_dataset, branch_state, node, tokenized_log_store):
    most_popular = max(node.score_counts, key=node.score_counts.get)
    sample_indices = []
    for log_index in log_dataset.log_score_indices[most_popular]:
        if node.all_vect[log_index] == -1 and log_dataset.log_scores[log_index] == most_popular:
            sample_indices.append(log_index)
            if len(sample_indices) >= 1000:
                break
    sampled_logs = [log_dataset.logs[log_index] for log_index in sample_indices]
    tokenized_logs = _read_tokenized_log_sample_from_store(tokenized_log_store, sample_indices)
    return sampled_logs, tokenized_logs


# Merge patterns discovered by one branch into the shared global list, deduplicated by regex.
def _merge_branch_patterns(global_patterns, branch_patterns):
    known_patterns = {pattern["pattern"] for pattern in global_patterns}
    for pattern in branch_patterns:
        if pattern["pattern"] not in known_patterns:
            global_patterns.append(copy.deepcopy(pattern))
            known_patterns.add(pattern["pattern"])
    global_patterns.sort(key=lambda pattern:pattern["pattern"])


def _apply_candidate_to_branch(node, branch_state, log_template, log_dataset):
    if exist_match(log_template, node.rep_logs) >= 0:
        raise RuntimeError("candidate overlaps with an existing template: "+str(log_template))
    removed_count = mark_matched_logs(log_dataset.logs, node.all_vect, node.rep_logs, log_template, len(node.log_templates), node.score_counts, log_dataset.log_scores)
    if removed_count == 0:
        raise RuntimeError("split candidate removed no logs in branch "+str(branch_state.branch_path)+": "+str(log_template))
    node.log_templates.append({"count":removed_count,"template":log_template})
    print("["+format(len(node.log_templates)-1,'3d')+"]", format(node.all_vect.count(-1),'5d'), format(removed_count,'4d'), log_template)


def _apply_linear_candidate(node, branch_state, log_template, log_dataset):
    tok_candi = tokenize_log_template(log_template)
    merged_template_indices = []
    for index, old_template in enumerate(node.log_templates):
        if old_template is None:
            continue
        tok_logtm = tokenize_log_template(old_template["template"])
        if len(tok_candi) != len(tok_logtm):
            continue
        diff_locations = [position for position in range(len(tok_candi)) if tok_candi[position] != tok_logtm[position]]
        if len(diff_locations) != 1:
            continue
        tok_logtm[diff_locations[0]] = ".*"
        log_template = generate_log_template_star(tok_logtm,True)
        tok_candi = tokenize_log_template(log_template)
        merged_template_indices.append(index)
        node.log_templates[index] = None
        node.rep_logs[index] = None
    removed_count = mark_matched_logs(log_dataset.logs, node.all_vect, node.rep_logs, log_template, len(node.log_templates), node.score_counts, log_dataset.log_scores)
    if removed_count == 0:
        raise RuntimeError("linear candidate removed no logs in branch "+str(branch_state.branch_path)+": "+str(log_template))
    for position, template_index in enumerate(node.all_vect):
        if template_index in merged_template_indices:
            node.all_vect[position] = len(node.log_templates)
            removed_count += 1
    node.log_templates.append({"count":removed_count,"template":log_template})
    print(len(node.log_templates)-1, removed_count, "\033[0;34m"+log_template+"\033[0m")


def _run_branch_until_split_or_leaf(node, branch_state, log_dataset, tokenized_log_store):
    _activate_branch_state(branch_state)
    while -1 in node.all_vect:
        sampled_logs, tokenized_logs = sample_by_token_length_and_space_count_multiprocessing(log_dataset, branch_state, node, tokenized_log_store)
        if len(sampled_logs) == 0:
            break
        if debug_mode:
            print("\033[93;100mSampled_logs size:", len(sampled_logs),"\033[0m ")
            for i in range(0,len(sampled_logs)):
                print("    [Sampled]", sampled_logs[i])
                print("             ", "".join(tokenized_logs[i]))
            print("Number of logs:",len(sampled_logs))
            print("Number of tokenized logs:",len(tokenized_logs))
        replace_known_patterns(tokenized_logs)
        apply_new_patterns_multiprocessing(tokenized_logs)
        candidate_set = construct_candidate_log_templates(tokenized_logs, node.rep_logs)
        if len(candidate_set) == 0:
            raise RuntimeError("No candidate generated for branch "+str(branch_state.branch_path))
        if len(candidate_set) > 1:
            _capture_branch_state(branch_state)
            return candidate_set
        _apply_linear_candidate(node, branch_state, candidate_set[0], log_dataset)
        if debug_mode:
            input("\033[1;94m->Press ENTER to continue ...\033[0m")
    _capture_branch_state(branch_state)
    if not node.is_leaf_node() or -1 in node.all_vect:
        raise RuntimeError("Branch stopped before covering all logs: "+str(branch_state.branch_path))
    return None


def _rebuild_tree_from_leaves(tree, log_count, leaves):
    tree.nodes = []
    tree.serial = 0
    root_node = tree.create_node("TOP", log_count, "top")
    leaves_by_path = {tuple(leaf["branch_path"]):leaf["node"] for leaf in leaves}
    identifiers = {():root_node.identifier}
    # Walk each leaf from root to leaf; reuse shared parents created by earlier paths.
    for leaf_path in sorted(leaves_by_path):
        for depth in range(1,len(leaf_path)+1):
            branch_path = leaf_path[:depth]
            if branch_path in identifiers:
                continue
            identifier = "i"+str(tree.serial)
            tree.create_node("n"+str(tree.serial), log_count, identifier, parent=identifiers[branch_path[:-1]])
            identifiers[branch_path] = identifier
            tree.serial += 1
    for leaf_path, leaf_node in leaves_by_path.items():
        node = tree.find_node(identifiers[leaf_path])
        node.log_templates = leaf_node.log_templates
        node.rep_logs = leaf_node.rep_logs
        node.all_vect = leaf_node.all_vect
        node.score_counts = leaf_node.score_counts


# Each Pool worker initializes these once before it starts processing branch tasks.
pool_log_dataset = None
pool_tokenized_log_store = None


def _run_parallel_pool_worker_branch(node, branch_state, candidate, candidate_index):
    try:
        _reset_discovery_timers()
        branch_state.branch_path = branch_state.branch_path+(candidate_index,)
        _activate_branch_state(branch_state)
        _apply_candidate_to_branch(node, branch_state, candidate, pool_log_dataset)
        _capture_branch_state(branch_state)
        candidate_set = _run_branch_until_split_or_leaf(node, branch_state, pool_log_dataset, pool_tokenized_log_store)
        if candidate_set is None:
            return BranchResult("leaf", node=node, branch_state=branch_state, timers=_discovery_timer_snapshot())
        return BranchResult("split", node=node, branch_state=branch_state, candidate_set=candidate_set, timers=_discovery_timer_snapshot())
    except BaseException:
        return BranchResult("failure", branch_state=branch_state, failure_traceback=traceback.format_exc(), timers=_discovery_timer_snapshot())


def _run_parallel_tree_discovery(tree, log_dataset, all_tlogs, worker_count):
    context = multiprocessing.get_context("fork")

    def initialize_worker(log_dataset, tokenized_log_store):
        global pool_log_dataset, pool_tokenized_log_store
        pool_log_dataset = log_dataset
        pool_tokenized_log_store = tokenized_log_store
        pool_tokenized_log_store.open()

    with tempfile.TemporaryDirectory(prefix="lognroll-token-store-") as token_directory:
        tokenized_log_store = TokenizedLogFileStore.write(all_tlogs, token_directory)
        # Free the parent's full token table after every tokenized log has been written to the temporary file.
        all_tlogs.clear()
        gc.collect()

        root_node = tree.find_node("top")
        root_branch_state = BranchState()
        root_branch_state.discovered_patterns = copy.deepcopy(discovered_patterns)
        root_branch_state.seqnum = seqnum
        root_branch_state.random_state = random.getstate()
        _reset_discovery_timers()
        runtime_checkpt = time.time()
        candidate_set = _run_branch_until_split_or_leaf(root_node, root_branch_state, log_dataset, tokenized_log_store)
        timer_totals = list(_discovery_timer_snapshot())
        leaves = []
        failures = []
        global_patterns = copy.deepcopy(root_branch_state.discovered_patterns)
        if candidate_set is None:
            leaves.append({"branch_path":root_branch_state.branch_path,"node":root_node})
        else:
            # Close the parent's file handle so every worker opens its own handle and keeps its own read position.
            tokenized_log_store.close()

            event_queue = queue.Queue()
            # Keep branch tasks in the parent until a Pool worker is available instead of filling Pool's internal queue.
            pending_branches = deque((root_node, root_branch_state, candidate, candidate_index) for candidate_index, candidate in enumerate(candidate_set))
            pool = context.Pool(worker_count, initialize_worker, (log_dataset, tokenized_log_store))
            pending_count = 0

            # Submit pending branches until every available worker slot is occupied.
            def submit_available_branches():
                nonlocal pending_count
                while pending_count < worker_count and len(pending_branches) > 0:
                    node, branch_state, candidate, candidate_index = pending_branches.popleft()
                    branch_state.discovered_patterns = copy.deepcopy(global_patterns)
                    pool.apply_async(_run_parallel_pool_worker_branch, (node, branch_state, candidate, candidate_index), callback=event_queue.put, error_callback=lambda error:event_queue.put(BranchResult("failure", failure_traceback=repr(error), timers=(0.0,)*11)))
                    pending_count += 1

            try:
                submit_available_branches()
                while pending_count > 0:
                    result = event_queue.get()
                    pending_count -= 1
                    _add_discovery_timers(timer_totals, result.timers)
                    if result.branch_status == "leaf":
                        _merge_branch_patterns(global_patterns, result.branch_state.discovered_patterns)
                        leaves.append({"branch_path":result.branch_state.branch_path,"node":result.node})
                    elif result.branch_status == "split":
                        _merge_branch_patterns(global_patterns, result.branch_state.discovered_patterns)
                        result.branch_state.discovered_patterns = copy.deepcopy(global_patterns)
                        pending_branches.extend((result.node, result.branch_state, candidate, candidate_index) for candidate_index, candidate in enumerate(result.candidate_set))
                    else:
                        failures.append(result)
                        break
                    submit_available_branches()
            finally:
                if len(failures) > 0:
                    pool.terminate()
                else:
                    pool.close()
                pool.join()
        tokenized_log_store.close()
    if len(failures) > 0:
        failure = failures[0]
        failure_path = "unknown" if failure.branch_state is None else failure.branch_state.branch_path
        raise RuntimeError("Parallel branch failed at "+str(failure_path)+"\n"+failure.failure_traceback)
    discovered_patterns[:] = copy.deepcopy(global_patterns)
    print("Final custom pattern count:", len(discovered_patterns))
    _rebuild_tree_from_leaves(tree, log_dataset.log_count, leaves)
    global tm001, tm002, tm003, tm004, tm005, tm006, tm007, tm008, tm009, tm010, tm011
    tm001, tm002, tm003, tm004, tm005, tm006, tm007, tm008, tm009, tm010, tm011 = timer_totals
    return time.time()-runtime_checkpt


# Thread-pool branch backend (--parallel-backend thread). Threads share the parent's
# memory directly, so unlike the process backend above there is no pickling boundary:
# all_tlogs never needs to be file-backed (branch tasks just index the in-memory list),
# but every task also needs its own private deep copy of `node`/`branch_state` before
# it starts running, since the process backend got that isolation for free from pickling
# at submission time and threads do not.
def _read_tokenized_log_sample_from_memory(all_tlogs, sample_indices):
    return [all_tlogs[log_index] for log_index in sample_indices]


def sample_by_token_length_and_space_count_thread(log_dataset, branch_state, node, all_tlogs):
    most_popular = max(node.score_counts, key=node.score_counts.get)
    sample_indices = []
    for log_index in log_dataset.log_score_indices[most_popular]:
        if node.all_vect[log_index] == -1 and log_dataset.log_scores[log_index] == most_popular:
            sample_indices.append(log_index)
            if len(sample_indices) >= 1000:
                break
    sampled_logs = [log_dataset.logs[log_index] for log_index in sample_indices]
    tokenized_logs = _read_tokenized_log_sample_from_memory(all_tlogs, sample_indices)
    return sampled_logs, tokenized_logs


def _activate_thread_local_branch_state(branch_state):
    _thread_local_discovery.active = True
    _thread_local_discovery.discovered_patterns = branch_state.discovered_patterns
    _thread_local_discovery.seqnum = branch_state.seqnum
    rng = random.Random()
    rng.setstate(branch_state.random_state)
    _thread_local_discovery.rng = rng


def _capture_thread_local_branch_state(branch_state):
    branch_state.discovered_patterns = _thread_local_discovery.discovered_patterns
    branch_state.seqnum = _thread_local_discovery.seqnum
    branch_state.random_state = _thread_local_discovery.rng.getstate()
    _thread_local_discovery.active = False


def _run_branch_until_split_or_leaf_thread(node, branch_state, log_dataset, all_tlogs):
    _activate_thread_local_branch_state(branch_state)
    while -1 in node.all_vect:
        sampled_logs, tokenized_logs = sample_by_token_length_and_space_count_thread(log_dataset, branch_state, node, all_tlogs)
        if len(sampled_logs) == 0:
            break
        replace_known_patterns(tokenized_logs)
        apply_new_patterns_multiprocessing(tokenized_logs)
        candidate_set = construct_candidate_log_templates(tokenized_logs, node.rep_logs)
        if len(candidate_set) == 0:
            raise RuntimeError("No candidate generated for branch "+str(branch_state.branch_path))
        if len(candidate_set) > 1:
            _capture_thread_local_branch_state(branch_state)
            return candidate_set
        _apply_linear_candidate(node, branch_state, candidate_set[0], log_dataset)
    _capture_thread_local_branch_state(branch_state)
    if not node.is_leaf_node() or -1 in node.all_vect:
        raise RuntimeError("Branch stopped before covering all logs: "+str(branch_state.branch_path))
    return None


def _run_parallel_thread_worker_branch(log_dataset, all_tlogs, node, branch_state, candidate, candidate_index):
    try:
        branch_state.branch_path = branch_state.branch_path+(candidate_index,)
        _activate_thread_local_branch_state(branch_state)
        _apply_candidate_to_branch(node, branch_state, candidate, log_dataset)
        _capture_thread_local_branch_state(branch_state)
        candidate_set = _run_branch_until_split_or_leaf_thread(node, branch_state, log_dataset, all_tlogs)
        if candidate_set is None:
            return BranchResult("leaf", node=node, branch_state=branch_state, timers=(0.0,)*11)
        return BranchResult("split", node=node, branch_state=branch_state, candidate_set=candidate_set, timers=(0.0,)*11)
    except BaseException:
        return BranchResult("failure", branch_state=branch_state, failure_traceback=traceback.format_exc(), timers=(0.0,)*11)


def _run_parallel_tree_discovery_thread(tree, log_dataset, all_tlogs, worker_count):
    root_node = tree.find_node("top")
    root_branch_state = BranchState()
    root_branch_state.discovered_patterns = copy.deepcopy(discovered_patterns)
    root_branch_state.seqnum = seqnum
    root_branch_state.random_state = random.getstate()
    runtime_checkpt = time.time()
    candidate_set = _run_branch_until_split_or_leaf_thread(root_node, root_branch_state, log_dataset, all_tlogs)
    leaves = []
    failures = []
    global_patterns = copy.deepcopy(root_branch_state.discovered_patterns)
    if candidate_set is None:
        leaves.append({"branch_path":root_branch_state.branch_path,"node":root_node})
    else:
        event_queue = queue.Queue()
        pending_branches = deque((root_node, root_branch_state, candidate, candidate_index) for candidate_index, candidate in enumerate(candidate_set))
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=worker_count)
        pending_count = 0

        # Threads share memory with the parent, so (unlike Pool.apply_async, which
        # implicitly deep-copies via pickling) node/branch_state must be explicitly
        # deep-copied per task here -- otherwise sibling branch tasks spawned from the
        # same split would mutate the same node/branch_state object concurrently.
        def submit_available_branches():
            nonlocal pending_count
            while pending_count < worker_count and len(pending_branches) > 0:
                node, branch_state, candidate, candidate_index = pending_branches.popleft()
                task_node = copy.deepcopy(node)
                task_branch_state = copy.deepcopy(branch_state)
                task_branch_state.discovered_patterns = copy.deepcopy(global_patterns)
                future = executor.submit(_run_parallel_thread_worker_branch, log_dataset, all_tlogs, task_node, task_branch_state, candidate, candidate_index)
                def on_done(future):
                    error = future.exception()
                    if error is None:
                        event_queue.put(future.result())
                    else:
                        event_queue.put(BranchResult("failure", failure_traceback=repr(error), timers=(0.0,)*11))
                future.add_done_callback(on_done)
                pending_count += 1

        try:
            submit_available_branches()
            while pending_count > 0:
                result = event_queue.get()
                pending_count -= 1
                if result.branch_status == "leaf":
                    _merge_branch_patterns(global_patterns, result.branch_state.discovered_patterns)
                    leaves.append({"branch_path":result.branch_state.branch_path,"node":result.node})
                elif result.branch_status == "split":
                    _merge_branch_patterns(global_patterns, result.branch_state.discovered_patterns)
                    result.branch_state.discovered_patterns = copy.deepcopy(global_patterns)
                    pending_branches.extend((result.node, result.branch_state, candidate, candidate_index) for candidate_index, candidate in enumerate(result.candidate_set))
                else:
                    failures.append(result)
                    break
                submit_available_branches()
        finally:
            # Python threads cannot be force-killed: on failure this only stops
            # not-yet-started tasks, already-running branch threads finish in the background.
            executor.shutdown(wait=True, cancel_futures=len(failures) > 0)
    if len(failures) > 0:
        failure = failures[0]
        failure_path = "unknown" if failure.branch_state is None else failure.branch_state.branch_path
        raise RuntimeError("Parallel branch failed at "+str(failure_path)+"\n"+failure.failure_traceback)
    discovered_patterns[:] = copy.deepcopy(global_patterns)
    print("Final custom pattern count:", len(discovered_patterns))
    print("Note: thread backend does not track per-phase timing breakdown; only wall-clock total is meaningful.")
    _rebuild_tree_from_leaves(tree, log_dataset.log_count, leaves)
    return time.time()-runtime_checkpt


if __name__ == '__main__':
    debug_mode = False
    openfile_list = []
    try:
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--logfile',  type=argparse.FileType('r'), nargs='+', required=True, help='List of one or more input log files')
        parser.add_argument('--debug',  action='store_true', required=False, help='When specified, it walks through each log processing and print out messages.')
        parser.add_argument('--linear',  action='store_true', required=False, help='Whether to follow linear execution path along the tree or not.')
        parser.add_argument('--workers', type=int, default=1, help='Maximum concurrent candidate branches (default: 1).')
        parser.add_argument('--parallel-backend', choices=['process', 'thread'], default='process', help='Execution backend for --workers > 1 (default: process).')
        parser.add_argument('--seed', type=int, required=False, help='Seed Python and NumPy random generators for reproducible discovery.')
        # Now, we don't need clean mode
        parser.add_argument('--clean',  action='store_true', required=False, help='When specified, it deletes intermediate pickle files of tokenized log data and reprocess them. It takes longer.')

        args = parser.parse_args()
        args.clean = True
        if args.workers < 1:
            parser.error('--workers must be at least 1')
        if args.debug and args.workers > 1:
            parser.error('--debug cannot be used with --workers > 1')
        if args.seed is not None:
            random.seed(args.seed)
            numpy.random.seed(args.seed)
            print('Random seed:',args.seed)
        openfile_list = args.logfile
        if len(openfile_list)>1:
            print("Specify only one log file. Currently",len(openfile_list),"are given.")
            sys.exit(0)

        if args.debug==False:
            debug_mode = False
        else:
            debug_mode = bool(args.debug)

        if args.linear==False:
            linear_mode = False
        else:
            linear_mode = bool(args.linear)

        if os.path.exists(reuse_logfilename):
            prev_reuse_logfilename = pickle.load(open(reuse_logfilename,"r"))
        else:
            prev_reuse_logfilename = "-"

        print("** Previous log file:",prev_reuse_logfilename)
        print("**    Input log file:",openfile_list[0].name)

        clean_mode = False
        if args.clean==False:
            if prev_reuse_logfilename==openfile_list[0].name:
               clean_mode = False
               print("\033[2;102mNon-Clean (cache reuse, fast) mode\033[0m")
            else:
                print("\033[31;91mAlthough you wanted FAST REUSE mode, the input log file is different from the previous run. Forcing clean mode ... \033[0m")
                print("\033[37;101mClean (slow) mode\033[0m")
                clean_mode = True
                os.remove(reuse_logfilename)
                pickle.dump(openfile_list[0].name, open(reuse_logfilename,"w"))
        else:
            print("\033[37;101mClean (slow) mode\033[0m")
            clean_mode = bool(args.clean)

    except Exception as e:
        print(('Error: %s' % str(e)))

    print("Loading all logs into memory.")
    raw_logs = read_log_files( openfile_list, None )

    old_log_count = len(raw_logs)
    rep_logs = remove_log_template_matches(raw_logs, prepopulated_log_templates)
    raw_log_count = len(raw_logs)
    print("==================================================================================================================================")
    print("Old Log count:",old_log_count)
    print("New Log count:",len(raw_logs))
    print("Prepopulated log template count:", len(prepopulated_log_templates))
    print("Representative logs count:", len(rep_logs))
    log_templates = copy.deepcopy(prepopulated_log_templates)

    print("Preprocessing logs...")
    all_logs = preprocess_known_patterns(raw_logs)
    raw_logs = None

    if clean_mode:
        if os.path.exists(reuse_filename):
            os.remove(reuse_filename)

    if os.path.exists(reuse_filename):
        print("Reusing reuse file ...")
        all_tlogs = pickle.load(open(reuse_filename,"rb"))
    else:
        print("Sequence number:", seqnum)
        print("Tokenizing all logs.")
        all_tlogs = do_tokenization(all_logs)
        print("Done tokenizing.", len(all_logs))

        #apply_all_patterns(all_tlogs)
        print("Rearranging numbers of known patterns to make values unique ...")
        uniquify_numbers(all_tlogs)
        print("Done rearranging values.")
        print("len(all_tlogs):",len(all_tlogs))

        pickle.dump(all_tlogs, open(reuse_filename,"wb"))

    # Maps each log index to its immutable score.
    # Example: all_log_scores[7] == 3004; mark_matched_logs() decrements score_counts[3004].
    all_log_scores = build_log_scores(all_logs)
    # Maps each score to its original log indices.
    # Example: all_log_score_indices[3004] == [1, 7]; sampling uses it for most_popular.
    all_log_score_indices = build_log_score_index(all_log_scores)
    log_dataset = LogDataset(raw_log_count, all_logs, all_log_scores, all_log_score_indices)

    tree = Tree()
    root_node = tree.create_node("TOP", log_dataset.log_count, "top")
    root_node.score_counts = dict(Counter(all_log_scores))

    discovery_elapsed = run_tree_discovery(tree, log_dataset, all_tlogs,  linear_mode, args.workers, args.parallel_backend)
    parallel_mode = args.workers > 1 and not linear_mode
    if parallel_mode:
        print("Timing mode: aggregate worker work; wall-clock runtime is reported below.")
    print("{0:8.3f}".format(tm001), "Apply all patterns")
    print("{0:8.3f}".format(tm002), "Apply new patterns")
    print("{0:8.3f}".format(tm003), "Random sampling logs")
    print("{0:8.3f}".format(tm006), "Sampling by split token length")
    print("{0:8.3f}".format(tm007), "Sampling by signature made of special characters")
    print("{0:8.3f}".format(tm008), "Sampling by term correlation analysis")
    print("{0:8.3f}".format(tm009), "Sampling by term filtering")
    print("{0:8.3f}".format(tm010), "Replenish term band")
    print("{0:8.3f}".format(tm004), "Tokenizing logs")
    print("{0:8.3f}".format(tm005), "Filtering all columns")
    print("{0:8.3f}".format(tm011), "Match and remove logs")
    if parallel_mode:
        print("     n/a", "Unaccounted (parallel overlap)")
    else:
        print("{0:8.3f}".format(discovery_elapsed-tm001-tm002-tm003-tm004-tm005-tm006-tm007-tm008-tm009-tm010-tm011), "Unaccounted")
    print("{0:8.3f}".format(discovery_elapsed), "\033[0;103mTemplate discovery runtime\033[0m")

    print("\033[1;34m",tree.show("top"),"...\033[0m")
    print(" ")

    scoring_checkpt = time.time()
    best_result = _select_best_leaf(tree,log_dataset.logs)
    scoring_elapsed = time.time() - scoring_checkpt
    if best_result is None:
        print("ERROR: No completed leaf has an effective template set.", file=sys.stderr)
        sys.exit(1)

    print("{0:8.3f}".format(scoring_elapsed), "\033[0;103mLeaf scoring runtime\033[0m")
    print("{0:8.3f}".format(discovery_elapsed+scoring_elapsed), "\033[0;103mDiscovery plus scoring runtime\033[0m")

    best_node = best_result.node
    print("Best leaf:", best_node.name, "["+best_node.identifier+"]")
    print("Best leaf SL:", best_result.sl)
    print("Best leaf CPL:", best_result.cpl)
    print("Best leaf score:", best_result.score)
    print("Final template count:", len(best_result.selected))
    for t in sorted(best_result.selected, key=lambda k: k["count"], reverse=True):
        print(t["count"],t["template"])
