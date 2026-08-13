#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Log preprocessing: mask the variable parts of raw log lines.

Rules (lifted from lognroll_actual.py) come in two flavours:

  LINE_RULES  matched anywhere inside a line -- URLs, emails, file paths, IP
              addresses and so on, which span delimiters and so have to be
              taken out before anything else looks at the line.
  TOKEN_RULE  matched against a whole word -- UUIDs, dates, numbers. Matching
              the whole word is what keeps the "123" inside "blk_123" intact;
              a word here is a run of text bounded by DELIMITERS, brackets or
              the ends of the line.

Everything a rule matches is replaced with the wildcard ".*"; the rest of the
line is left untouched.

Usage:
    python lognroll_preprocess.py -f <logfile> [-d]

Masked lines are echoed to stdout in normal mode. Every run also writes its own
directory so repeated runs never overwrite each other:

    preprocess_logs/<date>/<dataset>/<time>/
        masked.log        every line, with the matches replaced
        masked_debug.log  -d only: the console MASKED/MATCH view

Paths and counts go to stderr, which keeps stdout a clean stream that can be
piped.
"""

import os
import sys
import argparse
import contextlib
import unicodedata
from datetime import datetime
import regex

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

# Applied to the whole line, before tokenizing.
LINE_PATTERNS = [
    r"hdfs://([a-zA-Z0-9_\-\.\*:\+]+/)+([a-zA-Z0-9_\-\.\*:\+]*(\?[a-zA-Z0-9_\-:\+]+=[a-zA-Z0-9_\-:\+]+(&[a-zA-Z0-9_\-:\+]+=[a-zA-Z0-9_\-:\+]+)*)?)",
    r"hdfs://([a-zA-Z0-9_\-\.]+):([0-9]+)",
    r"http://([a-zA-Z0-9_\-\.\*:\+]+/)+([a-zA-Z0-9_\-\.\*:\+]*(\?[a-zA-Z0-9_\-:\+]+=[a-zA-Z0-9_\-:\+]+(&[a-zA-Z0-9_\-:\+]+=[a-zA-Z0-9_\-:\+]+)*)?)",
    r"http://([a-zA-Z0-9_\-\.]+):([0-9]+)",
    r"https://([a-zA-Z0-9_\-\.\*:\+]+/)+([a-zA-Z0-9_\-\.\*:\+]*(\?[a-zA-Z0-9_\-:\+]+=[a-zA-Z0-9_\-:\+]+(&[a-zA-Z0-9_\-:\+]+=[a-zA-Z0-9_\-:\+]+)*)?)",
    r"https://([a-zA-Z0-9_\-\.]+):([0-9]+)",
    r"[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+",
    # email
    r"http://([a-zA-Z0-9_\-\.]+):([0-9]+)(/[a-zA-Z0-9_\-\.\*\+\-]+)+",
    r"/(var|tmp|home|usr|home|etc|opt|gogo|airwordcount|wikimean|wikimedian|wikistandarddeviation|cluster|jobhistory|node|ws)(/[a-zA-Z0-9_\-\.\*\+\-]+)+",
    # known file path
    r"(?<![a-zA-Z0-9_\-\.\*\+])/(?:[a-zA-Z0-9_\-\.\*\+]+/)*[a-zA-Z0-9_\-\.\*\+]+",  # any file path
    r"(?<![^ ,;|:<>=/@(){}\[\]<>'\"])[^ ,;|:<>=/@(){}\[\]<>'\"]*\d+\.\d+\.\d+\.\d+(?::\d+)?[^ ,;|:<>=/@(){}\[\]<>'\"]*(?![^ ,;|:<>=/@(){}\[\]<>'\"])",  # whole token containing an IP address
    r"\-?\d+\.\d+ (KB|GB|MB)",  # 21.5 MB
    r"\-?\d+ (KB|GB|MB)",  # 5 GB
]

# Applied to a single token; the whole token has to match.
TOKEN_PATTERNS = [
    r"CID\-[a-zA-Z0-9]{8}\-[a-zA-Z0-9]{4}\-[a-zA-Z0-9]{4}\-[a-zA-Z0-9]{4}\-[a-zA-Z0-9]{12}",  # CID-<UUID>
    r"[\da-zA-Z]{8}\-[\da-zA-Z]{4}\-[\da-zA-Z]{4}\-[\da-zA-Z]{4}\-[\da-zA-Z]{12}",  # UUID
    r"req\-[a-z0-9]{8}\-[a-z0-9]{4}\-[a-z0-9]{4}\-[a-z0-9]{4}\-[a-z0-9]{12}",  # req-<UUID>
    r"\d{4}\-\d{2}\-\d{2}",  # date, YYYY-MM-DD
    r"\-?\d+",  # integer
    r"\-?\d+\.\d+",  # floating point
    r"\-?\d+(ms|msec|millisec|s|sec|second|seconds|us|microsec|KiB|GiB|MB|KB|GB|%)",  # integer with unit
    r"\-?\d+\.\d+(ms|msec|millisec|s|sec|second|seconds|us|microsec|KiB|GiB|MB|KB|GB|%)",  # float with unit
    r"\-?\d+\^\d+",  # exponent
    r"0x[\da-fA-F]+",  # hexadecimal
]

WILDCARD = ".*"

# Every run gets its own directory, preprocess_logs/<date>/<dataset>/<time>/,
# so runs never overwrite each other and one run's files stay together.
OUTPUT_ROOT = "preprocess_logs"
OUTPUT_NAME = "masked.log"
DEBUG_OUTPUT_NAME = "masked_debug.log"

# Characters that end a word. Anything between two of them (or a line end) is
# what a token rule has to match in full.
DELIMITERS = " ,;|:<>=/@"
BRACKETS = "(){}[]<>'\""
BOUNDARY = "".join(regex.escape(c) for c in sorted(set(DELIMITERS + BRACKETS)))

LINE_RULES = [regex.compile(p) for p in LINE_PATTERNS]

# One pass over the line replaces every whole word a rule matches. The
# lookarounds are what "whole word" means: the character on either side must be
# a boundary, so "blk_123" is left alone while "(123)" is not. They also force
# the alternation to match a word in full -- "12" inside "12.5ms" fails the
# lookahead and the engine backtracks into the longer alternative.
TOKEN_RULE = regex.compile(
    "(?<![^" + BOUNDARY + "])(?:" + "|".join(TOKEN_PATTERNS) + ")(?![^" + BOUNDARY + "])")

# The combined rule cannot say which alternative fired, which the -d report
# needs. Re-testing a matched word against the patterns in order recovers it:
# the alternation prefers the leftmost alternative that fits, and so does this.
TOKEN_RULES_BY_PATTERN = [(p, regex.compile("^" + p + "$")) for p in TOKEN_PATTERNS]


def matching_token_pattern(word):
    """Return the TOKEN_PATTERNS entry that TOKEN_RULE matched `word` with."""
    for pattern, rule in TOKEN_RULES_BY_PATTERN:
        if rule.match(word) is not None:
            return pattern
    return "<unknown>"


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def replace_rule(line, rule, pattern_for_match, record=None, placements=None):
    """Replace one rule and optionally track each match's final output column."""
    if record is None and placements is None:
        return rule.subn(WILDCARD, line)

    matches = list(rule.finditer(line))
    if not matches:
        return line, 0

    if placements is not None:
        surviving = []
        for position, original in placements:
            wildcard_end = position + len(WILDCARD)
            if any(match.start() < wildcard_end and match.end() > position for match in matches):
                continue
            shift = sum(len(WILDCARD) - (match.end() - match.start()) for match in matches if match.end() <= position)
            surviving.append((position + shift, original))
        placements[:] = surviving

    shift = 0
    for match in matches:
        original = match.group(0)
        pattern = pattern_for_match(original) if callable(pattern_for_match) else pattern_for_match
        if record is not None:
            record.setdefault(pattern, set()).add(original)
        if placements is not None:
            placements.append((match.start() + shift, original))
        shift += len(WILDCARD) - len(original)

    if placements is not None:
        placements.sort(key=lambda item: item[0])
    return rule.sub(WILDCARD, line), len(matches)


def mask_line_patterns(line, record=None, placements=None):
    """Replace every LINE_RULES match in the line with the wildcard.

    Returns (masked_line, number_of_replacements). When `record` is given, each
    match is also filed under the pattern that produced it. `placements`, when
    supplied, receives (final wildcard column, original text) pairs.
    """
    hits = 0
    for rule in LINE_RULES:
        line, count = replace_rule(line, rule, rule.pattern, record, placements)
        hits += count
    return line, hits


def preprocess_line(line, record=None, placements=None):
    """Mask the variable parts of a single log line.

    Returns (masked_line, matched), where matched tells whether any rule fired
    on the line at all. When `record` is given -- a dict of pattern -> set of
    matched strings -- every match is filed there for the -d report.
    """
    line, line_hits = mask_line_patterns(line, record, placements)
    line, token_hits = replace_rule(line, TOKEN_RULE, matching_token_pattern, record, placements)

    return line, (line_hits + token_hits) > 0


def display_width(text):
    """Return the terminal column width of text without third-party helpers."""
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in "WF" else 1
    return width


def match_marker(number):
    """Return a compact circled marker, with a numbered fallback above 20."""
    if number <= 20:
        return chr(0x2460 + number - 1)
    return "〔{0}〕".format(number)


def build_debug_lines(masked, placements):
    """Build one numbered masked line and one non-overlapping match line."""
    masked_parts = []
    annotations = []
    cursor = 0
    masked_width = 0
    for number, (position, original) in enumerate(placements, 1):
        unchanged = masked[cursor:position]
        marker = match_marker(number)
        masked_parts.append(unchanged)
        masked_width += display_width(unchanged)
        annotations.append((masked_width, marker, original))
        marked_wildcard = marker + WILDCARD
        masked_parts.append(marked_wildcard)
        masked_width += display_width(marked_wildcard)
        cursor = position + len(WILDCARD)
    masked_parts.append(masked[cursor:])

    match_parts = []
    match_width = 0
    for target_width, marker, original in annotations:
        start_width = max(target_width, match_width + 1 if match_parts else 0)
        match_parts.append(" " * (start_width - match_width))
        framed = marker + "【" + original + "】"
        match_parts.append(framed)
        match_width = start_width + display_width(framed)
    return "".join(masked_parts), "".join(match_parts)


def print_debug_alignment(masked, placements, stream=sys.stderr):
    """Print the numbered masked line and its matches to one ordered stream."""
    numbered_masked, match_line = build_debug_lines(masked, placements)
    print("MASKED: " + numbered_masked, file=stream, flush=True)
    if match_line:
        print("MATCH : " + match_line, file=stream, flush=True)


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------

def read_log_file(path):
    """Read a log file, collapsing runs of whitespace and dropping blank lines."""
    logs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = " ".join(line.split())
            if line:
                logs.append(line)
    return logs


def build_output_dir(logfile):
    """Create and return this run's own directory.

    The layout is preprocess_logs/<date>/<dataset>/<time>/, where <dataset> is
    the input file name without its extension. The date already scopes the path,
    so the leaf is just the wall-clock time.
    """
    now = datetime.now()
    dataset = os.path.splitext(os.path.basename(logfile))[0]

    parent = os.path.join(OUTPUT_ROOT, now.strftime("%Y-%m-%d"), dataset)

    # The stamp is only accurate to the second, so back-to-back runs would
    # otherwise land in the same directory.
    stem = now.strftime("%H%M%S")
    directory = os.path.join(parent, stem)
    attempt = 1
    while os.path.exists(directory):
        directory = os.path.join(parent, "{0}_{1}".format(stem, attempt))
        attempt += 1

    os.makedirs(directory)
    return directory


def main():
    parser = argparse.ArgumentParser(
        description="Log preprocessing: replace every token matching a rule with " + WILDCARD)
    parser.add_argument("-f", "--logfile", required=True, help="path to the log file to process")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="align each original match below its numbered wildcard and write "
                             + DEBUG_OUTPUT_NAME)
    args = parser.parse_args()

    logs = read_log_file(args.logfile)
    output_dir = build_output_dir(args.logfile)

    with contextlib.ExitStack() as stack:
        out = stack.enter_context(
            open(os.path.join(output_dir, OUTPUT_NAME), "w", encoding="utf-8"))

        debug_out = None
        if args.debug:
            debug_out = stack.enter_context(
                open(os.path.join(output_dir, DEBUG_OUTPUT_NAME), "w", encoding="utf-8"))

        debug_line_count = 0
        for log in logs:
            placements = [] if args.debug else None
            masked, _ = preprocess_line(log, placements=placements)
            print(masked, file=out)

            if args.debug:
                print_debug_alignment(masked, placements)
                print_debug_alignment(masked, placements, debug_out)
                debug_line_count += 1 + bool(placements)
            else:
                print(masked)  # echo to the screen as it goes

    # The summary goes to stderr so that stdout stays a clean stream of lines.
    print("-> " + output_dir, file=sys.stderr)
    print("   {0:<14} {1} lines".format(OUTPUT_NAME, len(logs)), file=sys.stderr)
    if args.debug:
        print("   {0:<14} {1} lines".format(DEBUG_OUTPUT_NAME, debug_line_count), file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # A downstream reader such as `head` closed the pipe. Point stdout at
        # devnull so the interpreter does not complain again while shutting down.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
