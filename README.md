# Lognroll
**source paper**: https://dl.acm.org/doi/abs/10.1145/3464298.3493400

Lognroll is a log parser that generates a log template through iterative filtering. 
It continuously learns one rule that occupies the largest part of the log data set, and generates a log template through iterative filtering that removes the logs targeted by the rule. 


### Running environment

For --parallel-backend process or --workers 1 runs:
python 3.7+
regex 2022.3.2 (or previous version)
numpy
scipy
argparse
pickle

For --parallel-backend thread:
python 3.13+, free-threaded build (e.g. python3.14t)
free-threaded builds of regex, numpy, scipy

### Expected log format for input
The format of logs input to lognroll are expected to follow this.
* Log files should be in plain text.
* Time, date fields are removed at the beginning of each line.
* It is not required, but better to have the line start with log level such as INFO, or DEBUG.
* These are the examples.

```
INFO org.apache.hadoop.hdfs.server.datanode.DataNode: STARTUP_MSG: 
INFO org.apache.hadoop.hdfs.server.datanode.DataNode: registered UNIX signal handlers for [TERM, HUP, INT]
INFO org.apache.hadoop.metrics2.impl.MetricsConfig: loaded properties from hadoop-metrics2.properties
``` 

### How to run
$ python ./lognroll_actual.py --linear --logfile logs/hadoop_clean.log

Template discovery can also branch out across multiple processes (or threads, with --parallel-backend thread) instead of walking the tree one candidate at a time:
$ python ./lognroll_actual.py --logfile logs/hadoop_clean.log --workers 20 --parallel-backend thread

### Execution Parameter
* --linear: Whether to follow linear execution path along the tree or not.
* --clean: When specified, it deletes intermediate pickle files of tokenized log data and reprocess them. It takes longer.
* --debug: When specified, it walks through each log processing and print out messages.
* --logfile: List of one or more input log files
* --workers: Maximum number of candidate tree branches to explore concurrently, each in its own process or thread depending on --parallel-backend (default: 1). Ignored when --linear is set.
* --parallel-backend: Execution backend for --workers > 1: "process" (default) or "thread" (experimental; only worth using under a free-threaded Python build).
* --seed: Seed for Python's and NumPy's random generators, for reproducible runs. Concurrent branches merge newly-discovered patterns as they finish, so results can still vary slightly run to run.

### Default log files
By default, the logs folder has log files extracted from Hadoop and Cassandra, respectively. If you want to input the log files you extracted yourself, you'd better put the timestamp-removed log file.

