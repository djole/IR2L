#!/bin/bash

source .virtual/bin/activate
# Uncomment the followinf line if running multiprocessing on Linux systems.
# Replace XXXX with a large number, such as 10000 (if paralel. over 10 cores) and
# 1000000 (if paralel. over 70 cores.
# If the you're getting the "Too many open files" error, try restarting the OS and trying again.

# ulimit -n XXXX

# It's useful to install the "unbuffer" tool on Linux systems
# to get immediate release of stdout into the log file.
# The tool can be installed: sudo apt install expect-dev
# After, add "unbuffer" at the beginning of the next line (without quotes).
mpirun -n 2 python main.py --debug
