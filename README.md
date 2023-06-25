Original repo: https://github.com/LLNL/LULESH

# Usage
Usage: ./lulesh [opts] - where [opts] is one or more of:
- -q              : quiet mode - suppress all stdout
- -i <iterations> : number of cycles to run
- -s <size>       : length of cube mesh along side
- -r <numregions> : Number of distinct regions (def: 11)
- -b <balance>    : Load balance between regions of a domain (def: 1)
- -c <cost>       : Extra cost of more expensive regions (def: 1)
- -f <numfiles>   : Number of files to split viz dump into (def: (np+10)/9)
- -p              : Print out progress
- -v              : Output viz file (requires compiling with -DVIZ_MESH
- -tel            : Number of tasks for element-wise loops (MPC).
- -tnl            : Number of tasks for node-wise loops (MPC).
- -persistent     : Persistent mode
- -h              : This message

# Memory consumption
Memory complexity is O(s^3) - here are a few memory usage records.

| -s value         | 100 | 128 | 150 | 196 |
|------------------|-----|-----|-----|-----|
| Memory used (GB) | 0.9 | 1.8 | 3.3 | 6.1 |

A good estimator is `925 . s**3` bytes.

# Versions
This repository contains 2 versions, which both supports MPI parallelization.

For every versions
- The correctness has been verified using section 4.3 of the LULESH 2.0 report.
- The application has been modified accordingly to the section « 6.3. Changes not to make to LULESH ». That is, the mesh representation, the loop structures, and extra computation were not changed.
- Memory is now being pre-allocated (`Allocate` moved outside of loops) - in order to reduce the footprint on execution time [1].

## BSP / parallel-for
This version uses a fork-join programming model, through OpenMP parallel-for.

## Tasks
The `-tel` and `-tnl` parameters indicates the number of tasks that should respectively decompose an element-wise loop, and a node-wise loop.
To ensure enough parallelism expression, they should be greater than the number of local threads. However, we also want a task grain between 0.1 ms. and 100ms. - otherwise the runtime overhead will prevale.

Here are some ~80% work time scenarios.

| Field                             |           |           |           |
|-----------------------------------|-----------|-----------|-----------|
| -s value                          | 120       | 120       | 120       |
| -i value                          | 16        | 16        | 16        |
| -te value (= -tn)                 | 128       | 256       | 128       |
| -b value                          | 1         | 1         | 4         |
| -c value                          | 1         | 1         | 8         |
| Graph generation (s.)             | 0.4       | 1.4       | 0.5       |
| Graph execution (s.)              | 3.17      | 2.97      | 7.45      |
| Median task duration (ms.)        | 0.50      | 0.25      | 1.42      |
| Number of tasks                   | 42,561    | 84,113    | 52,257    |
| Number of arcs                    | 1,114,345 | 2,259,721 | 1,696,393 |
| Time inside tasks (%)             | 82.3%     | 79.8%     | 82.7%     |
| Time outside tasks - overhead (%) | 12.0%     | 17.7%     | 4.8%      |
| Time outside tasks - idle (%)     | 5.7%      | 2.5%      | 12.4%     |

MPC commit: 63afeae8f9fba5c4ff8bab59dc42fe2d593d9ca1

# References
[1] Tuning the LULESH Mini-app for Current and Future Hardware - Karlin, I and McGraw, J and Keasler, J and Still, B - 2013
