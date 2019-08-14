

This code is associated with the following manuscript:
> Yun William Yu and Griffin Weber. *Federated queries of clinical data
> repositories: balancing accuracy and privacy.* Submitted and under review.

Here we have code to allow distributed hospital queries.

0. Requirements
    Python 3

    Python packages:
        numpy
        pycryptodome
        xxhash

    This code has been tested on Ubuntu 16.04 and 18.04, running
    under Bash 4.4.20.

    Additionally, while RAM requirements will vary with simulation parameters,
    we recommend at least 16 GiB of RAM (and preferably 32 GiB).
        

We allow several different methods including the following:
* Raw counts (possibly with masking)
* Hashed IDs
* HyperLogLog sketches (possibly with masking)
* MPC encrypted raw counts
* MPC encrypted HyperLogLog sketches

The code that a hospital would run is in hospital-bin/

The code that a server would run is in server-bin/

Note that using MPC involves multiple rounds of back and forth communication.

We have example simulation scripts in simulation/

library/ contains the programmatic API
library-tests/ contains unit and integration tests for library/
