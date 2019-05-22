#Distributed Hospital Queries

Here we have code to allow distributed hospital queries.

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
