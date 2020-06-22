*Important*: this is proof-of-concept academic code, and has not undergone a
security audit. This is especially true of the current incarnation of the
MPC code, which should NOT be used for any sensitive data.

This code is associated with the following manuscript:
> Yun William Yu and Griffin Weber. *Federated queries of clinical data
> repositories: balancing accuracy and privacy.* Submitted and under review.

Here we have code to allow distributed hospital queries.
We allow several different methods including the following:
* Raw counts (possibly with masking)
* Hashed IDs
* HyperLogLog sketches (possibly with masking)
* MPC encrypted raw counts
* MPC encrypted HyperLogLog sketches

Note that this is a partially cleaned up version of the code.
If you are looking for the original code associated with our manuscript, it
can be found in "paper_code_archive/". Be warned though that that code is
very brittle and designed for benchmarking, rather than for ordinary use.

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
        
1. Directory structure

hospital-bin/:  Command-line scripts for a hospital

    count-hospital.py:              summarizes a patient list as a count
    ids-hospital.py:                generates hashed IDs for a patient list
    hll-hospital.py:                summarizes a patient list via HLL
    
    mpc-keygen-hospital.py:         Generates a hospital private partial shared key

    mpc-count-hospital-round1.py:   Generates an encrypted count
    mpc-count-hospital-round2.py:   Takes in an encrypted sum, and produces
                                    a partial decryption using private key
                                    key and a public share to send to the server

    mpc-hll-hospital-round1.py:     Generates an encrypted HLL
    mpc-hll-hospital-round2.py:     Takes in an encrypted HLL, and produces
                                    a partial decryption using private key
    string2num.py:                  Hashes a list of strings to numbers

server-bin/:    Command-line scripts for the central server

    count-server.py:                combines a list of count files into a sum
    ids-server.py:                  takes a list of hashed ID files and finds
                                    the unique union size
    hll-server.py:                  Takes a list of HLL files, combines them,
                                    and reports the approximate union size.

    mpc-keygen-server.py:           Takes a list of hospital partial public
                                    keys and generates a protocol public key.
                                    
    mpc-count-server-round1.py:     Takes a list of encrypted counts, and sums
                                    them.
    mpc-count-server-round2.py:     Takes a list of partial decryptions, and
                                    combines them to output the decrypted sum.

    mpc-hll-server-round1.py:       Takes a list of encrypted HLLs, and
                                    combines them.
    mpc-hll-server-round2.py:       Takes a list of partial decryptions of HLL,
                                    and uses them to output the cardinality of
                                    the decrypted HLL.

library/:       Contains the programmatic API

    elgamal.py:                 code for the ElGamal cryptographic primitives
    hyperloglog.py:             code for HLL sketch functionality
    misc.py:                    miscellaneous I/O helper functions
    elgamal-billion-table.npy:  precomputed log table for decrypting mpc-count
                                (downloaded via setup.sh)

library-tests/: Contains unit and integration tests (via Python unittest)

    test_elgamal.py:        unit and integration tests for elgamal.py
    test_hyperloglog.py:    unit and integration tests for hyperloglog.py
    benchmarking.py:        speed benchmarks for everything
    
simulation/:    Command-line scripts to simulate a hospital network
    
    10_5.txt:               list of numbers 1 to 10,000 to serve as IDs for
                            testing small patient sets. (via setup.sh)
    million_test.txt:       list of numbers 1 to 1,000,000 to serve as IDs
                            for testing large patient sets. (via setup.sh)
    simulation_run.py:      Generates a simulated hospital network
    run_all_but_mpc.sh:     Takes in a simulated hospital network and runs
                            all of the sharing methods except the MPC ones.
    run_mpc_counts.sh:      Takes in a simulated hospital network, and runs
                            MPC-count
    run_mpc_hll.sh:         Takes in a simulated hospital network, and runs
                            MPC-HLL

paper_code_archive/:    Contains the extremely ugly version of the code
                        used in generating benchmarking results in the paper.
    README_archive.txt: Separate readme specifically for the old code.

2. Install and test:
    First download and expand the repo to a working directory:
        git clone https://github.com/yunwilliamyu/secure-distributed-union-cardinality.git
        cd secure-distributed-union-cardinality
        bash setup.sh

    The provided setup script will download a precomputed log table for
    MPC-count and generate a few initial patient lists for simulation.

    If you do not already have the requisite Python packages in your
    environment, install them now:
        pip3 install numpy pycryptodome xxhash

    To run the standard simulation, we first need to generate a hospital
    network and preprocess it with HLL sketches for the total list:
        simulation/simulation_run.py test_dir

    Then, we change to that directory, and run the various benchmarks:
        cd test_dir
        ../simulation/run_all_but_mpc.sh
    That will run all of the methods except for the MPC ones, which require
    multiple rounds of communication, and so were included in their own files.
        ../simulation/run_mpc_count.sh
        ../simulation/run_mpc_hll.sh

    The simulation generates patient IDs from 1 to 100 million, spread between
    100 hospitals. Instead of actually querying the hospital for a patient
    condition, we here query it for patient lists, either 1 to 10,000
    (10_5.txt) or 1 to 1 million (million_test.txt). In a real run, the
    hospital would receive a query of the form "diabetes OR hypertension" and
    use those to generate a patient list for summarization. That involves
    coordinating with the electronic medical record database however, and is
    not part of this paper. Semi-relatedly, the authors also work on i2b2, a
    clinical query database, but that was not incorporated into this study.

3. Usage:
    We have tried to be diligent about including docstrings and useful help
    messages in each of the Python scripts for the hospitals and servers.
    See there for more runtime options (such as turning on/off masking), 
    changing simulation parameters, etc.

Contact:
    Yun William Yu, ywyu@math.toronto.edu
    Griffin Weber, griffin_weber@hms.harvard.edu


        
        
        
    
