This code is associated with the following manuscript:
> Yun William Yu and Griffin Weber. *Federated queries of clinical data
> repositories: balancing accuracy and privacy.* Submitted and under review.

Note that this was the code used to generate the figures in the manuscript.
However, it is very brittle and designed for benchmarking on our personal
compute devices, rather than for ordinary use. Running it will reproduce our
entire benchmarking pipeline, which may take days to complete, and might
require large amounts of memory.

If you are simply running a quick sanity check or playing with the methods
themselves, we strongly suggest you go to the root directory of this repo and
run the code there, which has been written with nice easy to use command-line
parameters. The newly refactored code supersedes the code in this directory;
we include it here for the sake of scientific completeness and reproducibility.

1. Files included:

    setup.sh:                   Run this to download TAB-table.pickle
    elgamal.py:                 code for the ElGamal cryptographic primitives
    hyperloglog.py:             code for the HLL sketch functionality
    generate_hospital_lists.py: code to simulate the 100 hospital-network
    myglobals.py:               an empty Python module, allowing ugly globals
    run_once.py:                Command-line program to run the full simulation
                                once with the specific random seed
    multirun.bash:              Command-line script that runs "run_once.py" 100
                                times with different random seeds

2. Files generated (and in some cases included for completion):

    TAB-table.pickle:               A precomputed log table for ElGamal
    results_[seed]-[unixtime].dat:  The result file for a single "run_once" 
                                    simulation, as a Python text-format list of
                                    dictionaries.
    summary100.txt:                 The summary from analyzing the results* files

3. Run instructions: (given a 32-GB RAM Linux machine with all required Python
                      packages installed)

    First we download the log table:
        ./setup.sh

    (Optional: clean up the prerun data from our paper simulations):
        rm results*.dat
        rm summary100.txt

    Then we run all 100 simulations:
        ./multirun.bash

    Then we analyze the results:
        python3 analyzer.py *.dat > summary100.txt
        python3 analayzer_csv_out.py *.dat > summary100.csv

    summary100.txt looks like the tables given in the appendix to the paper
    summary100.csv is the same data, in slightly easier to parse CSV format.

    Note that the results include more statistics than we provided in the paper.
    Notably, we include both 2-anonymity (r1), 5-anonymity (r5), and
    10-anonymity (r9), whereas in the paper, we only discussed 10-anonymity.




