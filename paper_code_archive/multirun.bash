#!/usr/bin/bash

#for seed in 1 2 3 4 5
#for seed in 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
#for seed in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
for seed in `seq 1 100`
do
    python3 run_once.py $seed
done
