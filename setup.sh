#!/bin/sh

echo Downloading precomputed log tables for ElGamal Count MPC
# wget -c elgamal-billion-table.npy

echo Precomputing some initial patient lists for simulation
seq 1 100000 > simulation/10_5.txt
seq 1 1000000 > simulation/million_test.txt
