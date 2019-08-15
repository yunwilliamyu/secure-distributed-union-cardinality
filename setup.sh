#!/bin/sh

cd library
echo Downloading precomputed log tables for ElGamal Count MPC
wget -c http://static.ywyu.net/elgamal-billion-table.npy
cd ..

echo Precomputing some initial patient lists for simulation
seq 1 100000 > simulation/10_5.txt
seq 1 1000000 > simulation/million_test.txt
