#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
HOSP_DIR=$DIR/../hospital-bin
SERV_DIR=$DIR/../server-bin

for file in hosp*.npy
do
    fbname=$(basename "$file" .npy)
    $HOSP_DIR/mpc-keygen-hospital.py $fbname.privatekey $fbname.publickey
done

$SERV_DIR/mpc-keygen-server.py publickey_combined.bin *.publickey

for file in hosp*.npy
do
    fbname=$(basename "$file" .npy)
    $HOSP_DIR/mpc-hll-hospital-round1.py publickey_combined.bin $fbname-6hll.bin $fbname.r1ehll
done

$SERV_DIR/mpc-hll-server-round1.py combined_r1ehll.bin publickey_combined.bin *.r1ehll

for file in hosp*.npy
do
    fbname=$(basename "$file" .npy)
    $HOSP_DIR/mpc-hll-hospital-round2.py $fbname.privatekey combined_r1ehll.bin $fbname.r2sharehll
done

$SERV_DIR/mpc-hll-server-round2.py publickey_combined.bin combined_r1ehll.bin *.r2sharehll
