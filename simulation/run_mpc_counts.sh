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
    $HOSP_DIR/mpc-count-hospital-round1.py publickey_combined.bin $file $fbname.r1ect
done

$SERV_DIR/mpc-count-server-round1.py combined_r1ect.bin publickey_combined.bin *.r1ect

for file in hosp*.npy
do
    fbname=$(basename "$file" .npy)
    $HOSP_DIR/mpc-count-hospital-round2.py $fbname.privatekey combined_r1ect.bin $fbname.r2share
done

$SERV_DIR/mpc-count-server-round2.py publickey_combined.bin combined_r1ect.bin *.r2share
