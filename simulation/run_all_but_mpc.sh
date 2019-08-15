#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
HOSP_DIR=$DIR/../hospital-bin

for file in hosp*.npy
do
    fbname=$(basename "$file" .npy)
    $HOSP_DIR/count-hospital.py $file $fbname-count.bin
    $HOSP_DIR/hll-hospital.py $file $fbname-hll.bin
    $HOSP_DIR/ids-hospital.py $file $fbname-ids.bin
    $HOSP_DIR/count-hospital.py $DIR/million_test.txt --hospital-population $file $fbname-6count.bin
    $HOSP_DIR/hll-hospital.py $DIR/million_test.txt --hospital-population $file $fbname-6hll.bin
    $HOSP_DIR/hll-hospital.py $DIR/million_test.txt --hospital-population $file $fbname-6hllm.bin -m --hospital-hll-freqs-file $fbname.freqs
    $HOSP_DIR/ids-hospital.py $DIR/million_test.txt --hospital-population $file $fbname-6ids.bin
    $HOSP_DIR/count-hospital.py $DIR/10_5.txt --hospital-population $file $fbname-5count.bin
    $HOSP_DIR/hll-hospital.py $DIR/10_5.txt --hospital-population $file $fbname-5hll.bin
    $HOSP_DIR/hll-hospital.py $DIR/10_5.txt --hospital-population $file $fbname-5hllm.bin -m --hospital-hll-freqs-file $fbname.freqs
    $HOSP_DIR/ids-hospital.py $DIR/10_5.txt --hospital-population $file $fbname-5ids.bin
done

