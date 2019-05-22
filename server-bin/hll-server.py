#!/usr/bin/env python3
'''Takes as input a set of HLL sketches from hospitals, combines them, and
outputs an approximate count.
'''

import sys
import os
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import hyperloglog  # noqa: E402

__version__ = '0.0.1'


def main(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    # Shared arguments

    parser.add_argument(
        "-k", "--num-buckets",
        help="number of HLL buckets used",
        type=int, default=128)
    parser.add_argument(
        "input", help="list of all files of hospital inputs; will use shell expansion if variables given", nargs="*")

    args = parser.parse_args(argv)

    print(args)
    sys.stdout.flush()

    hospital_files = args.input

    print("Mode: HLL")
    hospital_hlls = []
    hospital_counts = []  # Ensure that hospital_counts isn't empty
    for fname in hospital_files:
        with open(fname, 'rb') as f:
            buf = f.read()
        if len(buf) == 8:
            farray = np.frombuffer(buf, dtype=np.int64)
            hospital_counts.append(farray)
        else:
            A = hyperloglog.HyperLogLog.safe_import(buf)
            hospital_hlls.append(A)
    if len(hospital_counts) == 0:
        hospital_counts = [0]
    if len(hospital_hlls) == 0:
        hospital_hlls = [hyperloglog.HyperLogLog(128)]
    print("Min: {}".format(max(max(hospital_counts), sum(hospital_hlls).count())))
    print("Max: {}".format(sum(hospital_counts) + sum(hospital_hlls).count()))


if __name__ == "__main__":
    main(sys.argv[1:])
