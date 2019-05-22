#!/usr/bin/env python3
'''Takes as input a set of hashed IDs from hospitals, and counts the number
of unique IDs.
'''

import sys
import argparse
import numpy as np

__version__ = '0.0.1'


def main(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    parser.add_argument(
        "input", help="list of all files of hospital inputs; will use shell expansion if variables given", nargs="*")

    args = parser.parse_args(argv)

    print(args)
    sys.stdout.flush()

    hospital_files = args.input

    print("Mode: IDs")
    hospital_lists = []
    for fname in hospital_files:
        with open(fname, 'rb') as f:
            buf = f.read()
        h_list = np.frombuffer(buf, dtype=np.uint64)
        hospital_lists.append(h_list)
    combined_list = np.concatenate(hospital_lists)
    unique = np.unique(combined_list)
    print("Patients: {}".format(len(unique)))


if __name__ == "__main__":
    main(sys.argv[1:])
