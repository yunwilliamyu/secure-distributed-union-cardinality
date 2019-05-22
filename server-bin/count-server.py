#!/usr/bin/env python3
'''Takes as input a set of counts from hospitals, and combines them.
'''

import sys
import argparse
import numpy as np

__version__ = '0.0.1'


class ArgClass:
    '''So I don't have to duplicate argument info'''
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def main(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    input_arg = ArgClass(
        "input", help="list of all files of hospital inputs; will use shell expansion if variables given", nargs="*")

    parser.add_argument(*input_arg.args, **input_arg.kwargs)

    args = parser.parse_args(argv)

    print(args)
    sys.stdout.flush()

    hospital_files = args.input

    print("Mode: Count")
    hospital_counts = []
    for fname in hospital_files:
        with open(fname, 'rb') as f:
            farray = np.frombuffer(f.read(), dtype=np.int64)
            assert len(farray) == 1, "When using count mode, every hospital result file must contain a single int64"
            hospital_counts.append(farray[0])
    print("Max: {}".format(sum(hospital_counts)))
    print("Min: {}".format(max(hospital_counts)))


if __name__ == "__main__":
    main(sys.argv[1:])
