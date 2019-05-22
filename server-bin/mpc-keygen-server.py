#!/usr/bin/env python3
'''
We provide several different mechanisms, ranging from the trivial count method,
to using HyperLogLog sketches, to using full-on homomorphic encryption and
secure MPC.
'''

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import elgamal  # noqa: E402

__version__ = '0.0.1'


def main(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    # Shared arguments
    parser.add_argument("keyfile", help="New public keyfile to be generated by combining hospital input.")
    parser.add_argument("input", help="list of all files of hospital inputs corresponding to partial keys; will use shell expansion if variables given", nargs="*")

    args = parser.parse_args(argv)

    print(args)
    sys.stdout.flush()

    hospital_public_shares = []
    for filename in args.input:
        with open(filename, 'rb') as f:
            buf = f.read()
        hospital_public_shares.append(buf)
    round0b = elgamal.hub_round0b(elgamal.key, hospital_public_shares)
    with open(args.keyfile, 'wb') as f:
        f.write(round0b['transmit'])


if __name__ == "__main__":
    main(sys.argv[1:])
