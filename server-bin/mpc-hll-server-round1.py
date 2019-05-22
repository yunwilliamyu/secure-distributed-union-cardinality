#!/usr/bin/env python3
'''
Server MPC HLL Round1
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
    parser.add_argument("output", help="Encrypted combined sketches output")
    parser.add_argument("publickey", help="Public key that the encrypted HLLs were encrypted using")
    parser.add_argument("input", help="list of all files of hospital inputs corresponding to encrypted HLLs", nargs="*")

    args = parser.parse_args(argv)

    print(args)
    sys.stdout.flush()

    round1a = []

    for filename in args.input:
        with open(filename, 'rb') as f:
            buf = f.read()
        round1a.append(buf)

    with open(args.publickey, 'rb') as f:
        key_public = elgamal.deserialize_ElGamalPublicKey(f.read())

    round1b = elgamal.hub_round1b_hll(key_public, round1a)

    with open(args.output, 'wb') as f:
        f.write(round1b['transmit'])


if __name__ == "__main__":
    main(sys.argv[1:])
