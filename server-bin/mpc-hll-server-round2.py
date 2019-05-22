#!/usr/bin/env python3
'''
Server round 2
'''

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import elgamal  # noqa: E402
from library import hyperloglog  # noqa: E402

__version__ = '0.0.1'


def main(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    # Shared arguments
    parser.add_argument("publickey", help="Public key that the encrypted HLLs were encrypted using")
    parser.add_argument("ehll", help="encrypted combined HLL from round 1")
    parser.add_argument("input", help="list of all files of hospital inputs corresponding to partial decryptions", nargs="*")

    args = parser.parse_args(argv)

    print(args)
    sys.stdout.flush()

    round2a = []

    for filename in args.input:
        with open(filename, 'rb') as f:
            buf = f.read()
        round2a.append(buf)

    with open(args.publickey, 'rb') as f:
        key_public = elgamal.deserialize_ElGamalPublicKey(f.read())

    with open(args.ehll, 'rb') as f:
        ehll_bytes = f.read()

    round2b = elgamal.hub_round2b_hll(key_public, ehll_bytes, round2a)

    combined_hll = hyperloglog.HyperLogLog(len(round2b['combined_buckets']))
    combined_hll.buckets = round2b['combined_buckets']

    print(combined_hll.count())


if __name__ == "__main__":
    main(sys.argv[1:])
