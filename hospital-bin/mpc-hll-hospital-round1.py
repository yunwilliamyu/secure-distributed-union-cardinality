#!/usr/bin/env python3
'''Does the first encryption round, where the hospital encrypts a count according to the public key sent by the server
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

    parser.add_argument("keyfile", help="public key file from server that we are encrypting to")
    parser.add_argument("input", help="Input HLL")
    parser.add_argument("output", help="binary output file to send to the central hub server")

    args = parser.parse_args(argv)

    print(args)
    sys.stdout.flush()

    output_string = None
    with open(args.keyfile, 'rb') as f:
        public_key = elgamal.deserialize_ElGamalPublicKey(f.read())

    with open(args.input, 'rb') as f:
        hll = hyperloglog.HyperLogLog.safe_import(f.read())

    ehll = elgamal.hospital_round1a_hll(public_key, hll)

    output_string = ehll['transmit']

    with open(args.output, 'wb') as f:
        f.write(output_string)


if __name__ == "__main__":
    main(sys.argv[1:])
