#!/usr/bin/env python3
'''Does the first encryption round, where the hospital encrypts a count according to the public key sent by the server
'''

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import elgamal  # noqa: E402
from library.misc import query_results_fn  # noqa: E402

__version__ = '0.0.1'


def main(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    parser.add_argument("keyfile", help="public key file from server that we are encrypting to")
    parser.add_argument("query", help="List of patients who match a query")
    parser.add_argument("output", help="binary output file to send to the central hub server")
    parser.add_argument("--hospital-population", help="npy numpy array of numeric patient IDs at the hospital. If unset, will assume that all patients listed in input query are at the hospital.")

    args = parser.parse_args(argv)

    print(args)
    sys.stdout.flush()

    output_string = None
    with open(args.keyfile, 'rb') as f:
        public_key = elgamal.deserialize_ElGamalPublicKey(f.read())
    query_results = query_results_fn(args.query, args.hospital_population)

    count = len(query_results)
    to_enc = pow(elgamal.Integer(4), elgamal.Integer(int(count)), public_key.p)

    output_string = elgamal.encrypt(public_key, to_enc).export_val()

    if output_string is not None:
        with open(args.output, 'wb') as f:
            f.write(output_string)


if __name__ == "__main__":
    main(sys.argv[1:])
