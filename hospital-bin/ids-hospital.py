#!/usr/bin/env python3
'''Takes as input a list of numeric patient IDs, and hashes them using Sha-1
to send to a central server.
'''

import sys
import os
import argparse
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library.misc import query_results_fn  # noqa: E402

__version__ = '0.0.1'


def main(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument(
        "-r", "--rehash",
        help="sets a rehash seed",
        default="")
    parser.add_argument(
        "input", help="newline delimited list of numeric patient ids matching a query, or a npy/npz numpy array of those numeric patient ids")
    parser.add_argument(
        "output", help="binary output file to send to the central hub server")
    parser.add_argument(
        "--hospital-population", help="npy numpy array of numeric patient IDs at the hospital. If unset, will assume that all patients listed in input are at the hospital.")

    args = parser.parse_args(argv)

    print(args)
    sys.stdout.flush()

    query_results = query_results_fn(args.input, args.hospital_population)

    print("Method: Hashed IDs")
    print("Hashing {} patient ID list with salt '{}'".format(len(query_results), args.rehash))
    hashed_results = [hashlib.sha1((args.rehash + 'a' + str(x)).encode()).digest()[0:8] for x in query_results]
    output_string = b''.join(hashed_results)
    with open(args.output, 'wb') as f:
        f.write(output_string)


if __name__ == "__main__":
    main(sys.argv[1:])
