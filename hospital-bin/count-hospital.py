#!/usr/bin/env python3
'''Takes as input a list of numeric patient IDs, and summarizes them for
sending to a central server by turning them into a single aggregate count
'''

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library.misc import query_results_fn, count_output  # noqa: E402

__version__ = '0.0.1'


def main(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    parser.add_argument(
        "-m", "--mask",
        help="specifies whether or not queries with less than 10 patients are masked",
        action="store_true")
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

    print("Method: Count")
    output_string = count_output(query_results, mask=args.mask)
    with open(args.output, 'wb') as f:
        f.write(output_string)


if __name__ == "__main__":
    main(sys.argv[1:])
