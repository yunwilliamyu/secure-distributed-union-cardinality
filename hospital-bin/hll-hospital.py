#!/usr/bin/env python3
'''Takes as input a list of numeric patient IDs, and summarizes them for
sending to a central server.

We provide several different mechanisms, ranging from the trivial count method,
to using HyperLogLog sketches, to using full-on homomorphic encryption and
secure MPC.
'''

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import hyperloglog  # noqa: E402
from library.misc import query_results_fn, count_output  # noqa: E402

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
        "-m", "--mask",
        help="specifies whether or not queries with less than 10 patients are masked",
        action="store_true")
    parser.add_argument(
        "-s", "--shuffle",
        help="sets a shuffle seed")
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
    parser.add_argument(
        "--hospital-hll-freqs-file",
        help='''Used only when mask is also set. Figures out if any
        buckets correspond to fewer than 10 patients in hospital_population, and if so,
        returns a count instead. Consists of an HLL Freqs object
        for the entire hospital population using the same default seed.
        Will not work if a rehash seed is set.''')

    args = parser.parse_args(argv)

    print(args)
    sys.stdout.flush()

    if args.mask:
        if not args.hospital_hll_freqs_file:
            print("Need hospital-hll-freqs-file to do masking in HLL mode", file=sys.stderr)
            sys.exit(-1)

    query_results = query_results_fn(args.input, args.hospital_population)

    print("Method: HLL")
    A = hyperloglog.HyperLogLog(args.num_buckets, salt=args.rehash)
    A.update(query_results)
    if args.hospital_hll_freqs_file:
        with open(args.hospital_hll_freqs_file, 'rb') as f:
            hosp_hll_freqs = hyperloglog.HyperLogLog_freqs.safe_import(f.read())
            assert hosp_hll_freqs.k == A.k, "Buckets much match to that of the bucket frequency files"
        hll_privacy = A.privacy(hosp_hll_freqs)
        print("Patients with <2 anonymity: {}".format(hll_privacy[2]))
        print("Patients with <5 anonymity: {}".format(hll_privacy[5]))
        print("Patients with <10 anonymity: {}".format(hll_privacy[10]))
    if args.mask:
        if hll_privacy[10] == 0:
            output_string = A.safe_export(shuffle=args.shuffle)
            print("HLL gives 10-anonymity to all patients, so no masking required.")
            print("True count: {}".format(len(query_results)))
            print("HLl estimate: {}".format(A.count()))
        else:
            output_string = count_output(query_results, mask=True)
            print("HLL not private. Using count+mask instead.")
    else:
        print("Using HLL without masking.")
        print("True count: {}".format(len(query_results)))
        print("HLl estimate: {}".format(A.count()))
        output_string = A.safe_export(shuffle=args.shuffle)
    with open(args.output, 'wb') as f:
        f.write(output_string)


if __name__ == "__main__":
    main(sys.argv[1:])
