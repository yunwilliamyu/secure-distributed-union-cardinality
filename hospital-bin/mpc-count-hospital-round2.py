#!/usr/bin/env python3
'''Does the second partial encryption round
'''

import sys
import os
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import elgamal  # noqa: 402

__version__ = '0.0.1'


def main(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    parser.add_argument("keyfile", help="private key file for hospital")
    parser.add_argument("input", help="binary encrypted count from central hub server")
    parser.add_argument("output", help="binary partial decryption to send to the central hub server")

    args = parser.parse_args(argv)

    print(args)
    sys.stdout.flush()

    output_string = None
    with open(args.keyfile, 'rb') as f:
        private_key = elgamal.deserialize_ElGamalPrivateKey(f.read())

    with open(args.input, 'rb') as f:
        round1b = f.read()

    padded_size = int(np.round(private_key.p.size_in_bits() / 8))
    output_string = elgamal.shared_secret(private_key, elgamal.CipherText.byte_init(private_key, round1b)).to_bytes(padded_size)

    if output_string is not None:
        with open(args.output, 'wb') as f:
            f.write(output_string)


if __name__ == "__main__":
    main(sys.argv[1:])
