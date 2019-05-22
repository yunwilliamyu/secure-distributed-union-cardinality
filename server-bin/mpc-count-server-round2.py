#!/usr/bin/env python3
'''
Server round 2
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
    parser.add_argument("publickey", help="Public key that the encrypted counts were encrypted using")
    parser.add_argument("esum", help="encrypted sum count from round 1")
    parser.add_argument("input", help="list of all files of hospital inputs corresponding to encrypted counts", nargs="*")

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

    with open(args.esum, 'rb') as f:
        encrypted_sum = elgamal.CipherText.byte_init(key_public, f.read())

    round2b_sec = elgamal.Integer(1)
    for sec in round2a:
        round2b_sec = (round2b_sec * elgamal.Integer.from_bytes(sec)) % key_public.p
    s_inv = pow(round2b_sec, key_public.p - 2, key_public.p)  # modular multiplicative inverse https://stackoverflow.com/questions/4798654/modular-multiplicative-inverse-function-in-python

    decrypted_sum = (s_inv * encrypted_sum.c2) % key_public.p

    elgamal.load_precomputed_discrete_log4_table()  # using precomputation instead of naive method

    logged_sum = elgamal.discrete_log4(key_public, decrypted_sum)

    print(decrypted_sum)
    print(logged_sum)


if __name__ == "__main__":
    main(sys.argv[1:])
