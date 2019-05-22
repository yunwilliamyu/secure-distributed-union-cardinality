#!/usr/bin/env python3
'''Generates a partial public/private key.

The partial public key should be sent to the server,
and the partial private key kept safe for later partial decryption.
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
    parser.add_argument("keyfile", help="private key file; do NOT send to anyone. If does not exist, will be generated and saved to this filename.")
    parser.add_argument(
        "publickey", help="binary output file to send to the central hub server")
    parser.add_argument(
        "--hospital-population", help="npy numpy array of numeric patient IDs at the hospital. If unset, will assume that all patients listed in input query are at the hospital.")

    args = parser.parse_args(argv)

    print(args)
    sys.stdout.flush()

    try:  # If the keyfile already exists, simply load the private_key
        print("Loading private keyfile")
        with open(args.keyfile, 'rb') as f:
            private_key = elgamal.deserialize_ElGamalPrivateKey(f.read())
        key_x = private_key.x
        round0a = elgamal.hospital_round0a(elgamal.key, key_x=key_x)
    except EnvironmentError:  # If the keyfile doesn't already exist, generate it and write out to the keyfile.
        print("Keyfile not found")
        print("Generating keyfile and writing it out")
        round0a = elgamal.hospital_round0a(elgamal.key)
        with open(args.keyfile, 'wb') as f:
            f.write(round0a['private_key_export'])
    public_key_partial = round0a['transmit']

    with open(args.publickey, 'wb') as f:
        f.write(public_key_partial)


if __name__ == "__main__":
    main(sys.argv[1:])
