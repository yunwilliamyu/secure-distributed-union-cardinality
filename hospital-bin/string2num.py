#!/usr/bin/env python3

import fileinput
from hashlib import sha1

for line in fileinput.input():
    print(int.from_bytes(sha1(line.encode()).digest()[0:8], byteorder='big'))
