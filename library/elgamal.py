#!/usr/bin/env python3

from Cryptodome.Math.Numbers import Integer
from Cryptodome.PublicKey import ElGamal
# from Cryptodome.Util import number
from Cryptodome.Random.random import randint
import numpy as np
import time
import pickle
from xxhash import xxh64_intdigest
import array
import sys
from . import hyperloglog
import os

# Modulus (p)
_key_p = Integer(98028036272659031371560742242955169861910550634811171303401358378816132636992115630674943994829223218783498111481383051240760656669150360813864861995762931751131872340634050642274733438822049237741423531698542796484148312185879186781490731771229178272760055281624698765084217068018168288476774517939074390767)

# Generator (g)
_key_g = Integer(24206408586964102421812765517240973131319511671044202856202992075603898484808882229676430259688543524828496724940498465738345083295910778354197713986919490726627115004690129373676590479696817812956963632330252490994726984437037113519675887167311715863923322341281192755180251058938103973809860137264177781321)

# Public key (y)
_key_y = Integer(80096741602525354944101845054039870332476839337863539479413005037477582331123699686812254092422852136355062013980002281901835350602431500121588552286563732089960299445687681063705627334844826476798554044907030853427595347879115979412442090396606519731100408891011646324225891205357430072337588847702393772336)

# Private key (x)
_key_x = Integer(62211585486928253378495522273311524980557226618410408675140847678303278202803964175172717266795468749770948295585597912105277041671442793825256634521813434123039348444762246359790146437221632991836988836372789033182258605363475756418157919779953874088211874519141900265542574397181153219096197383524991841700)

# constructs elgamal key from stored constants
key = ElGamal.construct((_key_p, _key_g, _key_y, _key_x))

key_public = key.publickey()


def serialize_ElGamalPrivateKey(key):
    '''export an ElGamal Public Key to a bytestring'''
    padded_size = int(np.round(key.p.size_in_bits() / 8))
    return key.p.to_bytes(padded_size) + key.g.to_bytes(padded_size) + key.y.to_bytes(padded_size) + key.x.to_bytes(padded_size)


def deserialize_ElGamalPrivateKey(bytestring):
    '''constructs an ElGamal Public Key from a bytestring'''
    padded_size = len(bytestring) // 4
    assert padded_size * 4 == len(bytestring), "Wrong bytestring length"
    p = Integer.from_bytes(bytestring[:padded_size])
    g = Integer.from_bytes(bytestring[padded_size:2 * padded_size])
    y = Integer.from_bytes(bytestring[2 * padded_size:3 * padded_size])
    x = Integer.from_bytes(bytestring[3 * padded_size:])
    return ElGamal.construct((p, g, y, x))


def serialize_ElGamalPublicKey(key):
    '''export an ElGamal Public Key to a bytestring'''
    padded_size = int(np.round(key.p.size_in_bits() / 8))
    return key.p.to_bytes(padded_size) + key.g.to_bytes(padded_size) + key.y.to_bytes(padded_size)


def deserialize_ElGamalPublicKey(bytestring):
    '''constructs an ElGamal Public Key from a bytestring'''
    padded_size = len(bytestring) // 3
    assert padded_size * 3 == len(bytestring), "Wrong bytestring length"
    p = Integer.from_bytes(bytestring[:padded_size])
    g = Integer.from_bytes(bytestring[padded_size:2 * padded_size])
    y = Integer.from_bytes(bytestring[2 * padded_size:])
    return ElGamal.construct((p, g, y))

    def export_val(self):
        '''Returns a bytestring.'''
        padded_size = int(np.round(key.p.size_in_bits() / 8))
        return self.c1.to_bytes(padded_size) + self.c2.to_bytes(padded_size)

    @classmethod
    def byte_init(cls, key, bytestring):
        '''Returns a Ciphertext from a bytestring containing c1 and c2'''
        padded_size = int(np.round(key.p.size_in_bits() / 8))
        assert(len(bytestring) == 2 * padded_size)
        c1 = Integer.from_bytes(bytestring[:padded_size])
        c2 = Integer.from_bytes(bytestring[padded_size:])
        return cls(key, c1, c2)


# constructs elgamal key with different private key (implying different public key)
# random_secret = randint(2, int(key_p - 1))
# key_x2 = random_secret
# key_y2 = pow(key_g, key_x2, key_p)
# key2 = ElGamal.construct((key_p, key_g, key_y2, key_x2))
# key = key2

def encrypt(key, message):
    """Encrypts an integer message using the provided ElGamal key

        Includes a copy of key in the output (so be sure to use a public key; will assert error out otherwise)

        Note that this is stochastic because we randomly generate an ephemeral key each time

        ******VERY IMPORTANT*******
        The message must be a quadratic residue for DDH to hold.
        We do not ensure this in this encrypt function, because we carefully control what input messages are possible.
        Do *NOT* use this for general purpose encryption unless you first encode the message as a quadratic residue.

        Using a non-quadratic residue will look like it works, but will not be secure.
        ***************************
    """
    assert(not key.has_private())
    z = randint(2, int(key.p - 1))  # Ephemeral key
    c_1 = pow(key.g, z, key.p)  # first part of ciphertext
    s = pow(key.y, z, key.p)  # shared secret
    m = message
    c_2 = (Integer(m) * Integer(s)) % key.p
    return CipherText(key, c_1, c_2)


def decrypt(key, ciphertext):
    """Decrypts a ciphertext pair using the provided ElGamal key"""
    assert(key.p == ciphertext.key.p)
    assert(key.g == ciphertext.key.g)
    assert(key.y == ciphertext.key.y)
    c_1 = ciphertext.c1
    c_2 = ciphertext.c2
    s = pow(c_1, key.x, key.p)  # shared secret
    s_inv = pow(s, key.p - 2, key.p)  # modular multiplicative inverse https://stackoverflow.com/questions/4798654/modular-multiplicative-inverse-function-in-python
    m = (s_inv * c_2) % key.p
    return m


def shared_secret(key, ciphertext):
    '''Returns the shared secret component from a personal key for a ciphertext'''
    s = pow(ciphertext.c1, key.x, key.p)
    return s


class CipherText(object):
    def __init__(self, key, c1, c2):
        self.key = key
        self.c1 = c1
        self.c2 = c2

    def __add__(self, other):
        '''The group operation of the enciphered ring is multiplication in plaintext'''
        assert(self.key.p == other.key.p)
        assert(self.key.g == other.key.g)
        assert(self.key.y == other.key.y)
        return CipherText(self.key, (self.c1 * other.c1) % self.key.p, (self.c2 * other.c2) % self.key.p)

    def __radd__(self, other):
        '''Allows sum to work as expected'''
        if other == 0:
            return self
        else:
            raise TypeError('Cannot add together types')

    def private_equality_test(self, other):
        '''Returns an enciphered 1 if the two are the self. Otherwise returns an enciphered random number.

        https://crypto.stackexchange.com/questions/9527/how-does-an-oblivious-test-of-plaintext-equality-work
        '''
        assert(self.key.p == other.key.p)
        assert(self.key.g == other.key.g)
        assert(self.key.y == other.key.y)
        other_c1_inv = pow(other.c1, self.key.p - 2, self.key.p)
        other_c2_inv = pow(other.c2, self.key.p - 2, self.key.p)
        new_c1 = (self.c1 * other_c1_inv) % self.key.p
        new_c2 = (self.c2 * other_c2_inv) % self.key.p
        z = randint(2, int(key.p - 1))  # To blind the ciphertext
        new_c1_z = pow(new_c1, Integer(z), self.key.p)
        new_c2_z = pow(new_c2, Integer(z), self.key.p)
        return CipherText(self.key, new_c1_z, new_c2_z)

    def export_val(self):
        '''Returns a bytestring.'''
        padded_size = int(np.round(self.key.p.size_in_bits() / 8))
        return self.c1.to_bytes(padded_size) + self.c2.to_bytes(padded_size)

    @classmethod
    def byte_init(cls, key, bytestring):
        '''Returns a Ciphertext from a bytestring containing c1 and c2'''
        padded_size = int(np.round(key.p.size_in_bits() / 8))
        assert(len(bytestring) == 2 * padded_size)
        c1 = Integer.from_bytes(bytestring[:padded_size])
        c2 = Integer.from_bytes(bytestring[padded_size:])
        return cls(key, c1, c2)

    def __eq__(self, other):
        return ((self.key.p == other.key.p)
                and (self.key.g == other.key.g)
                and (self.key.y == other.key.y)
                and (self.c1 == other.c1)
                and (self.c2 == other.c2)
                )


def unroll(i, l):
    '''Unrolls an int i=4 into an array [4,4,4,4,1,1,1], where i is the number of 4's, and l is the total length of the array

    Note that we use "4" instead of e.g. "2" because all messages to be encrypted using the Elgmal function must be quadratic residues.
    '''
    ans = [4 for _ in range(i)] + [1 for _ in range(l - i)]
    return ans


def reroll_hll(hll_unrolled):
    '''Reroll HLL buckets.'''
    num_buckets = len(hll_unrolled)
    hll = np.zeros(num_buckets, dtype=np.uint8)
    for i in range(num_buckets):
        try:
            b = hll_unrolled[i].index(1)
        except ValueError:
            b = 0
        hll[i] = b
    return hll


def construct_key(key_template, sec):
    '''Constructs an ElGamal key from a secret and a template'''
    key_x2 = Integer(sec)
    key_y2 = pow(key_template.g, key_x2, key_template.p)
    key = ElGamal.construct((key_template.p, key_template.g, key_y2, key_x2))
    return key


def unroll_and_encrypt_hll(key_public, hll, sketch_size=32):
    '''Takes as input a HyperLogLog object, and outputs a list of
       encrypted unrolled buckets.
    '''
    unrolled_hyperloglog = [unroll(i, sketch_size) for i in hll.buckets]
    encrypted_hyperloglog = [[encrypt(key_public, x) for x in il] for il in unrolled_hyperloglog]
    return encrypted_hyperloglog


def get_hospital_shared_secrets(hospital_key, cehll):
    secrets = []
    for bucket in cehll:
        secrets_by_bucket = []
        for cell in bucket:
            secrets_by_bucket.append(shared_secret(hospital_key, cell))
        secrets.append(secrets_by_bucket)
    return secrets


def decode_hospital_shared_secrets_bytestring(key_public, ss_bytes, sketch_size=32):
    '''Decodes a bytestring of hospital shared secrets corresponding to an encrypted
    HLL bytestring from the transmit entry of hospital_round2a_hll'''
    padded_size = int(np.round(key_public.p.size_in_bits() / 8))
    array_item_num = len(ss_bytes) // padded_size
    assert array_item_num * padded_size == len(ss_bytes)

    num_buckets = array_item_num // sketch_size
    assert num_buckets * sketch_size == array_item_num

    ss_array = [[] for _ in range(num_buckets)]
    pointer = 0
    curr_bucket = 0
    val = 0
    while pointer < len(ss_bytes):
        ss = Integer.from_bytes(ss_bytes[pointer:pointer + padded_size])
        ss_array[curr_bucket].append(ss)

        pointer = pointer + padded_size
        val = val + 1
        if val == sketch_size:
            val = 0
            curr_bucket = curr_bucket + 1
    return ss_array


def decode_encrypted_hll_bytestring(key_public, ehll_bytes, sketch_size=32):
    '''Decodes an encrypted HLL bytestring from the transmit entry of hospital_round1a_hll
       or from the transmit entry of hospital_round1b_hll
    '''
    padded_size = int(np.round(key_public.p.size_in_bits() / 8))
    array_item_num = len(ehll_bytes) // padded_size // 2
    assert array_item_num * 2 * padded_size == len(ehll_bytes)

    num_buckets = array_item_num // sketch_size
    assert num_buckets * sketch_size == array_item_num

    encrypted_hll = [[] for _ in range(num_buckets)]
    pointer = 0
    curr_bucket = 0
    val = 0
    while pointer < len(ehll_bytes):
        encrypted_hll[curr_bucket].append(CipherText.byte_init(key_public, ehll_bytes[pointer:pointer + padded_size * 2]))

        pointer = pointer + padded_size * 2
        val = val + 1
        if val == sketch_size:
            val = 0
            curr_bucket = curr_bucket + 1
    return encrypted_hll


def hospital_round0a(key_template, key_x=None):
    '''The partial key generation phase

    If key_x is set, we'll use that instead of a new random integer

    Returns a dictionary with:
        key_x:      a new Elgamal private key secret share
        private_key: an ElGamal private key using key_x
        curr_y:     a partial public key share
        transmit:   a bytestring encoding of curr_y to send to hub
        time:       amount of time elapsed for this function
    '''
    start = time.perf_counter()
    if key_x:
        x = key_x
    else:
        x = Integer(randint(2, int(key_template.p - 1)))
    curr_y = pow(key_template.g, x, key_template.p)

    padded_size = int(np.round(key_template.p.size_in_bits() / 8))
    curr_y_bytestring = curr_y.to_bytes(padded_size)

    private_key = construct_key(key_template, x)
    assert(private_key.y == curr_y)

    elapsed = time.perf_counter() - start

    assert(Integer.from_bytes(curr_y_bytestring) == curr_y)

    answer_dict = {}
    answer_dict['key_x'] = x
    answer_dict['key_x_bytestring'] = x.to_bytes(padded_size)
    answer_dict['private_key'] = private_key
    answer_dict['private_key_export'] = serialize_ElGamalPrivateKey(private_key)
    answer_dict['curr_y'] = curr_y
    answer_dict['transmit'] = curr_y_bytestring
    answer_dict['time'] = elapsed
    return answer_dict


def hub_round0b(key_template, hospital_public_shares):
    '''Generating the shared public key from a list of partial shares
    from the hospitals as bytestrings

    Returns a dictionary with:
        key_public: a new ElGamal public key
        time: amount of time elapsed for this function
    '''
    start = time.perf_counter()

    key_y = Integer(1)
    for bytestring_y in hospital_public_shares:
        curr_y = Integer.from_bytes(bytestring_y)
        key_y = (curr_y * key_y) % key_template.p
    key_public = ElGamal.construct((key_template.p, key_template.g, key_y))
    elapsed = time.perf_counter() - start

    answer_dict = {}
    answer_dict['key_public'] = key_public
    answer_dict['curr_y'] = key_y
    # answer_dict['transmit'] = key_y_bytestring
    answer_dict['transmit'] = serialize_ElGamalPublicKey(key_public)
    answer_dict['time'] = elapsed
    return answer_dict


def hospital_round1a_hll(key_public, hll):
    '''Unrolls and encrypts the HLL

    Returns a dictionary with:
        encrypted_hll: an array of arrays, where the outer array is the list of buckets,
            and the inner array is the unrolled bucket.
        transmit: a bytestring encoding of encrypted_hll to send to hub
        time: amount of time elapsed for this function
    '''
    start = time.perf_counter()
    encrypted_hll = unroll_and_encrypt_hll(key_public, hll)

    messages = (item.export_val() for sublist in encrypted_hll for item in sublist)
    transmit = b''.join(messages)

    elapsed = time.perf_counter() - start

    answer_dict = {}
    answer_dict['encrypted_hll'] = encrypted_hll
    answer_dict['time'] = elapsed
    answer_dict['transmit'] = transmit
    return answer_dict


def hub_round1b_hll(key_public, hospital_encrypted_hll_bytestrings):
    '''Collapse encrypted HLLs

    Returns a dictionary with:
        collapsed_enc_hll: the combined encrypted unrolled HLL
        transmit: a byetsring version of collapsed_enc_hll
        time    : elapsed time
    '''
    start = time.perf_counter()

    encrypted_hyperloglogs = [decode_encrypted_hll_bytestring(key_public, x) for x in hospital_encrypted_hll_bytestrings]
    num_buckets = len(encrypted_hyperloglogs[0])
    num_hospitals = len(encrypted_hyperloglogs)
    sketch_size = len(encrypted_hyperloglogs[0][0])

    collapsed_enc_hll = []
    for i in range(num_buckets):
        collapsed_enc_hll_i = []
        for j in range(sketch_size):
            one = encrypt(key_public, 1)
            curr = encrypt(key_public, 1)
            for k in range(num_hospitals):
                curr = curr + encrypted_hyperloglogs[k][i][j]
            curr = curr.private_equality_test(one)  # If equal 1, then there were no items with that many leading 0's
            collapsed_enc_hll_i.append(curr)
        collapsed_enc_hll.append(collapsed_enc_hll_i)

    messages = (item.export_val() for sublist in collapsed_enc_hll for item in sublist)
    transmit = b''.join(messages)

    elapsed = time.perf_counter() - start

    answer_dict = {}
    answer_dict['time'] = elapsed
    answer_dict['collapsed_enc_hll'] = collapsed_enc_hll
    answer_dict['transmit'] = transmit
    return answer_dict


def hospital_round2a_hll(hospital_key, cehll_bytes, sketch_size=32):
    '''Generates the secret shares from each hospital for decryption of the
    collapsed, encrypted, HLL bytestring from hub_round1b_hll['transmit'].

    Returns a dictionary with:
        secrets
        transmit
        time
    '''
    start = time.perf_counter()
    cehll = decode_encrypted_hll_bytestring(hospital_key, cehll_bytes)
    secrets = get_hospital_shared_secrets(hospital_key, cehll)
    padded_size = int(np.round(hospital_key.p.size_in_bits() / 8))
    messages = (item.to_bytes(padded_size) for sublist in secrets for item in sublist)
    transmit = b''.join(messages)
    elapsed = time.perf_counter() - start

    answer_dict = {}
    answer_dict['time'] = elapsed
    answer_dict['secrets'] = secrets
    answer_dict['transmit'] = transmit
    return answer_dict


def hub_round2b_hll(key_public, cehll_bytes, hospital_shared_secrets_bytestrings):
    '''
    Combines together the shared secrets to decrypt the combined, encrypted, HLL bytes
    '''
    start = time.perf_counter()

    cehll = decode_encrypted_hll_bytestring(key_public, cehll_bytes)
    all_hospital_secrets = [decode_hospital_shared_secrets_bytestring(key_public, bs) for bs in hospital_shared_secrets_bytestrings]
    num_hospitals = len(all_hospital_secrets)
    num_buckets = len(cehll)
    sketch_size = len(cehll[0])
    combined_hll_unrolled = []
    for i in range(num_buckets):
        m_values = []
        for j in range(sketch_size):
            curr_secret = Integer(1)
            for k in range(num_hospitals):
                curr_secret = (curr_secret * all_hospital_secrets[k][i][j]) % key_public.p
            s_inv = pow(curr_secret, key_public.p - 2, key_public.p)  # modular multiplicative inverse https://stackoverflow.com/questions/4798654/modular-multiplicative-inverse-function-in-python
            m = (s_inv * cehll[i][j].c2) % key_public.p
            sqrt_m = pow(m, (key_public.p + 1) // 4, key_public.p)  # taking square root
            if sqrt_m > key_public.p // 2:
                sqrt_m = key_public.p - sqrt_m
            m_values.append(sqrt_m)
        combined_hll_unrolled.append(m_values)

    combined_hll = reroll_hll(combined_hll_unrolled)

    elapsed = time.perf_counter() - start

    answer_dict = {}
    answer_dict['time'] = elapsed
    answer_dict['combined_buckets'] = combined_hll
    answer_dict['transmit'] = combined_hll.tobytes()
    return answer_dict


class discrete_log4_table:
    '''Allows quickly figuring out what the discrete log is with respect to 4'''
    def __init__(self, key, n):
        self.key = key
        self.n = n
        # Figure out sizes needed for array
        for t in 'BHILQ':
            if array.array(t).itemsize * 8 > np.log(n) / np.log(2):
                typecode = t
                break

        y = Integer(1)
        table = [array.array(typecode) for _ in range(2**16)]
        for x in range(n):
            h = xxh64_intdigest(str(y)) % 2**16
            table[h].append(x)
            y = Integer(y) * Integer(4) % key.p
            if x % 1000000 == 0:
                print(x)
        self.table = table

    def dlog4(self, q):
        for x in self.table[xxh64_intdigest(str(q)) % 2**16]:
            if pow(Integer(4), x, self.key.p) == q:
                return x

    @classmethod
    def fromtable(cls, key, table):
        '''Alternate constructor when the table is already known'''
        obj = cls(key, 1)
        obj.n = max([max(x) for x in table])
        obj.table = table
        return obj


_dl4_table = None


def load_precomputed_discrete_log4_table():
    '''Load precomputed discrete log table for ElGamal with this prime if exists'''
    global _dl4_table
    if _dl4_table is None:
        try:
            directory = os.path.dirname(os.path.abspath(hyperloglog.__file__))
            f_tab = open(directory + '/TAB-table.pickle', 'rb')
            _dl4_table = discrete_log4_table.fromtable(key, pickle.load(f_tab))
            f_tab.close()
        except FileNotFoundError:
            print("Precomputed table not found. Continuing to use Naive slow method.", file=sys.stderr)


def discrete_log4_naive(key, x):
    '''Finds the discrete log with respect to 4 for a really big number.

    Does it naively be counting up by powers of two until we get to x'''
    key_public = key
    s_inv = pow(Integer(4), key_public.p - 2, key_public.p)  # modular multiplicative inverse https://stackoverflow.com/questions/4798654/modular-multiplicative-inverse-function-in-python
    tmp = Integer(x)
    ans = 0
    while tmp != 1:
        tmp = tmp * s_inv % key_public.p
        ans = ans + 1
        if ans % 100000 == 0:
            print('.', end='', flush=True)
    return ans


def discrete_log4(key, x):
    '''Finds the discrete log with respect to 4 for a really big number and the prime in key'''
    if _dl4_table is None:
        print("No precomputed log table. Doing discrete log problem")
        logged_x = discrete_log4_naive(key, x)
    else:
        print("Using precomputed log table")
        logged_x = _dl4_table.dlog4(x)
    return logged_x


def full_simulation_hll(key_template, hospital_hlls):
    '''Runs a full simulation of an encrypted HLL merger'''
    num_hospitals = len(hospital_hlls)

    round0a = [hospital_round0a(key_template) for _ in range(num_hospitals)]
    round0a_transmissions = [x['transmit'] for x in round0a]
    round0b = hub_round0b(key_template, round0a_transmissions)
    key_public = round0b['key_public']
    hospital_keys = [x['private_key'] for x in round0a]
    round1a = [hospital_round1a_hll(key_public, hll) for hll in hospital_hlls]
    round1a_transmissions = [x['transmit'] for x in round1a]
    round1b = hub_round1b_hll(key_public, round1a_transmissions)
    round2a = [hospital_round2a_hll(hk, round1b['transmit']) for hk in hospital_keys]
    round2b = hub_round2b_hll(key_public, round1b['transmit'], [x['transmit'] for x in round2a])

    combined_hll = hyperloglog.HyperLogLog(hospital_hlls[0].k)
    combined_hll.buckets = round2b['combined_buckets']

    ans_dict = {}
    ans_dict['round0a_time'] = sum(x['time'] for x in round0a)
    ans_dict['round0b_time'] = round0b['time']
    ans_dict['round1a_time'] = sum(x['time'] for x in round1a)
    ans_dict['round1b_time'] = round1b['time']
    ans_dict['round2a_time'] = sum(x['time'] for x in round2a)
    ans_dict['round2b_time'] = round2b['time']
    return combined_hll, ans_dict


def full_simulation_counts(key_template, hospital_counts):
    '''Runs a full simulation of an encrypted sum'''
    padded_size = int(np.round(key_template.p.size_in_bits() / 8))
    num_hospitals = len(hospital_counts)
    round0a = [hospital_round0a(key_template) for _ in range(num_hospitals)]
    round0a_transmissions = [x['transmit'] for x in round0a]
    round0b = hub_round0b(key_template, round0a_transmissions)
    key_public = round0b['key_public']
    hospital_keys = [x['private_key'] for x in round0a]

    round1a_start = time.perf_counter()
    round1a = [encrypt(key_public, pow(Integer(4), Integer(int(count)), key_public.p)).export_val() for count in hospital_counts]
    round1a_time = time.perf_counter() - round1a_start

    round1b_start = time.perf_counter()
    round1b = sum([CipherText.byte_init(key_public, x) for x in round1a]).export_val()
    round1b_time = time.perf_counter() - round1b_start

    round2a_start = time.perf_counter()
    round2a = [shared_secret(hk, CipherText.byte_init(key_public, round1b)).to_bytes(padded_size) for hk in hospital_keys]
    round2a_time = time.perf_counter() - round2a_start

    round2b_start = time.perf_counter()
    round2b_sec = Integer(1)
    encrypted_sum = CipherText.byte_init(key_public, round1b)
    for k in range(num_hospitals):
        round2b_sec = (round2b_sec * Integer.from_bytes(round2a[k])) % key_public.p
    s_inv = pow(round2b_sec, key_public.p - 2, key_public.p)  # modular multiplicative inverse https://stackoverflow.com/questions/4798654/modular-multiplicative-inverse-function-in-python

    decrypted_sum = (s_inv * encrypted_sum.c2) % key_public.p

    load_precomputed_discrete_log4_table()  # using precomputation instead of naive method

    logged_sum = discrete_log4(key_public, decrypted_sum)
    round2b_time = time.perf_counter() - round2b_start

    ans_dict = {}
    ans_dict['round0a_time'] = sum(x['time'] for x in round0a)
    ans_dict['round0b_time'] = round0b['time']
    ans_dict['round1a_time'] = round1a_time
    ans_dict['round1b_time'] = round1b_time
    ans_dict['round2a_time'] = round2a_time
    ans_dict['round2b_time'] = round2b_time
    return logged_sum, ans_dict
