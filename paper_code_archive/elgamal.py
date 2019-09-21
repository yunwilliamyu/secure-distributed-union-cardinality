#!/usr/bin/env python3

from Cryptodome.Math.Numbers import Integer
from Cryptodome.PublicKey import ElGamal
from Cryptodome.Random.random import randint
import myglobals
import numpy as np
import time
from xxhash import xxh64_intdigest
import array

# Modulus (p)
key_p = Integer(98028036272659031371560742242955169861910550634811171303401358378816132636992115630674943994829223218783498111481383051240760656669150360813864861995762931751131872340634050642274733438822049237741423531698542796484148312185879186781490731771229178272760055281624698765084217068018168288476774517939074390767)

# Generator (g)
key_g = Integer(24206408586964102421812765517240973131319511671044202856202992075603898484808882229676430259688543524828496724940498465738345083295910778354197713986919490726627115004690129373676590479696817812956963632330252490994726984437037113519675887167311715863923322341281192755180251058938103973809860137264177781321)

# Public key (y)
key_y = Integer(80096741602525354944101845054039870332476839337863539479413005037477582331123699686812254092422852136355062013980002281901835350602431500121588552286563732089960299445687681063705627334844826476798554044907030853427595347879115979412442090396606519731100408891011646324225891205357430072337588847702393772336)

# Private key (x)
key_x = Integer(62211585486928253378495522273311524980557226618410408675140847678303278202803964175172717266795468749770948295585597912105277041671442793825256634521813434123039348444762246359790146437221632991836988836372789033182258605363475756418157919779953874088211874519141900265542574397181153219096197383524991841700)


# constructs elgamal key from stored constants
key = ElGamal.construct((key_p, key_g, key_y, key_x))

# constructs elgamal key with different private key (implying different public key)
random_secret = randint(2, int(key_p - 1))
key_x2 = random_secret
key_y2 = pow(key_g, key_x2, key_p)
key2 = ElGamal.construct((key_p, key_g, key_y2, key_x2))

key = key2


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
    assert not key.has_private()
    z = randint(2, int(key.p - 1))  # Ephemeral key
    c_1 = pow(key.g, z, key.p)  # first part of ciphertext
    s = pow(key.y, z, key.p)  # shared secret
    # m = pow(Integer(message), 2, key.p) # square the message based off https://github.com/Legrandin/pycryptodome/issues/90
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
    # We took out code encoding messages as quadratic residues because we can simply control the values we pass in.
    # sqrt_m = pow(m, (key.p + 1)//4, key.p) # taking square root
    # if sqrt_m > key.p//2:
    #    sqrt_m = key.p - sqrt_m
    # return sqrt_m
    return m


def shared_secret(key, ciphertext):
    '''Returns the shared secret component from a personal key for a ciphertext'''
    s = pow(ciphertext.c1, key.x, key.p)
    return s


class CipherText():
    def __init__(self, key, c1, c2):
        self.key = key
        self.c1 = c1
        self.c2 = c2

    def __add__(self, other):
        assert(self.key.p == other.key.p)
        assert(self.key.g == other.key.g)
        assert(self.key.y == other.key.y)
        return CipherText(self.key, (self.c1 * other.c1) % self.key.p, (self.c2 * other.c2) % self.key.p)

    def private_equality_test(self, other):
        # https://crypto.stackexchange.com/questions/9527/how-does-an-oblivious-test-of-plaintext-equality-work
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


def unroll(i, l):
    '''Unrolls an int i=4 into an array [4,4,4,4,1,1,1], where i is the number of 4's, and l is the total length of the array

    Note that we use "4" instead of e.g. "2" because all messages to be encrypted using the Elgmal function must be quadratic residues.
    '''
    ans = [4 for _ in range(i)] + [1 for _ in range(l - i)]
    return ans


def construct_key(key_template, sec):
    '''Constructs an ElGamal key from a secret and a template'''
    key_x2 = Integer(sec)
    key_y2 = pow(key_template.g, key_x2, key_template.p)
    key = ElGamal.construct((key_template.p, key_template.g, key_y2, key_x2))
    return key


def unroll_and_encrypt_hll(i):
    hll = myglobals.hyperloglogs[i]
    sketch_size = myglobals.sketch_size
    start = time.perf_counter()
    unrolled_hyperloglog = [unroll(i, sketch_size) for i in hll]
    encrypted_hyperloglog = [[encrypt(myglobals.key_public, x) for x in il] for il in unrolled_hyperloglog]
    elapsed = time.perf_counter() - start
    return (encrypted_hyperloglog, elapsed)


def get_hospital_shared_secrets(i):
    hospital_key = myglobals.hospital_keys[i]
    collapsed_enc_hll = myglobals.collapsed_enc_hll
    start = time.perf_counter()
    secrets = []
    for bucket in collapsed_enc_hll:
        secrets_by_bucket = []
        for cell in bucket:
            secrets_by_bucket.append(shared_secret(hospital_key, cell))
        secrets.append(secrets_by_bucket)
    elapsed = time.perf_counter() - start
    return (secrets, elapsed)


def enc_test_hll(key=key, hyperloglogs=None):
    '''Runs a full simulation of an encrypted HLL merger.

    You can pass a list of hyperloglogs in.
    If you don't, this test will randomly simulate hyperloglogs from 100 hospitals.
    '''
    answer_dict = {}
    sketch_size = 32
    if hyperloglogs is None:
        raise ValueError
    else:
        num_hospitals = len(hyperloglogs)
        num_buckets = len(hyperloglogs[0])

    # Preprocessing
    print("Preprocessing time for key sharing: ", end="")
    start = time.time()
    key_template = key
    hospital_private_keys = [Integer(randint(2, int(key_template.p - 1))) for _ in range(num_hospitals)]
    key_y = Integer(1)
    hospital_keys = [construct_key(key_template, x) for x in hospital_private_keys]

    elapsed = time.time() - start
    print(elapsed)
    answer_dict['hospital_round0_time'] = elapsed

    start = time.time()
    print("Combine to form public key: ", end="")
    for x in hospital_private_keys:
        curr_y = pow(key_template.g, x, key_template.p)
        key_y = (curr_y * key_y) % key_template.p
    key_public = ElGamal.construct((key_template.p, key_template.g, key_y))
    elapsed = time.time() - start
    print(elapsed)
    answer_dict['hub_round0_time'] = elapsed

    print("Hospital-side computation round 1: Unroll & encrypt: ", end="")
    start = time.time()
    unrolled_hyperloglogs = [[unroll(i, sketch_size) for i in hll] for hll in hyperloglogs]
    encrypted_hyperloglogs = []
    for k, uhll in enumerate(unrolled_hyperloglogs):
        encrypted_hyperloglog = [[encrypt(key_public, x) for x in il] for il in uhll]
        encrypted_hyperloglogs.append(encrypted_hyperloglog)
    elapsed = time.time() - start
    print(elapsed)
    answer_dict['hospital_round1_time'] = elapsed
    start = time.time()

    print("Server-side computation round 1.5: collapse encrypted HLL: ", end="")
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
    elapsed = time.time() - start
    print(elapsed)
    answer_dict['hub_round1.5_time'] = elapsed

    print("Hospital-side computation round 2: compute shared secret parts: ", end="")
    # Hospital-side computation round 2. Get all shared secrets
    try:
        p = myglobals.parallel_pool
        myglobals.hospital_keys = hospital_keys
        myglobals.collapsed_enc_hll = collapsed_enc_hll
        all_hospital_secrets_with_time = p.map(get_hospital_shared_secrets, range(len(myglobals.hospital_keys)))
        all_hospital_secrets = [x[0] for x in all_hospital_secrets_with_time]
        round2_times = [x[1] for x in all_hospital_secrets_with_time]
        elapsed = sum(round2_times)
    except AttributeError:
        start = time.time()
        all_hospital_secrets = []
        for k in range(num_hospitals):
            secrets = []
            for i in range(num_buckets):
                secrets_by_bucket = []
                for j in range(sketch_size):
                    secrets_by_bucket.append(shared_secret(hospital_keys[k], collapsed_enc_hll[i][j]))
                secrets.append(secrets_by_bucket)
            all_hospital_secrets.append(secrets)
        elapsed = time.time() - start
    print(elapsed)
    answer_dict['hospital_round2_time'] = elapsed
    start = time.time()

    print("Server-side computation round 2.5. Combine shared secrets to decrypt: ", end="")
    # Server-side computation round 2.5. Combine shared secrets to decrypt
    combined_hll_unrolled = []
    for i in range(num_buckets):
        m_values = []
        for j in range(sketch_size):
            curr_secret = Integer(1)
            for k in range(num_hospitals):
                curr_secret = (curr_secret * all_hospital_secrets[k][i][j]) % key_public.p
            s_inv = pow(curr_secret, key_public.p - 2, key_public.p)  # modular multiplicative inverse https://stackoverflow.com/questions/4798654/modular-multiplicative-inverse-function-in-python
            m = (s_inv * collapsed_enc_hll[i][j].c2) % key_public.p
            sqrt_m = pow(m, (key_public.p + 1) // 4, key_public.p)  # taking square root
            if sqrt_m > key_public.p // 2:
                sqrt_m = key_public.p - sqrt_m
            m_values.append(sqrt_m)
        combined_hll_unrolled.append(m_values)
    elapsed = time.time() - start
    print(elapsed)
    answer_dict['hub_round_2.5_time'] = elapsed
    start = time.time()

    # Answer if the central server actually knew the secret key
    if False:
        exact_combined_hll_unrolled = []
        for i in range(num_buckets):
            m_values = []
            for j in range(sketch_size):
                sqrt_m = decrypt(key, collapsed_enc_hll[i][j])
                m_values.append(sqrt_m)
            exact_combined_hll_unrolled.append(m_values)

    print("Server-side computation round 2.9. Combine HLLs: ", end="")
    combined_hll = []
    for i in range(num_buckets):
        try:
            b = combined_hll_unrolled[i].index(1)
        except ValueError:
            b = 0
        combined_hll.append(b)
    elapsed = time.time() - start
    print(elapsed)
    answer_dict['hub_round2.9_time'] = elapsed
    start = time.time()

    return(combined_hll_unrolled, combined_hll, answer_dict)


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


def discrete_log4(key, x):
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
    print()
    return ans


def enc_test_counts(key, num_patients=10**3):
    num_hospitals = 100
    counts = [int(num_patients / num_hospitals) for _ in range(num_hospitals - 1)]
    counts.append(num_patients - sum(counts))
    assert sum(counts) == num_patients
    return enc_test_counts_given(key, counts)


def enc_test_counts_given(key, counts):
    answer_dict = {}
    # Preprocessing
    num_hospitals = len(counts)
    num_patients = sum(counts)
    key_template = key

    start = time.time()
    print("Preprocessing time for key sharing: ", end="")
    hospital_private_keys = [Integer(randint(2, int(key_template.p - 1))) for _ in range(num_hospitals)]
    hospital_keys = [construct_key(key_template, x) for x in hospital_private_keys]
    elapsed = time.time() - start
    print(elapsed)
    answer_dict['hospital_round0_time'] = elapsed

    print("Combine to form public key: ", end="")
    start = time.time()
    key_y = Integer(1)
    for x in hospital_private_keys:
        curr_y = pow(key_template.g, x, key_template.p)
        key_y = (curr_y * key_y) % key_template.p
    key_public = ElGamal.construct((key_template.p, key_template.g, key_y))
    elapsed = time.time() - start
    print(elapsed)
    answer_dict['hub_round0_time'] = elapsed

    start = time.time()
    print("Hospital-side computation round 1: encrypt counts: ", end="")
    # Hospital-side computation round 1

    assert len(counts) == num_hospitals
    print(counts)

    encrypted_counts = []
    for k, count in enumerate(counts):
        encrypted_count = encrypt(key_public, pow(Integer(4), Integer(count), key_public.p))
        encrypted_counts.append(encrypted_count)
    elapsed = time.time() - start
    print(elapsed)
    answer_dict['hospital_round1_time'] = elapsed
    start = time.time()

    print("Server-side computation round 1.5: collapse encrypted counts: ", end="")
    # Server-side computation round 1.5
    curr = encrypt(key_public, 1)
    for k in range(num_hospitals):
        curr = curr + encrypted_counts[k]
    encrypted_sum = curr

    elapsed = time.time() - start
    print(elapsed)
    answer_dict['hub_round1.5_time'] = elapsed
    start = time.time()

    print("Hospital-side computation round 2: compute shared secret parts: ", end="")
    # Hospital-side computation round 2. Get all shared secrets
    all_hospital_secrets = []
    for k in range(num_hospitals):
        secret = shared_secret(hospital_keys[k], encrypted_sum)
        all_hospital_secrets.append(secret)
    elapsed = time.time() - start
    print(elapsed)
    answer_dict['hospital_round2_time'] = elapsed
    start = time.time()

    print("Server-side computation round 2.5. Combine shared secrets to decrypt: ", end="")
    # Server-side computation round 2.5. Combine shared secrets to decrypt
    curr_secret = Integer(1)
    for k in range(num_hospitals):
        curr_secret = (curr_secret * all_hospital_secrets[k]) % key_public.p
    s_inv = pow(curr_secret, key_public.p - 2, key_public.p)  # modular multiplicative inverse https://stackoverflow.com/questions/4798654/modular-multiplicative-inverse-function-in-python
    m = (s_inv * encrypted_sum.c2) % key_public.p
    decrypted_sum = m

    assert(decrypted_sum == pow(Integer(4), num_patients, key_public.p))
    if _dl4_table is None:
        print("No precomputed log table. Doing discrete log problem")
        logged_sum = discrete_log4(key_public, decrypted_sum)
    else:
        print("Using precomputed log table")
        logged_sum = _dl4_table.dlog4(decrypted_sum)

    elapsed = time.time() - start
    print(elapsed)
    answer_dict['hub_round2.5_time'] = elapsed
    print("decrypted_sum: " + str(logged_sum))
    start = time.time()

    return(logged_sum, answer_dict)
