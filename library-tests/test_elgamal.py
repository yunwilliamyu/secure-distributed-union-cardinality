#!/usr/bin/env python3

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import elgamal  # noqa: E402
from library.elgamal import *  # noqa: E402
import library.hyperloglog  # noqa: E402

class TestElgamal(unittest.TestCase):
    def test_roundtrip(self):
        E = encrypt(key_public, 4)
        D = decrypt(key, E)
        self.assertEqual(D, 4)
    def test_serialize(self):
        E = encrypt(key_public, 16)
        bytestring = E.export_val()
        E2 = CipherText.byte_init(key_public, bytestring)
        self.assertEqual(E2, E)
    def test_add(self):
        E = encrypt(key_public, 4)
        E2 = encrypt(key_public, 4)
        E3 = E + E2
        D3 = decrypt(key, E3)
        self.assertEqual(D3, 16)
    def test_private_equality(self):
        E = encrypt(key_public, 4)
        E2 = encrypt(key_public, 4)
        E3 = E.private_equality_test(E2)
        D3 = decrypt(key, E3)
        self.assertEqual(D3, 1)
    def test_not_private_equality(self):
        E = encrypt(key_public, 4)
        E2 = encrypt(key_public, 9)
        E3 = E.private_equality_test(E2)
        D3 = decrypt(key, E3)
        self.assertNotEqual(D3, 1)

class TestUnroll(unittest.TestCase):
    def test_unroll(self):
        self.assertEqual(unroll(5,10), [4,4,4,4,4,1,1,1,1,1])

class TestKeySerialization(unittest.TestCase):
    def test_public(self):
        B = serialize_ElGamalPublicKey(key_public)
        K = deserialize_ElGamalPublicKey(B)
        self.assertTrue((K.p == key_public.p) and (K.g == key_public.g)
                and (K.y == key_public.y))
    def test_private(self):
        B = serialize_ElGamalPrivateKey(key)
        K = deserialize_ElGamalPrivateKey(B)
        self.assertTrue((K.p == key.p) and (K.g == key.g)
                and (K.y == key.y) and (K.x == key.x))
    def test_mixed(self):
        B = serialize_ElGamalPrivateKey(key)
        with self.assertRaises(AssertionError) as context:
            K = deserialize_ElGamalPublicKey(B)
        self.assertTrue('Wrong bytestring length' in str(context.exception))
    def test_mixed2(self):
        B = serialize_ElGamalPublicKey(key_public)
        with self.assertRaises(ValueError) as context:
            K = deserialize_ElGamalPrivateKey(B)
        self.assertTrue('Invalid ElGamal key components' in str(context.exception))

class TestDistributedKeyGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        key_template = key_public
        num_hospitals = 100
        round0a = [hospital_round0a(key_template) for _ in range(num_hospitals)]
        round0a_transmissions = [x['transmit'] for x in round0a]
        round0b = hub_round0b(key_template, round0a_transmissions)
        kp = round0b['key_public']
        cls.key_public_distributed = kp

        sum_x = sum([x['key_x'] for x in round0a], Integer(0)) % (key_template.p-1)
        if sum_x == 0:
            sum_x = key_template.p - 1
        kp2 = construct_key(key_template, sum_x)
        cls.key_private_central = kp2
    def test_distributed_key_matches_private_generation(self):
        kp = self.key_public_distributed
        kp2 = self.key_private_central
        self.assertTrue((kp.p == kp2.p) and (kp.g == kp2.g)
                and (kp.y == kp2.y))
    def test_key_pair(self):
        E = encrypt(self.key_public_distributed, 4)
        D = decrypt(self.key_private_central, E)
        self.assertEqual(D, 4)

class TestEncryptedHLL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestEncryptedHLL, cls).setUpClass()
        key_template = elgamal.key_public
        num_hospitals = 3
        round0a = [hospital_round0a(key_template) for _ in range(num_hospitals)]
        cls.round0a = round0a
        round0a_transmissions = [x['transmit'] for x in round0a]
        round0b = hub_round0b(key_template, round0a_transmissions)
        key_public = round0b['key_public']
        cls.hospital_keys = [x['private_key'] for x in round0a]
        sum_x = sum([x['key_x'] for x in round0a], Integer(0)) % (key_template.p-1)
        if sum_x == 0:
            sum_x = key_template.p - 1
        key = construct_key(key_template, sum_x)

        cls.key_public = key_public
        cls.key = key

        A = hyperloglog.HyperLogLog(8, salt="")
        A.update(np.arange(1000))
        B = hyperloglog.HyperLogLog(8, salt="")
        B.update(np.arange(1000,1500))
        C = hyperloglog.HyperLogLog(8, salt="")
        C.update(np.arange(1500,2000))

        cls.hosps_hll = [A, B, C]
        cls.hosps_ehll = [hospital_round1a_hll(key_public, hll) for hll in cls.hosps_hll]
        cls.hosps_transmit = [x['transmit'] for x in cls.hosps_ehll]
        cls.round1b = hub_round1b_hll(key_public, cls.hosps_transmit)

        hospital_keys = [x['private_key'] for x in cls.round0a]
        cls.round2a = [hospital_round2a_hll(hk, cls.round1b['transmit']) for hk in hospital_keys]
    def test_roundtrip(self):
        round1a = hospital_round1a_hll(self.key_public, self.hosps_hll[0])
        B = decode_encrypted_hll_bytestring(self.key_public, round1a['transmit'])
        self.assertEqual(round1a['encrypted_hll'], B)
    def test_encrypted_collapse(self):
        collapsed_uhll = [[decrypt(self.key, x) for x in buck] for buck in self.round1b['collapsed_enc_hll']]
        collapsed_hll = reroll_hll(collapsed_uhll)
        self.assertTrue(np.array_equal(sum(self.hosps_hll).buckets, collapsed_hll))
    def test_roundtrip_shared_secrets(self):
        key_public = self.key_public
        X = decode_hospital_shared_secrets_bytestring(key_public, self.round2a[0]['transmit'])
        self.assertEqual(X, self.round2a[0]['secrets'])
    def test_encrypted_merging(self):
        round2b = hub_round2b_hll(self.key_public, self.round1b['transmit'], [x['transmit'] for x in self.round2a])
        self.assertTrue(np.array_equal(round2b['combined_buckets'], sum(self.hosps_hll).buckets))
    def test_full_simulation_hll(self):
        ans, _ = full_simulation_hll(elgamal.key_public, self.hosps_hll)
        C = sum(self.hosps_hll)
        C.sanitize()
        self.assertEqual(ans, C)


class TestEncryptedCount(unittest.TestCase):
    def test_full_simulation_counts(self):
        A = [10,20,30,40,50]
        eac, _ = full_simulation_counts(elgamal.key_public, A)
        self.assertEqual(sum(A), eac)


if __name__ == '__main__':
    unittest.main()
