#!/usr/bin/env python3
# flake8: noqa

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library.hyperloglog import *

class TestHyperLogLogSalts(unittest.TestCase):
    def test_salt_is_string(self):
        with self.assertRaises(TypeError) as context:
            _ = HyperLogLog(128, salt=5)
        self.assertTrue('salt must be type string' in str(context.exception))
    def test_salt_is_string_2(self):
        with self.assertRaises(TypeError) as context:
            A = HyperLogLog(128)
            A.set_salt(5)
        self.assertTrue('salt must be type string' in str(context.exception))
    def test_salt_not_clobbered(self):
        with self.assertRaises(AttributeError) as context:
            A = HyperLogLog(128, salt="a")
            A.set_salt("b")
        self.assertTrue('set_salt will not clobber an existing salt' in str(context.exception))
    def test_set_salt_successful(self):
        A = HyperLogLog(128)
        A.set_salt("a")
        self.assertEqual(A.salt, "a")

class TestHyperLogLogBase(unittest.TestCase):
    def test_hll_works(self):
        A = HyperLogLog(8, salt="")
        A.update(np.arange(10000))
        B = np.array([12, 12, 13, 12, 10, 11, 12, 10], dtype=np.uint8)
        self.assertTrue(np.array_equal(A.buckets, B))
    def test_hll_works2(self):
        A = HyperLogLog(8, salt="applesauce")
        A.update(np.arange(10000))
        B = np.array([14, 14, 10, 13, 11,  9, 12, 11], dtype=np.uint8)
        self.assertTrue(np.array_equal(A.buckets, B))
    def test_count(self):
        A = HyperLogLog(100, salt="")
        A.update(np.arange(10000))
        self.assertTrue(np.isclose(A.count(), 10000, rtol=0.3))
    def test_count2(self):
        A = HyperLogLog(2**15, salt="")
        A.update(np.arange(100000))
        self.assertTrue(np.isclose(A.count(), 100000, rtol=0.1))

class TestHyperLogLogSerialization(unittest.TestCase):
    def test_roundtrip(self):
        A = HyperLogLog(128, salt="")
        A.update(np.arange(10000))
        A_bytes = A.safe_export()
        B = HyperLogLog.safe_import(A_bytes)
        B.set_salt("")
        self.assertEqual(A, B)

class TestHyperLogLogFrequencies(unittest.TestCase):
    def test_sameness(self):
        A = HyperLogLog(128, salt="")
        A.update(np.arange(10000))
        B = HyperLogLog_freqs(np.arange(10000), 128, salt="")
        self.assertEqual(A,B.generate_HLL())
    def test_sanitize(self):
        A = HyperLogLog_freqs(np.arange(10000), 128, salt="")
        A.sanitize()
        B = A.generate_HLL()
        with self.assertRaises(NotImplementedError) as context:
            B.update(np.arange(5))
        self.assertTrue('update is only implemented if a string salt is set' in str(context.exception))
    def test_summation(self):
        A = HyperLogLog_freqs(np.arange(10000), 128, salt="applesauce")
        B = HyperLogLog_freqs(np.arange(10000,20000), 128, salt="applesauce")
        self.assertEqual(A + B, sum([A,B]))
    def test_summation2(self):
        A = HyperLogLog_freqs(np.arange(10000), 128, salt="")
        B = HyperLogLog_freqs(np.arange(10000,20000), 128, salt="applesauce")
        with self.assertRaises(AssertionError) as context:
            C = A + B
        self.assertTrue('Sanity check that salts are the same' in str(context.exception))
    def test_roundtrip(self):
        A = HyperLogLog_freqs(np.arange(10000), 128, salt="")
        A_bytes = A.safe_export()
        B = HyperLogLog_freqs.safe_import(A_bytes)
        B.set_salt("")
        self.assertEqual(A, B)


if __name__ == '__main__':
    unittest.main()
