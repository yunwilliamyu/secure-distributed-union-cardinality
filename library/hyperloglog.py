#!/usr/bin/env python3
'''
Library of all the major functions used to securely share patient data between
hospitals.
'''
import numpy as np
from hashlib import sha1
import random


class HyperLogLog(object):
    '''Class that stores a 32-bit HyperLogLog sketch'''
    def __init__(self, k, salt=None):
        '''k is the number of buckets, and the cryptographic salt is a prefix
        before hashing

        If salt is not set to a string, then we cannot update the HLL.
        '''
        if salt is not None:
            if isinstance(salt, str):
                self.salt = salt
            else:
                raise TypeError("salt must be type string")
        else:
            self.salt = None
        self.k = k
        self.buckets = np.zeros(k, dtype=np.uint8)

    def update(self, L):
        '''Inserts a list of items in L into the sketch
        '''
        if not isinstance(self.salt, str):
            raise NotImplementedError("update is only implemented if a string salt is set")
        salt = self.salt
        bucket_num = self.k
        plist_crh = (sha1((salt + str(x)).encode()).digest() for x in L)
        plist_hash = ((int.from_bytes(temp[0:8], byteorder='big') % bucket_num,
                      min(64 + 1 - int.from_bytes(temp[8:16], byteorder='big').bit_length(), 31))
                      for temp in plist_crh)
        for i, v in plist_hash:
            self.buckets[i] = max(self.buckets[i], v)

    def safe_export(self, shuffle=None):
        '''Returns a Bytes object containing only the buckets, but not the salt.

           Used by hospitals to send the HyperLogLog to an untrusted hub.
        '''
        if shuffle is None:
            buckets = self.buckets
        else:
            salt = int.from_bytes(sha1(str(shuffle).encode()).digest()[0:4], byteorder='big')
            prng = np.random.RandomState(seed=salt)
            buckets = prng.permutation(self.buckets)
        return buckets.tobytes()

    @classmethod
    def safe_import(cls, byte_array):
        '''Unserializes a bytes object'''
        buckets = np.frombuffer(byte_array, dtype=np.uint8)
        obj = cls(len(buckets))
        obj.buckets = buckets
        return obj

    def set_salt(self, salt):
        '''Activates the update function by adding a salt to a HyperLogLog.
        Should only be used on imported HyperLogLogs without a salt.
        '''
        if isinstance(salt, str):
            if self.salt is None:
                self.salt = salt
            else:
                raise AttributeError("set_salt will not clobber an existing salt")
        else:
            raise TypeError("salt must be type string")

    def __eq__(self, other):
        '''Test that two HyperLogLogs are exaclty the same'''
        if (self.k == other.k) and np.array_equal(self.buckets, other.buckets) and (self.salt == other.salt):
            return True
        else:
            return False

    def sanitize(self):
        '''Removes the salt in preparation for sending to a central hub
        '''
        self.salt = None

    def __add__(self, other):
        '''Combines two HyperLogLog sketches together, forming the sketch of the union'''
        assert(self.k == other.k)
        assert(self.salt == other.salt)
        ans = HyperLogLog(self.k, salt=self.salt)
        ans.buckets = np.maximum(self.buckets, other.buckets)
        return ans

    def __radd__(self, other):
        '''Allows summation by defining what to do if other doesn't know how to add'''
        if other == 0:
            return self
        else:
            raise TypeError('Cannot add together types')

    def count(self):
        '''Returns cardinality based on HLL estimator.
        '''
        buckets = np.float64(np.asarray((self.buckets)))
        bucketnum = len(buckets)
        if bucketnum == 16:
            alpha = 0.673
        elif bucketnum == 32:
            alpha = 0.697
        elif bucketnum == 64:
            alpha = 0.709
        else:
            alpha = 0.7213 / (1 + 1.079 / bucketnum)

        res = alpha * bucketnum**2 * 1 / sum([2**-val for val in buckets])
        if res <= (5. / 2.) * bucketnum:
            V = sum(val == 0 for val in buckets)
            if V != 0:
                res2 = bucketnum * np.log(bucketnum / V)  # linear counting
            else:
                res2 = res
        elif res <= (1. / 30.) * (1 << 32):
            res2 = res
        else:
            res2 = -(1 << 32) * np.log(1 - res / (1 << 32))
        return res2

    def privacy(self, hll_freqs, shuffle=False):
        '''Calculates the number of patients who are revealed to less than m-anonymity

        hll_freqs will be the HyperLogLog bucket frequencies of the background population
        (e.g. the hospital as a whole, or the entire hospital network for secure MPC),
        as given by a HyperLogLog_freqs object.

        returns a size 10 vector anon, where anon[m] is the number of patients without m-anonymity

        if shuffle=True, we'll merge all the bucket_freqs because shuffling causes an adversary to
        be unable to distinguish between the buckets.
        '''
        assert(self.k == hll_freqs.k)
        bucket_privacy = np.zeros(self.k, dtype=np.uint8)
        if shuffle:
            sum_hll_freqs = hll_freqs.combined_bucket_freqs()
            bucket_freqs = [sum_hll_freqs for _ in range(self.k)]
        else:
            bucket_freqs = hll_freqs.bucket_freqs
        for i in range(self.k):
            if self.buckets[i] > 0:
                bucket_privacy[i] = bucket_freqs[i][self.buckets[i]]
            else:
                bucket_privacy[i] = 100
        anon = []
        for m in range(11):
            anon.append(sum(bucket_privacy < m))
        return anon


def hll_r2510(hll_priv_list):
    '''returns the number of patients who are completely identified,
    the number who have less than 5-anonymity, and the number who have
    less than 10-anonymity.

    Takes in a list of hll_privacy returns
    '''
    rev_pat2 = sum([x[2] for x in hll_priv_list])
    rev_pat5 = sum([x[5] for x in hll_priv_list])
    rev_pat10 = sum([x[10] for x in hll_priv_list])
    return (rev_pat2, rev_pat5, rev_pat10)


class HyperLogLog_freqs(object):
    '''
    This class is a HyperLogLog bucket frequencies object.

    We can return a standard HyperLogLog by looking at the maximum values
    for which the frequencies are greater than 0 in the frequencies object.

    The bucket_freqs object is used for privacy analyses. Suppose A is the HLL
    sketch of a subset of L. Then if a bucket i in A has value x, a patient lacks
    m-anonymity if the bucket_freqs[i][x] < m.
    '''
    def __init__(self, L, k, salt=""):
        '''L is the list of patients at a hospital

        We will generate a bucket_freqs frequency table specifying the frequency
        of items within each HLL bucket, with a maximum of 100.

        Note that we do not do any batching with this class method, so it may use
        a lot of RAM when computing all the hashes.
        '''
        bucket_num = k

        plist_crh = [sha1((salt + str(x)).encode()).digest() for x in L]
        plist_hash = [(int.from_bytes(temp[0:8], byteorder='big') % bucket_num,
                       min(64 + 1 - int.from_bytes(temp[8:16], byteorder='big').bit_length(), 31))
                      for temp in plist_crh]
        bucket_freqs = np.zeros((bucket_num, 32), dtype=np.uint8)
        for i, v in plist_hash:
            bucket_freqs[i][v] = min(bucket_freqs[i][v] + 1, 100)

        self.salt = salt
        self.k = k
        self.bucket_freqs = bucket_freqs

    def generate_HLL(self):
        '''Generates a HyperLogLog object out of a HyperLogLog_freqs object'''
        bucket_freqs = self.bucket_freqs
        bucket_num = self.k
        buckets = np.zeros(bucket_num, dtype=np.uint8)
        for i in range(bucket_num):
            try:
                buckets[i] = np.max(np.nonzero(bucket_freqs[i])[0])
            except ValueError:
                buckets[i] = 0  # Redundant because value already 0
        obj = HyperLogLog(self.k, salt=self.salt)
        obj.buckets = buckets
        return obj

    def combined_bucket_freqs(self):
        '''Returns the frequencies of all the buckets, combined together'''
        ans = np.zeros(32, dtype=np.uint64)
        for x in self.bucket_freqs:
            ans = ans + x
        return np.minimum(ans, 100).astype(np.uint8)

    def sanitize(self):
        '''Removes the salt in preparation for sending to a central hub
        '''
        self.salt = None

    def set_salt(self, salt):
        '''Activates the update function by adding a salt to a HyperLogLog.
        Should only be used on imported HyperLogLogs without a salt.
        '''
        if isinstance(salt, str):
            if self.salt is None:
                self.salt = salt
            else:
                raise AttributeError("set_salt will not clobber an existing salt")
        else:
            raise TypeError("salt must be type string")

    def safe_export(self, shuffle=None):
        '''Returns a Bytes object containing the frequencies. Stored by a hospital locally.

        Does not include the HLL salt or the shuffle salt, both of which the hospital
        ought to store elsewhere.
        '''
        if shuffle is None:
            bucket_freqs = self.bucket_freqs
        else:
            shuffle_salt = int.from_bytes(sha1(str(shuffle).encode()).digest()[0:4], byteorder='big')
            prng = np.random.RandomState(seed=shuffle_salt)
            bucket_freqs = prng.permutation(self.bucket_freqs)
        freq_bytes = [freqs.tobytes() for freqs in bucket_freqs]
        return b''.join(freq_bytes)

    @classmethod
    def safe_import(cls, byte_array):
        '''Unserializes a bytes object'''
        unshaped_freqs = np.frombuffer(byte_array, dtype=np.uint8)
        num_buckets = len(unshaped_freqs) // 32
        assert len(unshaped_freqs) == 32 * num_buckets
        obj = cls([], num_buckets)
        obj.sanitize()
        obj.bucket_freqs = np.reshape(unshaped_freqs, (num_buckets, 32))
        return obj

    def __add__(self, other):
        '''Adds together two HyperLogLog_freqs'''
        assert self.k == other.k
        assert self.salt == other.salt, "Sanity check that salts are the same"
        result = HyperLogLog_freqs([], self.k, self.salt)
        result.bucket_freqs = [np.minimum(A + B, 100) for A, B in zip(self.bucket_freqs, other.bucket_freqs)]
        return result

    def __radd__(self, other):
        '''Allows summation by defining what to do if other doesn't know how to add'''
        if other == 0:
            return self
        else:
            raise TypeError('Cannot add together types')

    def __eq__(self, other):
        '''Test that two HyperLogLog_freqs are exaclty the same'''
        if (self.k == other.k) and np.array_equal(self.bucket_freqs, other.bucket_freqs) and (self.salt == other.salt):
            return True
        else:
            return False
