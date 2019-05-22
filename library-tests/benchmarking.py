#!/usr/bin/env python3

import sys
import os
import time
import numpy as np
import hashlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import hyperloglog  # noqa: E402
from library import elgamal  # noqa: E402


def rehashing_id_benchmark(n, num_patients):
    ans = []
    A = np.arange(num_patients)
    for i in range(n):
        start = time.perf_counter()
        _ = [hashlib.sha1(("" + str(x)).encode()).digest() for x in A]
        ans.append(time.perf_counter() - start)
    return np.mean(ans)


def union_ids(n, num_patients, overlap, num_hospitals):
    ans = []
    patients = np.random.randint(int(num_patients / overlap), size=num_patients)
    hosp_lists = []
    patients_per_hospital = int(num_patients / num_hospitals)
    for i in range(num_hospitals):
        h = patients[i * patients_per_hospital:(i + 1) * patients_per_hospital]
        hosp_lists.append(h)
    for i in range(n):
        start = time.perf_counter()
        patient_set = np.concatenate(hosp_lists)
        patient_set = np.unique(patient_set)
        _ = len(patient_set)
        ans.append(time.perf_counter() - start)
    return np.mean(ans)


def hyperloglog_benchmark(n, k, num_patients):
    ans = []
    for i in range(n):
        start = time.perf_counter()
        A = hyperloglog.HyperLogLog(k, salt=str(i * 1000000))
        A.update(np.arange(num_patients))
        ans.append(time.perf_counter() - start)
    return np.mean(ans)


def hll_combine_benchmark(n, k, num_hospitals):
    ans = []
    hosp_hlls = [hyperloglog.HyperLogLog(k, salt="") for _ in range(num_hospitals)]
    for h in hosp_hlls:
        h.buckets = np.minimum(np.random.poisson(1, size=k), 30).astype(np.uint8)
    for i in range(n):
        start = time.perf_counter()
        A = sum(hosp_hlls)
        _ = A.count()
        ans.append(time.perf_counter() - start)
    return np.mean(ans)


def hll_mpc_benchmark(n, k, num_hospitals):
    ans = []
    A = hyperloglog.HyperLogLog(k, salt="")
    A.update(np.arange(10000))
    hospital_hlls = [A for _ in range(num_hospitals)]
    for i in range(n):
        _, ans_dict = elgamal.full_simulation_hll(elgamal.key, hospital_hlls)
        ans.append(ans_dict)
    d = ans[0].copy()
    for key in d:
        d[key] = np.mean([e[key] for e in ans])
    return d


def count_mpc_benchmark(n, num_hospitals):
    ans = []
    for i in range(n):
        _, ans_dict = elgamal.full_simulation_counts(elgamal.key, np.arange(num_hospitals))
        ans.append(ans_dict)
    d = ans[0].copy()
    for key in d:
        d[key] = np.mean([e[key] for e in ans])
    return d


def count_benchmark(n, num_hospitals):
    ans = []
    for i in range(n):
        # hospital_counts = np.random.randint(10**7, size=num_hospitals)
        hospital_counts = np.arange(num_hospitals)
        start = time.perf_counter()
        _ = sum(hospital_counts)
        per_hosp = (time.perf_counter() - start) / num_hospitals
        ans.append(per_hosp)
    return np.mean(ans)


def shuffle_benchmark(n, k):
    ans = []
    for i in range(n):
        bucket_counts = np.arange(k)
        start = time.perf_counter()
        prng = np.random.RandomState(np.random.randint(2**32))
        _ = prng.permutation(bucket_counts)
        elapsed = (time.perf_counter() - start)
        ans.append(elapsed)
    return np.mean(ans)
