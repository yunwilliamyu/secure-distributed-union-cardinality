#!/usr/bin/env python3

import generate_hospital_lists
# from generate_hospital_lists import *
from generate_hospital_lists import hyperloglog, query1, sha1, simulate_population_and_hospitals_power_law


import sys
import myglobals
import elgamal
import pickle
import gc
import time

from multiprocessing import Pool


def query_oneparam(param):
    j = param['j']
    seed = param['seed']
    if j >= 0:
        num_matching = 10**j
    else:
        num_matching = -1
    print("\n====\nWorking on {} patients set\n=====\n".format(num_matching))

    return query1(
        myglobals.hospital_patients_array,
        num_matching=num_matching,
        hospital_full_hlls15=myglobals.hospital_full_hlls15,
        random_seed=314 * seed + j)


def hyperloglog_of_patients_array_by_index15(i):
    start = time.perf_counter()
    ans = hyperloglog(myglobals.hospital_patients_array[i], bucket_num=2**15)
    elapsed = time.perf_counter() - start
    return (ans, elapsed)


def sha1_with_time_all(i):
    start = time.perf_counter()
    _ = [sha1(str(x).encode()).digest() for x in myglobals.hospital_patients_array[i]]
    elapsed = time.perf_counter() - start
    return elapsed


def single(seed=1000):
    generate_hospital_lists._global_salt = str(seed) + 'a'  # Allows us to randomize the experiment while still keeping patient IDs consecutive
    myglobals.hospital_patients_array = simulate_population_and_hospitals_power_law(random_seed=seed)
    myglobals.summed_hospital_num = sum([len(x) for x in myglobals.hospital_patients_array])

    # myglobals.hospital_full_hlls = [hyperloglog(x) for x in myglobals.hospital_patients_array]
    # myglobals.hospital_full_hlls15 = [hyperloglog(x, bucket_num=2**15) for x in myglobals.hospital_patients_array]

    # Do precomputations for the full patient set in parallel to save time
    with Pool(6) as p:
        print('Precomputing HLL-15')
        res_full15 = p.map_async(hyperloglog_of_patients_array_by_index15, range(len(myglobals.hospital_patients_array)))
        res_full_results15 = res_full15.get()
        myglobals.hospital_full_hlls15 = [x[0] for x in res_full_results15]
        myglobals.hospital_full_hlls15_hashing_times = [x[1] for x in res_full_results15]

    with Pool(6) as p:
        print('Pretiming Sha1 all')
        sha1_times = p.map_async(sha1_with_time_all, range(len(myglobals.hospital_patients_array)))
        myglobals.sha1_hashing_times = sha1_times.get()

    gc.collect()

    # load precomputed discrete log table for Elgamal if exists
    try:
        f_tab = open('TAB-table.pickle', 'rb')
        elgamal._dl4_table = elgamal.discrete_log4_table.fromtable(elgamal.key, pickle.load(f_tab))
    except IOError:
        pass

    param_base = {'seed': seed}
    experiment_param_list = [dict(param_base, **{'j': j}) for j in range(0, 8)] + [dict(param_base, **{'j': -1})]
    with Pool(6) as p:
        experiments = p.map(query_oneparam, experiment_param_list)

    with open("results" + "_" + str(seed) + "-" + str(int(time.time())) + ".dat", "a") as f:
        print(experiments, file=f)
    return experiments


if __name__ == '__main__':
    print("Seed = {}".format(sys.argv[1]))
    single(seed=int(sys.argv[1]))
