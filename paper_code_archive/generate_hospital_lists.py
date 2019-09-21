#!/usr/bin/env python3
'''Generates a network of 100 hospitals, with 10k to 10M patients per hospital, and 100M total patients


'''

import numpy as np
from hashlib import sha1
import time
import elgamal
# from functools import reduce
# import gc
import myglobals
import array
from itertools import repeat


def simulate_population_and_hospitals_power_law(random_seed=10000, num_actual_population=10**8):
    '''This new model first generates a power law distribution of hospital sizes,
    places them in a 2D unit square, and randomly assigns second hospitals by
    probability inversely proportional to the square distance to the first hospital
    and proportional to the size of the new hospital.

    i.e. patients are more likely to go to nearby hospitals by square distance,
    but also more likely to go to big hospitals by population
    '''
    '''Parameters'''
    num_hospitals = 100
    assert(num_actual_population < 2**32)  # since we use np.uint32

    prng = np.random.RandomState(random_seed)  # Set random state
    '''We start by generating the hospital locations'''
    hospital_locations = [p for p in zip(prng.uniform(size=num_hospitals), prng.uniform(size=num_hospitals))]

    '''On each node of the network, we assign a random initial population according to a power law distribution'''
    X = prng.lognormal(sigma=1.2, size=num_hospitals)
    X = X / sum(X) * num_actual_population
    X = np.round(X)
    X[0] = X[0] + num_actual_population - sum(X)  # Correct for rounding error
    hospital_sizes = [int(y) for y in X]

    assert sum(hospital_sizes) == num_actual_population

    '''We assign a unique numeric identifier for each patient (something like a SSN).
    Here we just count up from 0, to make things easier, but the choice of identifier does not matter
    Then we assign the patient to a home hospital.
    '''
    hospital_patients_array = [array.array('L') for _ in range(num_hospitals)]
    curr_patient = 0
    for i, npa in enumerate(hospital_sizes):
        hospital_patients_array[i].extend(np.arange(curr_patient, curr_patient + npa, dtype=np.uint32))
        curr_patient = curr_patient + npa
    print("Generated home hospitals")

    '''We now assign every patient to between 0 and 9 additional hospitals. We do this
    by adding a hospital with prob 1/9, a total of 9 times, so in expectation, every patient
    will appear at 2 hospitals, but some patients at only 1, and other patients
    at up to 10.

    The probability of a patient from hospital A being assigned to a particular hospital B is proportional
    to the size of hospital B and inversely proportional to the squared distance.

    Note that we discount the probability of a patient appearing twice at another hospital.
    If this happens, the patient just appears once at that hospital, but this does mean that
    patents in expectation appear at *slightly* less than 2 hospitals.
    '''
    hospital_patients_array_extension = [array.array('L') for _ in range(num_hospitals)]
    for h in range(num_hospitals):
        hospitals_per_patient = prng.binomial(9, 1 / 9, size=hospital_sizes[h])

        squared_distances = [(p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 for p1, p2 in zip(hospital_locations, repeat(hospital_locations[h]))]
        # distances = [np.sqrt(x) for x in squared_distances]
        transition_probs = np.zeros(num_hospitals)
        for i in range(num_hospitals):
            try:
                transition_probs[i] = 1 / squared_distances[i] * hospital_sizes[i]
            except ZeroDivisionError:
                transition_probs[i] = 0
        transition_probs[h] = 0  # Make sure there's no transition probability back to the original hospital
        transition_probs = transition_probs / sum(transition_probs)

        draws = prng.choice(num_hospitals, size=sum(hospitals_per_patient), p=transition_probs)
        j = 0
        for i, x in enumerate(hospital_patients_array[h]):
            for _ in range(hospitals_per_patient[i]):
                hospital_patients_array_extension[draws[j]].append(x)
                j = j + 1
            if x % 10000000 == 0 and x > 0:
                print(x)
    for h in range(num_hospitals):
        hospital_patients_array[h].extend(hospital_patients_array_extension[h])

    return hospital_patients_array


# This is a hack. Basically allows us to randomize the experiment without changing the patient IDs
_global_salt = None
_bucket_num = 128


def hyperloglog(hosp_patient_list, candidates=-1, bucket_num=None, salt=None):
    '''hosp_patient_list is the list of patients at a hospital

    candidates is a list of people matching the condition at hand
        Note that if you pass is candidates=-1, then we will assume
        that every patient matches.

    Returns the normal HLL buckets, and also a frequency of items within each HLL bucket
    Currently, the frequency maxes out at 100.

    The frequency is primarily useful for privacy analyses when comparing against
    the frequencies that would be present when candidates=-1

    If no salt is specified, will use module level global variable "global_salt" instead,
    which defaults to an empty string
    '''
    if candidates is -1:
        plist = hosp_patient_list
    else:
        plist = np.intersect1d(hosp_patient_list, candidates)

    # Only use global salt if no salt is specified
    if _global_salt is not None:
        if salt is None:
            salt = _global_salt
    else:
        salt = ""
    if _bucket_num is not None:
        if bucket_num is None:
            bucket_num = _bucket_num
    else:
        raise ValueError

    plist_crh = [sha1((salt + str(x)).encode()).digest() for x in plist]
    plist_hash = [(int.from_bytes(temp[0:8], byteorder='big') % bucket_num,
                   min(64 + 1 - int.from_bytes(temp[8:16], byteorder='big').bit_length(), 31))
                  for temp in plist_crh]
    bucket_freqs = [np.zeros(32, dtype=np.ubyte) for _ in range(bucket_num)]
    for i, v in plist_hash:
        bucket_freqs[i][v] = min(bucket_freqs[i][v] + 1, 100)
    buckets = np.zeros(bucket_num, np.ubyte)
    for i in range(bucket_num):
        try:
            buckets[i] = np.max(np.nonzero(bucket_freqs[i])[0])
        except ValueError:
            buckets[i] = 0  # Redundant because value already 0
    return (buckets, bucket_freqs)


def hyperloglog_reduce_buckets(hll, new_bucket_num):
    '''Takes a [buckets, bucket_freqs] from the hyperloglog function and reduces
       the number of buckets. The new_bucket_num must divide len(buckets)'''
    buckets, bucket_freqs = hll
    assert (len(buckets) % new_bucket_num) == 0
    div_ratio = len(buckets) // new_bucket_num
    new_bucket_freqs = [np.zeros(32, dtype=np.ubyte) for _ in range(new_bucket_num)]
    for i in range(new_bucket_num):
        for j in range(div_ratio):
            new_bucket_freqs[i] = new_bucket_freqs[i] + bucket_freqs[i + j * new_bucket_num]
            new_bucket_freqs[i] = np.minimum(new_bucket_freqs[i], 100)
    del buckets
    del bucket_freqs
    new_buckets = np.zeros(new_bucket_num, np.ubyte)
    for i in range(new_bucket_num):
        try:
            new_buckets[i] = np.max(np.nonzero(new_bucket_freqs[i])[0])
        except ValueError:
            new_buckets[i] = 0  # Redundant because value already 0
    return (new_buckets, new_bucket_freqs)


def bucket_freq_sum(X):
    return sum([x.astype(np.uint32) for x in X])


def hll_privacy(buckets, bucket_freqs, shuffle=False):
    '''Calculates the number of patients who are revealed to less than m-anonymity

    buckets will be the HyperLogLog sketch of the query population

    bucket_freqs will be the HyperLogLog bucket frequencies of the background population
    (e.g. the hospital as a whole, or the entire hospital network for secure MPC

    returns a size 10 vector anon, where anon[m] is the number of patients without m-anonymity

    if shuffle=True, we'll merge all the bucket_freqs because shuffling causes an adversary to
    be unable to distinguish between the buckets.
    '''
    assert(len(buckets) == len(bucket_freqs))
    bucket_privacy = [0 for _ in range(len(buckets))]

    if shuffle:
        bucket_agg_freqs = bucket_freq_sum(bucket_freqs)
        bucket_freqs = [bucket_agg_freqs for _ in range(len(bucket_freqs))]
    for i in range(len(buckets)):
        if buckets[i] > 0:
            bucket_privacy[i] = bucket_freqs[i][buckets[i]]
        else:
            bucket_privacy[i] = 100000  # The bucket is completely private if there is no patient in it
    anon = []
    for m in range(11):
        anon.append(sum(np.asarray(bucket_privacy) < m))
    return anon


def hll_r2510(hll_priv):
    '''returns the number of patients who are completely identified,
    the number who have less than 5-anonymity, and the number who have
    less than 10-anonymity.

    Takes in a list of hll_privacy returns
    '''
    rev_pat2 = sum([x[2] for x in hll_priv])
    rev_pat5 = sum([x[5] for x in hll_priv])
    rev_pat10 = sum([x[10] for x in hll_priv])
    return (rev_pat2, rev_pat5, rev_pat10)


def hll_estimator(buckets):
    '''Returns cardinality based on HLL estimator, given a list of buckets,
    each with a single integer specifying the maximum number of leading zeros
    within that bucket.
    '''
    if len(buckets) == 1:
        buckets = [np.float64(np.asarray(buckets))]  # because np.float64 converts a length-1 array to a scalar
    else:
        buckets = np.float64(np.asarray((buckets)))
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


def hll_combine_all(list_of_hlls):
    '''Combines a list of HLLs with both buckets and bucket_freqs into a single HLL (with buckets and bucket_freqs)'''
    try:
        n = len(list_of_hlls[0][0])
    except IndexError:
        n = 128
    hll = hyperloglog([], bucket_num=n)
    for x in list_of_hlls:
        for j in range(len(hll[0])):
            hll[0][j] = max(hll[0][j], x[0][j])
            hll[1][j] = np.ubyte(np.minimum(np.uint64(hll[1][j]) + np.uint64(x[1][j]), 100))
    return hll


def hll_combine(list_of_hlls):
    '''Old function. Throw error if trying to use'''
    assert(False)


def hll_combine_buckets(list_of_hlls):
    '''Combines a list of HLLs, where each hll is a list of buckets, into a single HLL

    Does not combine bucket frequencies, which one should do separately with the hll_combine_all function'''
    try:
        n = len(list_of_hlls[0])  # bucket_num
    except IndexError:
        n = 128
    hll = np.zeros(n, np.ubyte)
    for x in list_of_hlls:
        for j in range(n):
            hll[j] = max(hll[j], x[j])
    return hll


def aggregate_count_stats(hospital_patients, num_patients, mask=False, mpc=False):
    '''Aggregate count stats'''
    title = "agg"
    if mask:
        hospital_counts = []
        for x in hospital_patients:
            if len(x) == 0:
                ax = 0
            else:
                ax = max(len(x), 10)
            hospital_counts.append(ax)
        # hospital_counts = [max(len(x), 10) for x in hospital_patients]
        title = title + "_mask"
    else:
        hospital_counts = [len(x) for x in hospital_patients]
    if not mpc:
        start = time.perf_counter()
        est_pat = sum(hospital_counts)
        est_min = max(hospital_counts)
        hub_elapsed = time.perf_counter() - start
        rev_pat2 = sum(np.logical_and(np.asarray(hospital_counts) > 0, np.asarray(hospital_counts) < 2))
        rev_pat5 = sum(np.logical_and(np.asarray(hospital_counts) > 0, np.asarray(hospital_counts) < 5))
        rev_pat10 = sum(np.logical_and(np.asarray(hospital_counts) > 0, np.asarray(hospital_counts) < 10))
        title = title + "_stats"
        agg_stats = {
            "title": title,
            "num_patients": num_patients,
            "estimate_min": est_min,
            "estimate": est_pat,
            "revealed": rev_pat2,
            "revealed_5anon": rev_pat5,
            "revealed_10anon": rev_pat10,
            "hub_elapsed": hub_elapsed
        }
    else:
        est_pat, answer_dict = elgamal.enc_test_counts_given(elgamal.key, hospital_counts)
        title = "agg_mpc_stats"
        agg_stats = {
            "title": title,
            "num_patients": num_patients,
            "estimate": est_pat,
            "revealed": 0,
            "revealed_5anon": 0,
            "revealed_10anon": 0
        }
        agg_stats.update(answer_dict)
    return agg_stats


def hyperloglog_with_time(param):
    x = param[0]
    bucket_num = param[1]
    start = time.perf_counter()
    ans = hyperloglog(x, bucket_num=bucket_num)
    total_time = time.perf_counter() - start
    return (ans, total_time)


def hyperloglog_generic_stats(hospital_patients, hospital_full_hlls, num_patients, mpc=False):
    '''Generic hyperloglog stats for a run, including data for:
            naive
            mask
            rehash
            shuffle
            (mpc)
            (shuffle+mpc)

        The last two only if mpc==True
    '''
    combined_full_hll = hll_combine_all(hospital_full_hlls)

    ans = []

    bucket_num = len(hospital_full_hlls[0][0])
    summed_hospital_num = sum([len(x) for x in hospital_patients])
    try:
        if myglobals.summed_hospital_num != summed_hospital_num:
            raise ValueError('Not the right number of patients for precomputed list')
        hashing_times = []
        hospital_hlls = []
        for i, hll in enumerate(myglobals.hospital_full_hlls15):
            start = time.perf_counter()
            hospital_hlls.append(hyperloglog_reduce_buckets(hll, bucket_num))
            hashing_times.append(time.perf_counter() - start + myglobals.hospital_full_hlls15_hashing_times[i])
    except (AttributeError, ValueError):
        hospital_hlls = []
        hashing_times = []
        for x in hospital_patients:
            start = time.perf_counter()
            hospital_hlls.append(hyperloglog(x, bucket_num=bucket_num))
            hashing_times.append(time.perf_counter() - start)
    hosp_elapsed = sum(hashing_times)
    myglobals.hospital_patients = hospital_patients

    # Combined HLL with both buckets and frequencies. Used for privacy analysis and sanity checking,
    # but not the runtime.
    combined_hll = hll_combine_all(hospital_hlls)

    # We'll combine just the buckets here and measure the time actually spent by the hub
    start = time.perf_counter()
    combined_hll_buckets = hll_combine_buckets([x[0] for x in hospital_hlls])
    est_pat = hll_estimator(combined_hll_buckets)
    hub_elapsed = time.perf_counter() - start

    assert np.array_equal(combined_hll[0], combined_hll_buckets), "Internal logic error. Buckets should be the same from both functions"

    base_title = "hll" + str(int(np.log(bucket_num) / np.log(2)))
    print(base_title, num_patients)
    base_stats = {
        "title": None,
        "num_patients": num_patients,
        "estimate": est_pat
    }

    '''rehashed hll'''
    hll_rehashed_stats = base_stats.copy()
    hll_rehashed_stats['title'] = base_title + '_rehashed_stats'
    hll_rehashed_stats['revealed'] = 0
    hll_rehashed_stats['revealed_5anon'] = 0
    hll_rehashed_stats['revealed_10anon'] = 0
    hll_rehashed_stats['hub_elapsed'] = hub_elapsed
    hll_rehashed_stats['hosp_elapsed'] = hosp_elapsed
    hll_rehashed_stats['max_hosp_elapsed'] = max(hashing_times)
    ans.append(hll_rehashed_stats)

    '''ordinary hll'''
    hll_revealed = [hll_privacy(hospital_hlls[i][0], hospital_full_hlls[i][1]) for i in range(len(hospital_hlls))]
    rp2, rp5, rp10 = hll_r2510(hll_revealed)
    hll_stats = base_stats.copy()
    hll_stats['title'] = base_title + "_stats"
    hll_stats['hub_elapsed'] = hub_elapsed
    hll_stats['revealed'] = rp2
    hll_stats['revealed_5anon'] = rp5
    hll_stats['revealed_10anon'] = rp10
    ans.append(hll_stats)

    '''shuffled hll'''
    hll_shuffle_revealed = [hll_privacy(hospital_hlls[i][0], hospital_full_hlls[i][1], shuffle=True) for i in range(len(hospital_hlls))]
    srp2, srp5, srp10 = hll_r2510(hll_shuffle_revealed)
    hll_shuffle_stats = base_stats.copy()
    hll_shuffle_stats['title'] = base_title + "_shuffle_stats"
    hll_shuffle_stats['hub_elapsed'] = hub_elapsed
    hll_shuffle_stats['revealed'] = srp2
    hll_shuffle_stats['revealed_5anon'] = srp5
    hll_shuffle_stats['revealed_10anon'] = srp10
    ans.append(hll_shuffle_stats)

    if mpc:
        '''mpc hll'''
        mpc_hll_unrolled, mpc_combined_hll, mpc_answer_dict = elgamal.enc_test_hll(hyperloglogs=[x[0] for x in hospital_hlls])
        mpc_hll_revealed = [hll_privacy(x[0], combined_full_hll[1]) for x in hospital_hlls]
        mrp2, mrp5, mrp10 = hll_r2510(mpc_hll_revealed)
        hll_mpc_stats = {
            "title": base_title + "_mpc_stats",
            "num_patients": num_patients,
            "estimate": hll_estimator(mpc_combined_hll),
            "revealed": mrp2,
            "revealed_5anon": mrp5,
            "revealed_10anon": mrp10,
        }
        hll_mpc_stats.update(mpc_answer_dict)
        ans.append(hll_mpc_stats)
        '''HLL_MPC_shuffled_stats'''
        mpc_hll_shuffle_revealed = [hll_privacy(x[0], combined_full_hll[1], shuffle=True) for x in hospital_hlls]
        msrp2, msrp5, msrp10 = hll_r2510(mpc_hll_shuffle_revealed)
        hll_mpc_shuffle_stats = {
            "title": base_title + "_mpc_shuffle_stats",
            "num_patients": num_patients,
            "estimate": hll_estimator(mpc_combined_hll),
            "revealed": msrp2,
            "revealed_5anon": msrp5,
            "revealed_10anon": msrp10,
        }
        hll_mpc_shuffle_stats.update(mpc_answer_dict)
        ans.append(hll_mpc_shuffle_stats)

    '''HLL masking w/ agg_mask for any reveals of privacy less than 10'''
    hospital_counts = [max(len(x), 10) for x in hospital_patients]
    hll_revealed_mask = [hll_privacy(hospital_hlls[i][0], hospital_full_hlls[i][1]) for i in range(len(hospital_hlls))]
    safe_hospitals_bool = [x[10] == 0 for x in hll_revealed_mask]
    safe_hospitals_hll = []
    unsafe_count = 0
    maximum_unsafe_hospital_size = 0
    for i, shb in enumerate(safe_hospitals_bool):
        if shb:
            safe_hospitals_hll.append(hospital_hlls[i])
        else:
            unsafe_count = unsafe_count + hospital_counts[i]
            maximum_unsafe_hospital_size = max(hospital_counts[i], maximum_unsafe_hospital_size)

    # Combined HLL with both buckets and frequencies. Used for privacy analysis and sanity checking,
    # but not the runtime.
    safe_combined_hll = hll_combine_all(safe_hospitals_hll)

    # We'll combine just the buckets here and measure the time actually spent by the hub
    start = time.perf_counter()
    safe_combined_hll_buckets = hll_combine_buckets([x[0] for x in safe_hospitals_hll])
    est_pat = hll_estimator(safe_combined_hll_buckets) + unsafe_count
    est_min = max(hll_estimator(safe_combined_hll_buckets), maximum_unsafe_hospital_size)
    hub_elapsed = time.perf_counter() - start

    assert np.array_equal(safe_combined_hll[0], safe_combined_hll_buckets), "Internal logic error. Buckets should be the same from both functions"

    hll_mask_stats = {
        "title": base_title + "_mask_stats",
        "num_patients": num_patients,
        "estimate": est_pat,
        "estimate_min": est_min,
        "revealed": 0,
        "revealed_5anon": 0,
        "revealed_10anon": 0,
        "hub_elapsed": hub_elapsed
    }
    ans.append(hll_mask_stats)
    return ans


def query1(hospital_patients_array, num_matching=-1, random_seed=1, hospital_full_hlls15=None):
    '''Runs a query of all patients at all hospitals that matches a random subset num_matching of the patients

        if num_matching==-1, then we select all patients

        random_seed can be used to choose a different subsample of matching patients

        hospital_full_hlls can be preprocessed for speed
        hospital_full_hlls15 can also be preprocessed for speed
    '''
    ans = []

    if hospital_full_hlls15 is None:
        hospital_full_hlls15 = [hyperloglog(x, bucket_num=2**15) for x in hospital_patients_array]

    hospital_full_hlls0 = [hyperloglog_reduce_buckets(hll, 2**0) for hll in hospital_full_hlls15]
    hospital_full_hlls1 = [hyperloglog_reduce_buckets(hll, 2**1) for hll in hospital_full_hlls15]
    hospital_full_hlls4 = [hyperloglog_reduce_buckets(hll, 2**4) for hll in hospital_full_hlls15]
    hospital_full_hlls7 = [hyperloglog_reduce_buckets(hll, 2**7) for hll in hospital_full_hlls15]

    prng = np.random.RandomState(random_seed)  # Set random state
    if num_matching >= 0:
        patient_set = np.sort(prng.choice(10**8, num_matching, replace=False))  # Get patient subset
        hospital_patients = [np.intersect1d(x, patient_set).astype(np.uint32) for x in hospital_patients_array]
        num_patients = num_matching
    else:
        patient_set = np.arange(10**8, dtype=np.uint32)
        hospital_patients = hospital_patients_array
        num_patients = 10**8

    '''Agg'''
    agg_stats = aggregate_count_stats(hospital_patients, num_patients)
    agg_mask_stats = aggregate_count_stats(hospital_patients, num_patients, mask=True)
    agg_mpc_stats = aggregate_count_stats(hospital_patients, num_patients, mpc=True)
    ans.extend([agg_stats, agg_mask_stats, agg_mpc_stats])

    '''HLL'''
    ans.extend(hyperloglog_generic_stats(hospital_patients, hospital_full_hlls0, num_patients, mpc=True))
    ans.extend(hyperloglog_generic_stats(hospital_patients, hospital_full_hlls1, num_patients, mpc=True))
    ans.extend(hyperloglog_generic_stats(hospital_patients, hospital_full_hlls4, num_patients, mpc=True))
    ans.extend(hyperloglog_generic_stats(hospital_patients, hospital_full_hlls7, num_patients, mpc=True))
    ans.extend(hyperloglog_generic_stats(hospital_patients, hospital_full_hlls15, num_patients, mpc=False))

    '''Hashed IDs'''
    summed_hospital_num = sum([len(x) for x in hospital_patients])
    try:
        if myglobals.summed_hospital_num != summed_hospital_num:
            raise ValueError('Not the right number of patients for precomputed list')
        hashing_times = myglobals.sha1_hashing_times
    except (AttributeError, ValueError):
        hashing_times = []
        for hosp in hospital_patients:
            start = time.perf_counter()
            for x in hosp:
                _ = sha1(str(x).encode()).digest()
            hashing_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    # full_union_num_patients = reduce(np.union1d, hospital_patients).size
    full_union_num_patients = len(np.unique(np.concatenate(hospital_patients)))
    hub_union_time = time.perf_counter() - start

    revealed_patients = sum([len(x) for x in hospital_patients])

    union_stats = {
        "title": "all_stats",
        "num_patients": num_patients,
        "estimate": full_union_num_patients,
        "revealed": revealed_patients,
        "revealed_5anon": revealed_patients,
        "revealed_10anon": revealed_patients,
        "hub_elapsed": hub_union_time
    }

    union_rehashed_stats = union_stats.copy()
    union_rehashed_stats['title'] = 'all_rehashed_stats'
    union_rehashed_stats['revealed'] = 0
    union_rehashed_stats['revealed_5anon'] = 0
    union_rehashed_stats['revealed_10anon'] = 0
    union_rehashed_stats['hosp_elapsed'] = sum(hashing_times)
    union_rehashed_stats['max_hosp_elapsed'] = max(hashing_times)
    ans.extend([union_stats, union_rehashed_stats])

    return ans


def sha1_with_time(i):
    start = time.perf_counter()
    _ = [sha1(str(x).encode()).digest() for x in myglobals.hospital_patients[i]]
    elapsed = time.perf_counter() - start
    return elapsed
