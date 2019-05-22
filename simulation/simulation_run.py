#!/usr/bin/env python3
'''Generates a hospital network with patient lists.

Also preprocesses it with HLL of full patient lists to allow for privacy
analyses when using masked HLL.
'''

import sys
import os
import argparse
import numpy as np
import array
from itertools import repeat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import hyperloglog  # noqa: E402

__version__ = '0.0.1'


def main(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument('-n', '--num-hospitals', help='number of hospitals in network',
                        type=int, default=100)
    parser.add_argument('-m', '--num-patients', help='number of unique patients in network',
                        type=int, default=10**8)
    parser.add_argument('-d', '--duplicates', help='average number of hospitals each'
                        'patient is at', type=int, default=2)
    parser.add_argument('--max-duplicates', help='max number of hospitals each '
                        'patient is at, with a minimum value of either 10 or '
                        'twice the number of duplicates', type=int, default=0)
    parser.add_argument('-r', '--random-seed', help='random seed for generation',
                        type=int, default=314)
    parser.add_argument('-s', '--salt', help='salt hashing or for HLL',
                        default="")
    parser.add_argument('-k', '--buckets', help='buckets for the HLL',
                        type=int, default=128)
    parser.add_argument('out_dir', help='directory for output files',
                        default=None)
    args = parser.parse_args(argv)
    print(args)
    sys.stdout.flush()
    hospital_patients_array = simulate_population_and_hospitals_power_law(
        random_seed=args.random_seed,
        num_actual_population=args.num_patients,
        num_hospitals=args.num_hospitals,
        duplicates=args.duplicates,
        max_duplicates=args.max_duplicates)
    print([len(x) for x in hospital_patients_array])
    os.mkdir(args.out_dir)
    for i, h in enumerate(hospital_patients_array):
        np.save(os.path.join(args.out_dir, 'hosp{:03d}.npy'.format(i)), h)
        h_freq = hyperloglog.HyperLogLog_freqs(h, args.buckets, salt=args.salt)
        with open(os.path.join(args.out_dir, 'hosp{:03d}.freqs'.format(i)), 'wb') as f:
            f.write(h_freq.safe_export())
        with open(os.path.join(args.out_dir, 'config.txt'), 'w') as f:
            print(args, file=f)


def simulate_population_and_hospitals_power_law(random_seed=10000, num_actual_population=10**8, num_hospitals=100, duplicates=2, max_duplicates=10):
    '''This model first generates a power law distribution of hospital sizes,
    places them in a 2D unit square, and randomly assigns second hospitals by
    probability inversely proportional to the square distance to the first
    hospital and proportional to the size of the new hospital.

    i.e. patients are more likely to go to nearby hospitals by square distance,
    but also more likely to go to big hospitals by population
    '''
    '''Parameters'''
    assert(num_actual_population < 2**63)  # since we use np.int64
    if max_duplicates < 2 * duplicates:
        max_duplicates = 2 * duplicates
    max_duplicates = max(10, max_duplicates)

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
    Then we assign the patient to a home hospital.
    '''
    patient_list = prng.permutation(num_actual_population)
    hospital_patients_array = [array.array('L') for _ in range(num_hospitals)]
    curr_patient = 0
    for i, npa in enumerate(hospital_sizes):
        hospital_patients_array[i].extend(patient_list[curr_patient:curr_patient + npa])
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
    sys.stdout.write('Assigning extra hospital locations\n')
    hospital_patients_array_extension = [array.array('L') for _ in range(num_hospitals)]
    tot_proc = 0
    for h in range(num_hospitals):
        hospitals_per_patient = prng.binomial(max_duplicates - 1, (duplicates - 1) / (max_duplicates - 1), size=hospital_sizes[h])

        squared_distances = [(p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 for p1, p2 in zip(hospital_locations, repeat(hospital_locations[h]))]
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
                tot_proc = tot_proc + 1
            if tot_proc % 10**6 == 0:
                sys.stdout.write('\r' + str(tot_proc))
                sys.stdout.flush()
    sys.stdout.write('\n')
    for h in range(num_hospitals):
        hospital_patients_array[h].extend(hospital_patients_array_extension[h])

    return hospital_patients_array


if __name__ == "__main__":
    main(sys.argv[1:])
