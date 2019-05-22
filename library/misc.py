'''Some useful utility functions'''

import os
import numpy as np


def read_patient_list(fname):
    '''Loads a newline-delimited list of integer numeric patient ids'''
    filename, file_extension = os.path.splitext(fname)
    if file_extension == '.npy' or file_extension == '.npz':
        A = np.load(fname)
    else:
        A = np.loadtxt(fname, dtype=np.uint64)
    return A


def query_results_fn(patient_list_file, hospital_population_file):
    query_patients = read_patient_list(patient_list_file)
    if hospital_population_file is None:
        query_results = query_patients
    else:
        hospital_population = read_patient_list(hospital_population_file)
        query_results = np.intersect1d(query_patients, hospital_population)
    return query_results


def count_output(query_results, mask=False):
    '''Result from using the raw count method'''
    count = len(query_results)
    print("True query output = {} patients".format(count))
    if mask:
        if count == 0:
            ans = 0
        elif count < 10:
            ans = 10
            print("Masked output = {} patients".format(ans))
        else:
            ans = count
    else:
        ans = count
    output_string = np.int64(ans).tobytes()
    return output_string
