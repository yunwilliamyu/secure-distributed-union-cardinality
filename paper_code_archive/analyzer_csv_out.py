#!/usr/bin/env python3
import sys
import numbers
import numpy as np


def read_experiments(file_list):
    experiments = []
    for fn in file_list:
        with open(fn) as f:
            data = f.read().rstrip('\n')
            exec('experiments.append(' + data + ')')
    return experiments


def aggregate(exps, cond_dict):
    '''Returns a list of the experiments that match the conditions given'''
    return [e for e in exps if len(e.items() & cond_dict.items()) == len(cond_dict.items())]


def reduce(exps):
    '''Takes the average of all the values in in a list of experiments

        Returns results as a dict containing:
            (1) the arithmetic mean of any numeric values
            (2) a string if and only if that string has the same value for all experiments
        Note that any key that is not present in all experiments is discarded. Similarly, if a string has multiple values, it too is discarded.
    '''
    e0 = exps[0]
    ans = {}
    for k in e0.keys():
        if isinstance(e0[k], numbers.Number):
            try:
                ans[k] = sum([e[k] for e in exps]) / len(exps)
                ans[k + '_std'] = np.std([e[k]for e in exps])
                ans[k + '_low95'] = np.percentile([e[k]for e in exps], 2.5)
                ans[k + '_high95'] = np.percentile([e[k]for e in exps], 97.5)
            except (TypeError, KeyError):
                pass
        if isinstance(e0[k], str):
            try:
                if len(set([e[k] for e in exps])) == 1:
                    ans[k] = e0[k]
            except (TypeError, KeyError):
                pass
    return ans


def percent_err(x_est, x):
    '''Returns relative percent error'''
    return 100 * (x_est - x) / x


def fnum(x):
    '''Formats a number with commas and no decimal places if >1,000, and with 2 digits after the decimal point otherwise'''
    if x > 1000:
        return "{:,.0f}".format(x)
    elif x > 100:
        return "{:.1f}".format(x)
    elif x > 10:
        return "{:.2f}".format(x)
    else:
        return "{:.3f}".format(x)


def count_string(c):
    '''Used for all the raw aggregate count data, and also hll-mask, since that incorporates an aggregate count like portion'''
    num_patients = c['num_patients']
    try:
        wait = (c['hospital_round0_time'] / 100
                + c['hub_round0_time']
                + c['hospital_round1_time'] / 100
                + c['hub_round1.5_time']
                + c['hospital_round2_time'] / 100
                + c['hub_round2.5_time'])
        wait_string = fnum(wait)
    except KeyError:
        wait = c['hub_elapsed']
        wait_string = fnum(wait)

    if 'estimate_min' not in c:
        c['estimate_min'] = c['estimate']
        c['estimate_min_std'] = c['estimate_std']
        c['estimate_min_low95'] = c['estimate_low95']

    c_string = ",".join([
        c['title'],  # Method
        "{},{}".format(fnum(c['estimate_min_low95']), fnum(c['estimate_high95'])),  # Estimate (actual)
        "{:.0f}%,{:.0f}%".format(percent_err(c['estimate_min_low95'], num_patients), percent_err(c['estimate_high95'], num_patients)),  # Estimate (relative)
        "{}".format(wait_string),  # Wait (mean)
        "{}".format(wait_string),  # Wait (max)
        "{}".format(fnum(c['revealed'])),  # Hub r1
        "{}".format(fnum(c['revealed_5anon'])),  # Hub r4
        "{}".format(fnum(c['revealed_10anon'])),  # Hub r9
        "{}".format(fnum(c['revealed'])),  # Hosp r1
        "{}".format(fnum(c['revealed_5anon'])),  # Hosp r4
        "{}".format(fnum(c['revealed_10anon'])),  # Hosp r9
    ])
    return c_string


def hll_string(h, h_hosp=None):
    '''h contains data for the method we are currently using, and h_hosp contains data for the equivalent method
    if a hospital conspires with the hub to reveal the secret salt for rehashing or shuffling

    Note: do not use for the hll-mask method, but use count_string instead because it also includes a summation term.
    '''
    num_patients = h['num_patients']

    if h_hosp is None:
        h_hosp = h

    assert 'estimate_min' not in h, "Please don't use hll_string for hll-mask method or count methods. Use count_string instead"

    if 'hub_elapsed' in h:
        if 'hosp_elapsed' not in h:
            h['hosp_elapsed'] = 0
        if 'max_hosp_elapsed' not in h:
            h['max_hosp_elapsed'] = 0

    try:
        # Using MPC
        wait_mean = (
            h['hospital_round0_time'] / 100
            + h['hub_round0_time']
            + h['hospital_round1_time'] / 100
            + h['hub_round1.5_time']
            + h['hospital_round2_time'] / 100
            + h['hub_round_2.5_time']
            + h['hub_round2.9_time']
        )
        wait_mean_string = fnum(wait_mean)
        wait_max_string = wait_mean_string
    except KeyError:
        try:
            # Rehashing
            wait_mean = h['hosp_elapsed'] / 100 + h['hub_elapsed']
            wait_max = h['max_hosp_elapsed'] + h['hub_elapsed']
            wait_mean_string = fnum(wait_mean)
            wait_max_string = fnum(wait_max)
        except KeyError:
            wait_mean_string = fnum(h['hub_elapsed'])
            wait_max_string = wait_mean_string

    h_string = ",".join([
        h['title'],  # Method
        "{},{}".format(fnum(h['estimate_low95']), fnum(h['estimate_high95'])),  # Estimate (actual)
        "{:.0f}%,{:.0f}%".format(percent_err(h['estimate_low95'], num_patients), percent_err(h['estimate_high95'], num_patients)),  # Estimate (relative)
        wait_mean_string,  # Wait (mean)
        wait_max_string,  # Wait (max)
        "{}".format(fnum(h['revealed'])),  # Hub r1
        "{}".format(fnum(h['revealed_5anon'])),  # Hub r4
        "{}".format(fnum(h['revealed_10anon'])),  # Hub r9
        "{}".format(fnum(h_hosp['revealed'])),  # Hosp r1
        "{}".format(fnum(h_hosp['revealed_5anon'])),  # Hosp r4
        "{}".format(fnum(h_hosp['revealed_10anon'])),  # Hosp r9
    ])
    return h_string


def summary_count(num_patients, exps):
    '''Returns a TSV string for information on counts with the following rows

    Method, Estimate, Hub-r1, Hub-r4, Hub-r9, HubHosp-r1, HubHosp-r4, HubHosp-r9, Wait

    Where 'Method' is one of the following:
        count
        count, mask
        hll
        hll, mask
        hll, rehash
        hll, shuffle
        count, mpc
        hll, mpc
        hll, mpc, shuffle
    '''
    ans_array = []

    c = reduce(aggregate(exps, {'title': 'agg_stats', 'num_patients': num_patients}))
    cm = reduce(aggregate(exps, {'title': 'agg_mask_stats', 'num_patients': num_patients}))
    ce = reduce(aggregate(exps, {'title': 'agg_mpc_stats', 'num_patients': num_patients}))
    ans_array.append(count_string(c))
    ans_array.append(count_string(cm))
    ans_array.append(count_string(ce))

    h = reduce(aggregate(exps, {'title': 'hll0_stats', 'num_patients': num_patients}))
    hm = reduce(aggregate(exps, {'title': 'hll0_mask_stats', 'num_patients': num_patients}))
    hr = reduce(aggregate(exps, {'title': 'hll0_rehashed_stats', 'num_patients': num_patients}))
    hs = reduce(aggregate(exps, {'title': 'hll0_shuffle_stats', 'num_patients': num_patients}))
    he = reduce(aggregate(exps, {'title': 'hll0_mpc_stats', 'num_patients': num_patients}))
    hes = reduce(aggregate(exps, {'title': 'hll0_mpc_shuffle_stats', 'num_patients': num_patients}))
    ans_array.append(hll_string(h))
    ans_array.append(hll_string(hs, h_hosp=h))
    ans_array.append(hll_string(hr, h_hosp=h))
    ans_array.append(count_string(hm))
    ans_array.append(hll_string(he))
    ans_array.append(hll_string(hes, h_hosp=he))

    h = reduce(aggregate(exps, {'title': 'hll1_stats', 'num_patients': num_patients}))
    hm = reduce(aggregate(exps, {'title': 'hll1_mask_stats', 'num_patients': num_patients}))
    hr = reduce(aggregate(exps, {'title': 'hll1_rehashed_stats', 'num_patients': num_patients}))
    hs = reduce(aggregate(exps, {'title': 'hll1_shuffle_stats', 'num_patients': num_patients}))
    he = reduce(aggregate(exps, {'title': 'hll1_mpc_stats', 'num_patients': num_patients}))
    hes = reduce(aggregate(exps, {'title': 'hll1_mpc_shuffle_stats', 'num_patients': num_patients}))
    ans_array.append(hll_string(h))
    ans_array.append(hll_string(hs, h_hosp=h))
    ans_array.append(hll_string(hr, h_hosp=h))
    ans_array.append(count_string(hm))
    ans_array.append(hll_string(he))
    ans_array.append(hll_string(hes, h_hosp=he))

    h = reduce(aggregate(exps, {'title': 'hll4_stats', 'num_patients': num_patients}))
    hm = reduce(aggregate(exps, {'title': 'hll4_mask_stats', 'num_patients': num_patients}))
    hr = reduce(aggregate(exps, {'title': 'hll4_rehashed_stats', 'num_patients': num_patients}))
    hs = reduce(aggregate(exps, {'title': 'hll4_shuffle_stats', 'num_patients': num_patients}))
    he = reduce(aggregate(exps, {'title': 'hll4_mpc_stats', 'num_patients': num_patients}))
    hes = reduce(aggregate(exps, {'title': 'hll4_mpc_shuffle_stats', 'num_patients': num_patients}))
    ans_array.append(hll_string(h))
    ans_array.append(hll_string(hs, h_hosp=h))
    ans_array.append(hll_string(hr, h_hosp=h))
    ans_array.append(count_string(hm))
    ans_array.append(hll_string(he))
    ans_array.append(hll_string(hes, h_hosp=he))

    h = reduce(aggregate(exps, {'title': 'hll7_stats', 'num_patients': num_patients}))
    hm = reduce(aggregate(exps, {'title': 'hll7_mask_stats', 'num_patients': num_patients}))
    hr = reduce(aggregate(exps, {'title': 'hll7_rehashed_stats', 'num_patients': num_patients}))
    hs = reduce(aggregate(exps, {'title': 'hll7_shuffle_stats', 'num_patients': num_patients}))
    he = reduce(aggregate(exps, {'title': 'hll7_mpc_stats', 'num_patients': num_patients}))
    hes = reduce(aggregate(exps, {'title': 'hll7_mpc_shuffle_stats', 'num_patients': num_patients}))
    ans_array.append(hll_string(h))
    ans_array.append(hll_string(hs, h_hosp=h))
    ans_array.append(hll_string(hr, h_hosp=h))
    ans_array.append(count_string(hm))
    ans_array.append(hll_string(he))
    ans_array.append(hll_string(hes, h_hosp=he))

    h = reduce(aggregate(exps, {'title': 'hll15_stats', 'num_patients': num_patients}))
    hm = reduce(aggregate(exps, {'title': 'hll15_mask_stats', 'num_patients': num_patients}))
    hr = reduce(aggregate(exps, {'title': 'hll15_rehashed_stats', 'num_patients': num_patients}))
    hs = reduce(aggregate(exps, {'title': 'hll15_shuffle_stats', 'num_patients': num_patients}))
    # he = reduce(aggregate(exps, {'title': 'hll15_mpc_stats', 'num_patients': num_patients}))
    # hes = reduce(aggregate(exps, {'title': 'hll15_mpc_shuffle_stats', 'num_patients': num_patients}))
    ans_array.append(hll_string(h))
    ans_array.append(hll_string(hs, h_hosp=h))
    ans_array.append(hll_string(hr, h_hosp=h))
    ans_array.append(count_string(hm))
    # ans_array.append(hll_string(he))
    # ans_array.append(hll_string(hes, h_hosp=he))

    au = reduce(aggregate(exps, {'title': 'all_stats', 'num_patients': num_patients}))
    aur = reduce(aggregate(exps, {'title': 'all_rehashed_stats', 'num_patients': num_patients}))
    ans_array.append(hll_string(au))
    ans_array.append(hll_string(aur, h_hosp=au))

    return "\n".join(ans_array)


if __name__ == '__main__':
    experiments = read_experiments(sys.argv[1:])
    experiment_list = [z for x in experiments for y in x for z in y]

    print(",".join([
        "Method",
        "Estimate (actual)-95% CI lower bound",
        "Estimate (actual)-95% CI upper bound",
        "Estimate (relative)-95% CI lower bound",
        "Estimate (relative)-95% CI upper bound",
        "Wait (mean)",
        "Wait (max)",
        "Hub r1",
        "Hub r4",
        "Hub r9",
        "Hosp r1",
        "Hosp r4",
        "Hosp r9"
    ]))

    for x in [10**i for i in range(0, 9)]:
        print(summary_count(x, experiment_list))
