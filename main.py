import numpy as np
import pandas as pd
from gurobipy import *
from matplotlib import pyplot as plt
import collections
import cvxpy as cp
from scheduling import *
from wasserstein import *
from estimate import *
from scipy.stats import skewnorm

def create_world():
    # create the pickup and delivery zones
    pickup_loc = [(x, y) for x in range(4) for y in range(4)]
    delivery_loc = [(x, y) for x in range(16, 20, 1) for y in range(16, 20, 1)]
    locs = pickup_loc + delivery_loc

    pickup_nodes_x = set()
    for i in pickup_loc:
        for t in range(0, int(T/2)):
            pickup_nodes_x.add((i, t))

    delivery_nodes_x = set()
    for i in delivery_loc:
        for t in range(int(T/2), T):
            delivery_nodes_x.add((i, t))

    return pickup_loc, delivery_loc, pickup_nodes_x, delivery_nodes_x

def create_ods(true_marginal, true_conds, scale, od_pairs=[]):
    true_ods, mod_ods= [], []
    req_ods_dict, k2q_dict = collections.defaultdict(list), collections.defaultdict(list)

    for k in range(K):
        if od_pairs != []:
            p_idx, d_idx = od_pairs[k]
        else:
            p_idx, d_idx = np.random.randint(len(pickup_loc)), np.random.randint(len(delivery_loc))
        d_t = np.random.randint(T-20, T)
        mod_p_t = d_t - (model.travel_time(pickup_loc[p_idx], delivery_loc[d_idx], 0) + 20)
        mod_ods.append((pickup_loc[p_idx], delivery_loc[d_idx], mod_p_t, d_t))

        true_level = np.random.choice(list(range(0, len(true_marginal))), p=true_marginal)
        true_p_t = mod_p_t + true_level * scale
        true_ods.append((pickup_loc[p_idx], delivery_loc[d_idx], true_p_t, d_t))

        for std, true_cond in true_conds.items():
            req_level = np.random.choice(list(range(0, len(true_marginal))), p=true_cond[true_level])
            k2q_dict[std].append(req_level)
            req_p_t = mod_p_t + req_level * scale
            req_ods_dict[std].append((pickup_loc[p_idx], delivery_loc[d_idx], req_p_t, d_t))

    return true_ods, req_ods_dict, mod_ods, k2q_dict

def get_distribution():
    # customer distribution generate
    true_marginal = [0.0] + [0.25, 0.25, 0.25, 0.25]
    true_conds_v = {}
    for param in params:
        np.random.seed(1234)
        true_cond = np.zeros((n_report, n_report))
        true_cond[0, 0] = 1
        for i in range(100000):
            region = np.random.choice(range(5), p=true_marginal)
            exact_time = np.random.uniform((region-1)*scale, region*scale)
            noise = skewnorm.rvs(skew, loc=0, scale=param)
            bias_time = exact_time + noise
            exact_level = int(np.ceil(exact_time / scale))
            bias_level = int(np.ceil(bias_time / scale))
            if 1 <= bias_level <= n_report-1:
                true_cond[exact_level, bias_level] += 1
        true_cond = true_cond / np.sum(true_cond, axis=1, keepdims=True)
        true_conds_v[param] = true_cond

    true_conds_r = {}
    for param in params:
        true_cond_v = true_conds_v[param]
        true_cond_r = []
        for i in range(n_report):
            dense = np.array([true_marginal[j] * true_cond_v[j][i] for j in range(1, n_report)])
            dense = dense / sum(dense)
            true_cond_r.append(dense)
        true_conds_r[param] = true_cond_r

    return true_marginal, true_conds_v, true_conds_r

def adoption_rate(param):
    probs_w, probs_w_single = [], []
    for i in range(n_report-1):
        marginal = true_marginal[1:]
        report = [0] * i + [1.0] + [0] * (len(true_marginal)-i-2)
        #conditional = true_conds_r[param][i+1]
        conditional = est_conds_r[param][i+1]

        d_cr = compute_wasserstein(report, conditional)
        d_cm = compute_wasserstein(marginal, conditional)
        d_mr = compute_wasserstein(marginal, report)
        _lambda = d_mr / (d_cr + d_cm)

        delta = {}
        delta['r'] = d_cr * (1 + _lambda) * 0.5
        delta['m'] = d_cm * (1 + _lambda) * 0.5
        delta['c'] = max(d_cr, d_cm) * 100

        if param < (1 - _lambda) * 0.5:
            return [], []

        probs_w.append(compute_worst_case(marginal, report, conditional, delta))

        delta['r'] = 100
        delta['m'] = 100
        probs_w_single.append(compute_worst_case(marginal, report, conditional, delta))

    probs_w = [probs_w[0]] + probs_w
    probs_w = np.round(probs_w, 2)

    probs_w_single = [probs_w_single[0]] + probs_w_single
    probs_w_single = np.round(probs_w_single, 2)

    return probs_w, probs_w_single

# model parameters
T = 80
n_batch = 25
K = 10
n_level = 21
scale = 5
n_serve, n_report = int((n_level-1) / scale) - 1, int((n_level-1) / scale) + 1

# std = 4
skew = 0
stds = [0, 2, 4, 6, 8]
params = stds

pickup_loc, delivery_loc, pickup_nodes_x, delivery_nodes_x = create_world()
model = scheduling(T, K, pickup_loc, delivery_loc, pickup_nodes_x, delivery_nodes_x)

np.random.seed(1234)
od_pairs = []
for k in range(K):
    p_idx, d_idx = np.random.randint(len(pickup_loc)), np.random.randint(len(delivery_loc))
    od_pairs.append((p_idx, d_idx))

true_marginal, true_conds_v, true_conds_r = get_distribution()

est_conds_r = {}
for param in params:
    np.random.seed(1234)
    est_cond_r = []
    for i in range(1, n_report):
        samples = [0, 0, 0, 0]
        for j in range(20):
            k = np.random.choice(a=4, p=true_conds_r[param][i])
            samples[k] += 1
        est_cond_r.append(np.array(samples) / np.sum(samples))
    est_cond_r = [est_cond_r[0]] + est_cond_r
    est_conds_r[param] = est_cond_r

output = []
for b in range(n_batch):
    result = {'report':[], 'marginal':[], 'conditional':[], 'robust':[], 'robust_single': []}
    print('------Batch %d-------' % b)

    true_ods, req_ods_dict, mod_ods, k2q_dict = create_ods(true_marginal, true_conds_v, scale, od_pairs)

    # ## M3
    # print('conditional')
    # for param in params:
    #     req_ods, k2q = req_ods_dict[param], k2q_dict[param]
    #     probs = est_conds_r[param]
    #     request_accept, true_accept, serve_levels, request_levels = model.run(true_ods, req_ods, mod_ods, k2q, probs, scale, 1800)
    #     result['conditional'].append((sum(request_accept), sum(true_accept)))

    ## M4
    for param in params:
        probs_w, probs_w_single = adoption_rate(param)
        if probs_w == []:
            result['robust'].append((0, 0))
            result['robust_single'].append((0, 0))
            print(param)
            continue

        #req_ods, k2q = req_ods_dict[param], k2q_dict[param]
        req_ods, k2q = req_ods_dict[params[0]], k2q_dict[params[0]]
        print('robust')
        request_accept, true_accept, serve_levels, request_levels = model.run(true_ods, req_ods, mod_ods, k2q, probs_w, scale, 1800)
        result['robust'].append((sum(request_accept), sum(true_accept)))

        # print('standard wasserstein')
        # request_accept, true_accept, serve_levels, request_levels = model.run(true_ods, req_ods, mod_ods, k2q, probs_w_single, scale, 1800)
        # result['robust_single'].append((sum(request_accept), sum(true_accept)))

    ## M1
    probs_r = np.concatenate(([[1.0] + [0.0] * n_serve], np.eye(n_serve+1)))
    for param in params:
        req_ods, k2q = req_ods_dict[param], k2q_dict[param]
        print('report')
        request_accept, true_accept, serve_levels, request_levels = model.run(true_ods, req_ods, mod_ods, k2q, probs_r, scale)
        result['report'].append((sum(request_accept), sum(true_accept)))


    ## M2
    probs_m = np.tile(true_marginal[1:], (n_report,1))
    req_ods, k2q = req_ods_dict[param], k2q_dict[param]
    print('marginal')
    request_accept, true_accept, serve_levels, request_levels = model.run(true_ods, req_ods, mod_ods, k2q, probs_m, scale, 1800)
    for param in params:
        result['marginal'].append((sum(request_accept), sum(true_accept)))

    output.append(result)

# summarize output
res_conditional, res_report, res_marginal, res_robust, res_robust_single = [], [], [], [], []
can_conditional, can_report, can_marginal, can_robust, can_robust_single = [], [], [], [], []

for i in range(len(params)):
    res_report.append(np.mean([res['report'][i][1] for res in output]))
    res_marginal.append(np.mean([res['marginal'][i][1] for res in output]))
    # res_conditional.append(np.mean([res['conditional'][i][1] for res in output]))
    res_robust.append(np.mean([res['robust'][i][1] for res in output if res['robust'][i][1] != 0]))
    # res_robust_single.append(np.mean([res['robust_single'][i][1] for res in output if res['robust_single'][i][1] != 0]))

    can_report.append(np.mean([res['report'][i][0] for res in output]))
    can_marginal.append(np.mean([res['marginal'][i][0] for res in output]))
    # can_conditional.append(np.mean([res['conditional'][i][0] for res in output]))
    can_robust.append(np.mean([res['robust'][i][0] for res in output if res['robust'][i][0] != 0]))
    # can_robust_single.append(np.mean([res['robust_single'][i][0] for res in output if res['robust_single'][i][0] != 0]))

#print(res_conditional)
print(res_robust)
# print(res_robust_single)
print(res_marginal)
print(res_report)

print(can_robust)
# print(rat_robust_single)
print(can_marginal)
print(can_report)

plt.plot(params, res_marginal, '^--', ms=5.0, lw=1.0, label='marginal')
plt.plot(params, res_report, 's--', ms=5.0, lw=1.0, label='report')
plt.plot(params, res_robust, 'o--', ms=5.0, lw=1.0, label='robust')
# plt.plot(params, res_robust_single, 'o--', ms=5.0, lw=1.0, label='standard_wasserstein')
# plt.plot(params, res_conditional, 'o--', ms=5.0, lw=1.0, label='conditional')
plt.legend()
plt.xlabel('standard deviation')
plt.ylabel('average adoptions')
plt.savefig('two_ball_uniform_skew0.png')

df = pd.
path = 'result.xlsx'
writer = pd.ExcelWriter(path, engine = 'xlsxwriter')
df.to_excel(writer, sheet_name = 'two_ball_uniform_skew0')
writer.save()
writer.close()
