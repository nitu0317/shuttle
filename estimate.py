import cvxpy as cp
import numpy as np



def learn_valuation(n_a, n_r, n_var):
    x = cp.Variable(n_var)
    G = np.zeros((2*n_serve*n_report, n_var))
    h = np.zeros((2*n_serve*n_report,))

    k = 0
    # monotone for each report value
    for i in range(n_report):
        breakpoint = sum(end[q]-begin[q] for q in range(i))
        for j in range(end[i]-begin[i]-1):
            bp1, bp2 = j+breakpoint, j+1+breakpoint
            G[k, [bp1, bp2]] = [1.0, -1.0]
            k += 1

    # monotone for each serve value
    for i in range(n_serve):
        for j in range(n_report-1):
            if begin[j] <= i+1 and begin[j+1] <= i+1 and end[j] > i+1 and end[j+1] > i+1:
                bp1, bp2 = sum(end[q]-begin[q] for q in range(j)) + i-begin[j]+1, sum(end[q]-begin[q] for q in range(j+1)) + i-begin[j+1]+1
                G[k, [bp1, bp2]] = [-1.0, 1.0]
                k += 1

    # Define and solve the CVXPY problem.
    model = cp.Problem(cp.Maximize(n_a.T@cp.log(x) + n_r.T@cp.log(1-x)),[G@x <= h])
    model.solve()
    res = x.value
    return res

# # tactical operations
# n_batch = 50
# n_customer = 10
# n_level = 21
# scale = 5
# n_serve, n_report = int((n_level-1) / scale) - 1, int((n_level-1) / scale) + 1
#
# # true distribution
# true_dist = []
# true_dist.append([1.0, 0.0, 0.0, 0.0])
# true_dist.append([0.5, 0.5, 0.0, 0.0])
# true_dist.append([0.25, 0.5, 0.25, 0.0])
# true_dist.append([0.0, 0.25, 0.5, 0.25])
# true_dist.append([0.0, 0.0, 0.5, 0.5])
# req_dist = [0.25, 0.25, 0.25, 0.25]
#
# # initialize without report
# n_accept = {j:{i:0 for i in range(n_serve+2)} for j in range(n_report)}
# n_reject = {j:{i:0 for i in range(n_serve+2)} for j in range(n_report)}
# for i in range(n_report):
#     n_reject[i][0] = 1
#     n_accept[i][n_serve+1] = 1
#
# prior = [[0.25] * 4] * 5
# probs = prior
# result = []
#
# for b in range(n_batch):
#     print('------Batch %d-------' % b)
#
#     ## get routing result
#     K = n_customer
#     request_accept, true_accept, serve_levels, request_levels = routing(true_dist, req_dist, probs, scale)
#
#     for c in range(n_customer):
#         if request_accept[c] == 0:
#             continue
#         serve = serve_levels[c]
#         request = request_levels[c]
#         res = true_accept[c]
#
#         if res == 1:
#             n_accept[request][serve] += 1
#         else:
#             n_reject[request][serve] += 1
#
#     ## get some empirical stats
#     begin = {}
#     for j in range(n_report):
#         for i in range(n_serve+2):
#             if n_accept[j][i] > 0:
#                 begin[j] = i
#                 break
#
#     end = {}
#     for j in range(n_report):
#         for i in range(n_serve, -1, -1):
#             if n_reject[j][i] > 0:
#                 end[j] = i + 1
#                 break
#
#     argm = 0
#     while argm < n_report-1:
#         argm_new = np.argmin([begin[j] for j in range(argm, n_report)]) + argm
#         for i in range(argm, argm_new):
#             begin[i] = begin[argm_new]
#         argm = argm_new + 1
#
#     argm = n_report
#     while argm > 0:
#         argm_new = np.argmax([end[j] for j in range(argm)])
#         for i in range(argm_new, argm):
#             end[i] = end[argm_new]
#         argm = argm_new
#
#     for i in range(n_report):
#         if begin[i] > end[i]:
#             end[i] = begin[i]
#
#     ## get some empirical stats
#     n_var = sum(end[j]-begin[j] for j in range(n_report))
#     n_a = matrix([n_accept[j][i] for j in range(n_report) for i in range(begin[j], end[j])], (n_var, 1))
#     n_r = matrix([n_reject[j][i] for j in range(n_report) for i in range(begin[j], end[j])], (n_var, 1))
#
#     ## update valuation distribution
#     if n_var > 0:
#         new_cdf = learn_valuation(n_a, n_r, n_var)
#     else:
#         new_cdf = []
#
#     post = []
#     for j in range(n_report):
#         if begin[j] == end[j]:
#             probs = [0.0] * (n_serve+1)
#             probs[begin[j]-1] = 1.0
#         else:
#             breakpoint = sum(end[q]-begin[q] for q in range(j))
#             probs = [0.0] * (begin[j]-1) + [round(new_cdf[breakpoint],2)]
#             for i in range(end[j]-begin[j]-1):
#                 probs.append(round(new_cdf[i+1+breakpoint] - new_cdf[i+breakpoint], 2))
#             probs += [round(1-new_cdf[end[j]-begin[j]-1+breakpoint],2)] + [0.0] * (n_serve+1-end[j])
#         post.append(probs)
#
#     print(post)
#     #probs = np.round(np.array(prior) * (1-(b+1)/n_batch) + np.array(post) * ((b+1)/n_batch), 2)
#     probs = prior
#     print(probs)
#
#     result.append({'probs':probs, 'req_accept':sum(request_accept), 'true_accept':sum(true_accept)})
