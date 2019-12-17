import numpy as np
import collections
from gurobipy import *

c = {}
for i in range(20):
    for j in range(20):
        c[(i, j)] = abs(i-j)

def compute_worst_case(marginal, report, conditional, delta):
    def ambiguity(u):
        model = Model()
        model.setParam("LogToConsole", 0)
        q = model.addVars([i for i in range(len(marginal))], vtype=GRB.CONTINUOUS, lb=0, ub=1, name='q')
        flow_qm = model.addVars([(i, j) for i in range(len(marginal)) for j in range(len(marginal))], vtype=GRB.CONTINUOUS, lb=0, ub=1, name='flow_qm')
        flow_qr = model.addVars([(i, j) for i in range(len(marginal)) for j in range(len(marginal))], vtype=GRB.CONTINUOUS, lb=0, ub=1, name='flow_qr')
        flow_qc = model.addVars([(i, j) for i in range(len(marginal)) for j in range(len(marginal))], vtype=GRB.CONTINUOUS, lb=0, ub=1, name='flow_qc')

        model.setObjective(sum(q[i] for i in range(u)), GRB.MINIMIZE)

        model.addConstr(sum(flow_qc[i, j] * c[i, j] for i in range(len(marginal)) for j in range(len(marginal))) <= delta['c'])

        for i in range(len(marginal)):
            model.addConstr(flow_qc.sum(i, '*') == q[i])
            model.addConstr(flow_qc.sum('*', i) == conditional[i])

        model.addConstr(sum(flow_qm[i, j] * c[i, j] for i in range(len(marginal)) for j in range(len(marginal))) <= delta['m'])

        for i in range(len(marginal)):
            model.addConstr(flow_qm.sum(i, '*') == q[i])
            model.addConstr(flow_qm.sum('*', i) == marginal[i])

        model.addConstr(sum(flow_qr[i, j] * c[i, j] for i in range(len(marginal)) for j in range(len(marginal))) <= delta['r'])

        for i in range(len(marginal)):
            model.addConstr(flow_qr.sum(i, '*') == q[i])
            model.addConstr(flow_qr.sum('*', i) == report[i])

        model.optimize()

        return model.ObjVal

    p = []
    for u in range(1, len(marginal)+1):
        p.append(ambiguity(u) - sum(p))

    return p

def compute_wasserstein(p1, p2):
    model = Model()
    model.setParam("LogToConsole", 0)
    flow = model.addVars([(i, j) for i in range(len(p1)) for j in range(len(p1))], vtype=GRB.CONTINUOUS, lb=0, ub=1, name='flow')

    model.setObjective(sum(flow[i, j] * c[i, j] for i in range(len(p1)) for j in range(len(p1))) , GRB.MINIMIZE)

    for i in range(len(p1)):
        model.addConstr(flow.sum(i, '*') == p1[i])
        model.addConstr(flow.sum('*', i) == p2[i])

    model.optimize()

    return model.ObjVal
