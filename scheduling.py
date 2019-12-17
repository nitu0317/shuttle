import numpy as np
import collections
from gurobipy import *

class scheduling(object):
    def __init__(self, T, K, pickup_loc, delivery_loc, pickup_nodes_x, delivery_nodes_x):
        #super(scheduling, self).__init__()
        self.T = T
        self.K = K
        self.pickup_loc = pickup_loc
        self.delivery_loc = delivery_loc
        self.pickup_nodes_x = pickup_nodes_x
        self.delivery_nodes_x = delivery_nodes_x

    def travel_time(self, i, j, walk=1):
        if type(i) == str and type(j) == str:
            return self.T
        if type(i) == str or type(j) == str:
            return 0

        dist = abs(i[0] - j[0]) + abs(i[1] - j[1])
        if walk:
            return dist * 5
        else:
            return dist

    def create_vehicle_network(self):

        nodes_x = set()
        nodes_x.add(('source', 0))
        nodes_x.add(('sink', self.T))
        nodes_x = nodes_x.union(self.pickup_nodes_x)
        nodes_x = nodes_x.union(self.delivery_nodes_x)

        arcs_x = set()
        for i, t in self.pickup_nodes_x:
            arcs_x.add(('source', 0, i, t))
            for j, s in self.pickup_nodes_x:
                if i != j and self.travel_time(i, j, 0) == s - t and s - t == 1:
                    arcs_x.add((i, t, j, s))

            for j, s in self.delivery_nodes_x:
                if self.travel_time(i, j, 0) == s - t and i == self.pickup_loc[-1] and j == self.delivery_loc[0]:
                    arcs_x.add((i, t, j, s))

        for i, t in self.delivery_nodes_x:
            arcs_x.add((i, t, 'sink', self.T))
            for j, s in self.delivery_nodes_x:
                if i != j and self.travel_time(i, j,  0) == s - t and s - t == 1:
                    arcs_x.add((i, t, j, s))

        return nodes_x, arcs_x

    def create_customer_network(self, mod_ods):

        arcs_y, nodes_y = [], []
        for k in range(self.K):
            customer_arcs, customer_nodes = set(), set()
            p, d, pt, dt = mod_ods[k]
            for t in range(21):
                customer_nodes.add((p, pt + t))
            for t in range(21):
                customer_nodes.add((d, dt - t))
            customer_nodes.add(('source' + str(k), pt))
            customer_nodes.add(('sink' + str(k), dt))

            for i, t in customer_nodes:
                if i == p:
                    for j, s in self.pickup_nodes_x:
                        if i != j and self.travel_time(i, j, 1) == s - t and (s - t) <= 0:
                            customer_arcs.add((i, t, j, s))
                elif i == d:
                    for j, s in self.delivery_nodes_x:
                        if i != j and self.travel_time(j, i, 1) == t - s and (t - s) <= 0:
                            customer_arcs.add((j, s, i, t))

                elif i == 'source' + str(k):
                    for j, s in customer_nodes:
                        if j == p:
                            customer_arcs.add((i, t, j, s))
                elif i == 'sink' + str(k):
                    for j, s in customer_nodes:
                        if j == d:
                            customer_arcs.add((j, s, i, t))

            customer_arcs.add((p, pt, d, dt))

            arcs_y.append(customer_arcs)
            nodes_y.append(customer_nodes)


        return nodes_y, arcs_y

    def create_distance(self, arcs_x, arcs_y):

        D = {}
        for i, t, j, s in arcs_x:
            D[i, t, j, s] = self.travel_time(i, j, 0)

        for k in range(self.K):
            for i, t, j, s in arcs_y[k]:
                D[i, t, j, s] = self.travel_time(i, j, 1)

        return D

    def create_flow_consumption(self, nodes_x, nodes_y):

        b_x = {}
        for i, t in nodes_x:
            if i == 'source':
                b_x[i, t] = 1.0
            elif i == 'sink':
                b_x[i, t] = -1.0
            else:
                b_x[i, t] = 0

        b_y = []
        for k in range(self.K):
            balance = {}
            for i, t in nodes_x.union(nodes_y[k]):
                if i == 'source' + str(k):
                    balance[i, t] = 1.0
                elif i == 'sink' + str(k):
                    balance[i, t] = -1.0
                else:
                    balance[i, t] = 0

            b_y.append(balance)

        return b_x, b_y

    def optimize_once(self, arcs_x, arcs_y, nodes_x, nodes_y, b_x, b_y, mod_ods, probs, D, scale, k2q, TL=900):
        '''use gurobi to run the optimization model once'''
        # define the model
        model = Model('model')
        model.setParam("LogToConsole", 0)
        model.Params.Method = 3

        model.Params.Timelimit = TL

        # define the variables
        ## routing variables
        vehicle = model.addVars([(i, t, j, s) for i, t, j, s in arcs_x], vtype=GRB.BINARY, lb=0, ub=1, name='vehicle')
        customers = model.addVars([(k, i, t, j, s) for k in range(self.K) for i, t, j, s in arcs_y[k].union(arcs_x)], vtype=GRB.BINARY, lb=0, ub=1, name='customers')

        ## decision-making variables
        values = model.addVars([k for k in range(self.K+1)], vtype=GRB.CONTINUOUS, lb=0, name='values')
        choices = model.addVars([(k, i) for k in range(self.K) for i in range(len(probs[k2q[k]]))], vtype=GRB.BINARY, lb=0, ub=1, name='choices')

        # define the objective function

        model.setObjective(sum(sum(choices[k, i] * probs[k2q[k]][i] for i in range(len(probs[k2q[k]]))) for k in range(self.K)) - 0.001 * values[self.K], GRB.MAXIMIZE)

        # define the constraints
        ## vehicle flow balance
        for i, t in b_x:
            model.addConstr(sum(vehicle[i, t, j, s] for j, s in nodes_x if (i, t, j, s) in vehicle) - sum(vehicle[j, s, i, t] for j, s in nodes_x if (j, s, i, t) in vehicle) == b_x[i, t])

        ## customers flow balance
        for k in range(self.K):
            for i, t in b_y[k]:
                model.addConstr(sum(customers[k, i, t, j, s] for j, s in nodes_x.union(nodes_y[k]) if (k, i, t, j, s) in customers) - sum(customers[k, j, s, i, t] for j, s in nodes_x.union(nodes_y[k]) if (k, j, s, i, t) in customers) == b_y[k][i, t])

        ## arc capacity
        for k in range(self.K):
            for (i, t, j, s) in arcs_x:
                model.addConstr(customers[k, i, t, j, s] <= vehicle[i, t, j, s])

        ## customer value of surplus
        for k in range(self.K):
            model.addConstr(values[k] == sum((s-mod_ods[k][2]) * customers[k, i, t, j, s] for (i, t, j, s) in arcs_y[k].union(arcs_x) if i == 'source' + str(k)))

        ## customer choices
        for k in range(self.K):
            model.addConstr(values[k] >= sum(choices[k, i] * scale for i in range(len(probs[k2q[k]]))))

            for i in range(len(probs[k2q[k]])-1):
                model.addConstr(choices[k, i] >= choices[k, i+1])

        ## vehicle route
        model.addConstr(values[self.K] == sum(vehicle[i, t, j, s] * D[i, t, j, s] for (i, t, j, s) in arcs_x))

        # start optimization
        model.optimize()
        return model, vehicle, customers, values, choices

    def get_routes(self, model, vehicle):
        _vehicle = model.getAttr("X", vehicle)
        routes = []
        for key, value in _vehicle.items():
            if value > 0.1:
                if type(key[0]) is not str and type(key[2]) is not str:
                    routes.append((key[0], key[2]))
        return routes

    def get_pd_times(self, model, customers):
        _customers = model.getAttr("X", customers)
        pickup_times, delivery_times = {}, {}
        for k, v in _customers.items():
            if v > 0.1:
                if type(k[1]) is str and type(k[3]) is str:
                    pickup_times[k[0]] = -1
                    delivery_times[k[0]] = T
                elif type(k[1]) is str and type(k[3]) is not str:
                    pickup_times[k[0]] = k[4]
                elif type(k[1]) is not str and type(k[3]) is str:
                    delivery_times[k[0]] = k[2]
        return pickup_times, delivery_times

    def get_metric(self, model, mod_ods, true_ods, req_ods, pickup_times, delivery_times, scale):
        request_accept = []
        true_accept = []
        serve_levels = []
        request_levels = []
        for k in range(self.K):
            if pickup_times[k] > mod_ods[k][2]:
                request_accept.append(1)
                serve_levels.append(int((pickup_times[k] - mod_ods[k][2])/scale))
                request_levels.append(int((req_ods[k][2] - mod_ods[k][2])/scale))
            else:
                request_accept.append(0)
                serve_levels.append(-1)
                request_levels.append(-1)

            if pickup_times[k] >= true_ods[k][2]:
                true_accept.append(1)
            else:
                true_accept.append(0)

        print('matching customers based on request is %d, but true acceptance is %d' % (sum(request_accept), sum(true_accept)))

        return request_accept, true_accept, serve_levels, request_levels

    def run(self, true_ods, req_ods, mod_ods, k2q, probs, scale, TL=900):
        # create true od pair demand
        nodes_x, arcs_x = self.create_vehicle_network()
        nodes_y, arcs_y = self.create_customer_network(mod_ods)

        D = self.create_distance(arcs_x, arcs_y)
        b_x, b_y = self.create_flow_consumption(nodes_x, nodes_y)

        model, vehicle, customers, values, choices = self.optimize_once(arcs_x, arcs_y, nodes_x, nodes_y, b_x, b_y, mod_ods, probs, D, scale, k2q, TL)

        routes = self.get_routes(model, vehicle)
        pickup_times, delivery_times = self.get_pd_times(model, customers)

        request_accept, true_accept, serve_levels, request_levels = self.get_metric(model, mod_ods, true_ods, req_ods, pickup_times, delivery_times, scale)

        return request_accept, true_accept, serve_levels, request_levels
