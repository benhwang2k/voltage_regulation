import cvxpy as cp
import numpy as np
import pickle


class Algorithm():
    """
    This is the base functions needed for most algorithms

    Communication
    Topology
    Logging
    """

    # Constructor for group 6
    def __init__(self, nodes, neighbors, lines, neighbor_lines, generator):
        self.solver = "MOSEK"
        self.nodes = nodes
        self.neighbors = neighbors
        self.lines = lines
        self.neighbor_lines = neighbor_lines
        self.generator = generator
        self.active_injection_constraints = {}
        self.reactive_injection_constraints = {}
        self.line_constraints = {}

        self.lam = {}

        self.Y = pickle.load(open("Line_Ymatrix.pickle", "rb")) * (12470.*12470./1000000.)
        allnodes = self.neighbors + self.nodes

        # initialze constraints
        for node in self.nodes:
            if node != self.generator:
                self.active_injection_constraints[node] = (0,0)
                self.reactive_injection_constraints[node] = (0,0)
            else:
                self.active_injection_constraints[node] = (-100, 100)
                self.reactive_injection_constraints[node] = (-100, 100)

        for line in self.lines:
            self.line_constraints[line] = 10000000


        # Construct A_i for each node i in the group
        # also Construct A_ik for each line (i,k) in the group
        n = len(self.nodes) + len(self.neighbors)
        buses = self.nodes + self.neighbors
        # assign each node an self.index
        self.index = {}
        count = 0
        for node in buses:
            self.index[node] = count
            count += 1
        # construct matricies from the dictionary mapping two nodes to the bus admittance matrix entry
        Y = np.array(np.zeros((n, n), dtype=complex))
        for i in buses:
            for k in buses:
                Y[self.index[i]][self.index[k]] = self.Y[i][k]

        # Fmake Ai
        self.A = {}
        self.B = {}
        for node in self.nodes:
            E = np.array(np.zeros((n, n), dtype=complex))
            E[self.index[node]][self.index[node]] = 1
            A_temp = (0.5) * (np.transpose(np.conjugate(Y)) @ E + E @ Y)
            B_temp = (1.0 / (2j)) * (np.transpose(np.conjugate(Y)) @ E - E @ Y)
            self.A[node] = A_temp
            self.B[node] = B_temp

        # For each line make Aik
        self.A_line = {}
        for line in self.lines:
            Yik = self.Y[line[0]][line[1]]
            A_temp = np.array(np.zeros((n, n), dtype=complex))
            for l in range(n):
                for m in range(n):
                    if l == m and m == self.index[line[1]]:
                        A_temp[l][m] = np.real(Yik)
                    elif l == self.index[line[1]] and m == k:
                        A_temp[l][m] = (-Yik) / 2
                    elif l == k and m == self.index[line[1]]:
                        A_temp[l][m] = (-np.transpose(np.conjugate(Yik))) / 2
                    else:
                        A_temp[l][m] = 0
            self.A_line[line] = A_temp
    
    def set_net_load(self, load_act, load_react, cap_act, cap_react):
        # construct map and matricies
        for node in self.nodes:
            self.active_injection_constraints[node] = (load_act[node] - cap_act[node], load_act[node] + cap_act[node])
            self.reactive_injection_constraints[node] = (load_react[node] - cap_react[node], load_react[node] + cap_react[node])




    def update_lambda(self, alpha, index_i, W_i, index_k, W_k):
        lam = {}
        # create a list of shared nodes
        shared_nodes = []
        for node in index_i:
            if node in index_k:
                shared_nodes += [node]

        # check for edges
        for i in shared_nodes:
            for k in shared_nodes:
                if (i,k) in self.lines:
                    # if this is a line in the network, update the associated lambda
                    print(f'line: ({i},{k}), Wi {W_i[index_i[i]][index_i[k]]}, Wk {W_k[index_k[i]][index_k[k]]}, differenc: {W_i[index_i[i]][index_i[k]] - W_k[index_k[i]][index_k[k]]}')

                    lam_ik = self.lam[(i,k)][0] + alpha * (W_i[index_i[i]][index_i[k]] - W_k[index_k[i]][index_k[k]])
                    lam_ki = self.lam[(i, k)][1] + alpha * (W_k[index_k[i]][index_k[k]] - W_i[index_i[i]][index_i[k]])
                    self.lam[(i,k)] = (abs(lam_ik), abs(lam_ki))
                    lam[(i,k)] = self.lam[(i,k)]
                    #print(f'edge: {(i,k)}, lam: {lam[(i,k)]}')
        return lam

    def set_lambda(self, key, value):
        self.lam[key] = value

    def set_power_constraints(self, P_up, P_low, Q_up, Q_low, P_line):
        self.P_up = P_up
        self.P_low = P_low
        self.Q_up = P_up
        self.Q_low = Q_low
        self.P_line = P_line

    def calculate_multi(self, v, W_shared):
        # n is the total number of buses including neighbors -> buses
        n = len(self.nodes) + len(self.neighbors)
        buses = self.nodes + self.neighbors

        # use (13) as the objective function and (12) as constraints without f, g

        # optimization variable is n by b and a positive semi definite
        W = cp.Variable((n, n), PSD=True)

        # list of constraints
        constraints = []

        # This constrains the voltage at all buses even the neighbors
        for node in buses:
            constraints += [(W[self.index[node]][self.index[node]] <= (v[node]+0.05) * (v[node]+0.05))]
            constraints += [(W[self.index[node]][self.index[node]] >= (v[node]-0.05) * (v[node]-0.05))]

        #constrain the power injection Only for the nodes that are in the group (i.e. not the neighbors)
        constraints += [(cp.real(cp.trace(self.A[node] @ W)) <= self.active_injection_constraints[node][1]) for node in self.nodes]
        constraints += [(cp.real(cp.trace(self.A[node] @ W)) >= self.active_injection_constraints[node][0]) for node in self.nodes]
        constraints += [(cp.real(cp.trace(self.B[node] @ W)) <= self.reactive_injection_constraints[node][1]) for node in self.nodes]
        constraints += [(cp.real(cp.trace(self.B[node] @ W)) >= self.reactive_injection_constraints[node][0]) for node in self.nodes]

        # For all lines including neighbor lines constrain the power flow
        # for line in self.lines:
        #     constraints += [(cp.abs(cp.trace(self.A_line[line] @ W)) <= self.line_constraints[line])]


        # construct the objective function as per equations (13) from Alejandros paper mangled to fit (8)
        f = []
        for node in self.nodes:
            f.append(cp.real(cp.trace(self.A[node]@W)))
        for line in self.neighbor_lines:
            f.append(cp.real(cp.abs(self.lam[line][0]*(W[self.index[line[0]]][self.index[line[1]]]-W_shared[line][0])) + cp.abs(self.lam[line][1]*(W[self.index[line[0]]][self.index[line[1]]]-W_shared[line][1]))))

        prob = cp.Problem(cp.Minimize(sum(f)), constraints)



        # SOLVE!!!!!!! if you have Mosek installed the  it will use that as default.
        # Can use "prob.solve(verbose = True)" to see more stuff
        if self.solver == "SCS":
            prob.solve(solver=cp.SCS)
        else:
            try:
                prob.solve()
            except cp.error.SolverError:
                print("ERROR SOlving", self.generator)
                return (self.index, None, [], [])

        # Print result.
        # print("The optimal value is", prob.value)
        # print("A solution W is")
        # print(W.value)

        # print metrics
        # for node in self.nodes:
        #     sat_act = (self.active_injection_constraints[node][0]<= np.real(np.trace(self.A[node] @ W.value))) and (self.active_injection_constraints[node][1]>= np.real(np.trace(self.A[node] @ W.value)))
        #     sat_react = (self.reactive_injection_constraints[node][1] >= np.real(np.trace(self.B[node] @ W.value))) and  (self.reactive_injection_constraints[node][0] <= np.real(np.trace(self.B[node] @ W.value)))
        #     #print(f"node: {node}, satisfied: {sat}, active inj: {np.real(np.trace(A[node] @ W.value))}, lower: {self.active_injection_constraints[node][0]} , upper: {self.active_injection_constraints[node][1]}")
        #     if not (sat_act and sat_react):
        #         print("-----------------")
        #         print(f"node: {node}, active: {sat_act} , reactive: {sat_react}")
        #         print(f"Constraints - active: ({self.active_injection_constraints[node][0]},{self.active_injection_constraints[node][1]}) | react: ({self.reactive_injection_constraints[node][0]},{self.reactive_injection_constraints[node][1]})")
        #         print(f"values - active: {np.real(np.trace(self.A[node] @ W.value))} | reactive: {np.real(np.trace(self.B[node] @ W.value))}")

        # Return W, and the power setpoints for each node
        P = {}
        Q = {}
        if not (W.value is None):
            for node in self.nodes:
                P[node] = np.real(np.trace(self.A[node] @ W.value))
                Q[node] = np.real(np.trace(self.B[node] @ W.value))
                # print("----------------------")
                # print("node: ", node)
                # print("active: ", np.trace(A[node] @ W.value))
                # print("reactive", np.trace(B[node]@W.value))
        return (self.index, W.value, P, Q)




