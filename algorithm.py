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
        self.nodes = nodes
        self.neighbors = neighbors
        self.lines = lines
        self.neighbor_lines = neighbor_lines
        self.generator = generator
        self.active_injection_constraints = {}
        self.reactive_injection_constraints = {}
        self.line_constraints = {}

        self.lam = {}

        self.Y = pickle.load(open("Line_Ymatrix.pickle", "rb"))
        allnodes = self.neighbors + self.nodes

        #construct map and matricies
        for node in self.nodes:
            if node != self.generator:
                self.active_injection_constraints[node] = (0,0)
                self.reactive_injection_constraints[node] = (0,0)
            else:
                self.active_injection_constraints[node] = (-100, 100)
                self.reactive_injection_constraints[node] = (-100, 100)

        for line in self.lines:
            self.line_constraints[line] = 100000
    
    def set_net_load(self, load_act, load_react, cap_act, cap_react):
        # construct map and matricies
        for node in self.nodes:
            self.active_injection_constraints[node] = (-load_act[node], -load_act[node] + cap_act[node])
            self.reactive_injection_constraints[node] = (-load_react[node], -load_react[node] + cap_react[node])
            # print(f'node {node:d} has active capacity: ', self.active_injection_constraints[node])
            # print(f'node {node:d} has reactive capacity: ', self.reactive_injection_constraints[node])
            # if node != self.generator:
            #     self.active_injection_constraints[node] = (-load_act[node],-load_act[node])
            #     self.reactive_injection_constraints[node] = (-load_react[node],-load_react[node])
            #     print(f'node {node:d} has active capacity: ', self.active_injection_constraints[node])
            #     print(f'node {node:d} has reactive capacity: ', self.reactive_injection_constraints[node])
            # else:
            #     self.active_injection_constraints[node] = (-load_act[node], -load_act[node] + cap_act[node])
            #     self.reactive_injection_constraints[node] = (-load_react[node], -load_react[node] + cap_react[node])
            #     print(f'node {node:d} has active capacity: ', self.active_injection_constraints[node])
            #     print(f'node {node:d} has reactive capacity: ', self.reactive_injection_constraints[node])



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
        n = len(self.nodes) + len(self.neighbors)
        buses = self.nodes + self.neighbors
        # assign each node an index
        index = {}
        count = 0
        for node in buses:
            index[node] = count
            count += 1
        # construct matricies from the dictionary mapping two nodes to the bus admittance matrix entry
        Y = np.array(np.zeros((n,n),dtype=complex))
        for i in buses:
            for k in buses:
                Y[index[i]][index[k]] = self.Y[i][k]

        # For each group node make Ai and Ai_til
        A = {}
        B = {}
        for node in self.nodes:
            A_temp = np.array(np.zeros((n,n),dtype=complex))
            E = np.array(np.zeros((n,n),dtype=complex))
            E[index[node]][index[node]] = 1
            A_temp = (0.5)*(np.transpose(np.conjugate(Y))@E + E@Y)
            B_temp = (1.0/(2j)) * (np.transpose(np.conjugate(Y)) @ E - E @ Y)
            A[node] = A_temp
            B[node] = B_temp

        # For each line make Aik
        A_line = {}
        for line in self.lines:
            Yik = self.Y[line[0]][line[1]]
            A_temp = np.array(np.zeros((n,n),dtype=complex))
            for l in range(n):
                for m in range(n):
                    if l == m and m == index[line[1]]:
                        A_temp[l][m] = np.real(Yik)
                    elif l == index[line[1]] and m == k:
                        A_temp[l][m] = (-Yik)/2
                    elif l == k and m == index[line[1]]:
                        A_temp[l][m] = (-np.transpose(np.conjugate(Yik)))/ 2
                    else:
                        A_temp[l][m] = 0
            A_line[line] = A_temp

        # use (13) as the objective function and (12) as constraints without f, g
        W = cp.Variable((n, n), PSD=True)

        constraints = [W >> 0]

        # This constrains the voltage at all buses even the neighbors
        for node in buses:
            constraints += [(W[index[node]][index[node]] == v[node] * v[node])]


        # Only for the nodes that are in the group (i.e. not the neighbors) constrain the power injection
        constraints += [(cp.real(cp.trace(A[node] @ W)) <= self.active_injection_constraints[node][1]) for node in self.nodes]
        constraints += [(cp.real(cp.trace(A[node] @ W)) >= self.active_injection_constraints[node][0]) for node in self.nodes]
        constraints += [(cp.real(cp.trace(B[node] @ W)) <= self.reactive_injection_constraints[node][1]) for node in self.nodes]
        constraints += [(cp.real(cp.trace(B[node] @ W)) >= self.reactive_injection_constraints[node][0]) for node in self.nodes]

        # For all lines including neighbor lines constrain the power flow
        for line in self.lines:
            constraints += [(cp.abs(cp.trace(A_line[line] @ W)) <= self.line_constraints[line])]


        # construct the objective function as per equations (13) from Alejandros paper mangled to fit (8)
        f = []
        for node in self.nodes:
            f.append(cp.real(cp.trace(A[node]@W)))
        for line in self.neighbor_lines:
            f.append(cp.real(cp.abs(self.lam[line][0]*(W[index[line[0]]][index[line[1]]]-W_shared[line][0])) + cp.abs(self.lam[line][1]*(W[index[line[0]]][index[line[1]]]-W_shared[line][1]))))

        prob = cp.Problem(cp.Minimize(sum(f)), constraints)
        # prob.solve(verbose=True)
        #
        # Print result.
        # print("The optimal value is", prob.value)
        # print("A solution W is")
        # print(W.value)


        prob.solve()

        # Print result.
        # print("The optimal value is", prob.value)
        # print("A solution W is")
        # print(W.value)

        # Return W, and the power setpoints for each node
        P = {}
        Q = {}
        for node in self.nodes:
            P[node] = [np.real(np.trace(A[node] @ W.value))]
            Q[node] = [np.real(np.trace(B[node] @ W.value))]
            # print("----------------------")
            # print("node: ", node)
            # print("active: ", np.trace(A[node] @ W.value))
            # print("reactive", np.trace(B[node]@W.value))
        return (index, W.value, P, Q)




