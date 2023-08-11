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
        n = len(nodes) + len(neighbors)
        buses = nodes + neighbors
        self.nodes = nodes
        self.neighbors = neighbors
        self.lines = lines
        self.neighbor_lines = neighbor_lines
        self.generator = generator
        self.Y = pickle.load(open("Line_Ymatrix.pickle", "rb")) * (12470. * 12470. / 1000000.)

        # shared lagrangian multipliers
        self.lam = {}
        for line in neighbor_lines:
            rev_line = (line[1],line[0])
            self.lam[line] = 0.00001
            self.lam[rev_line] = 0.00001

        # construct an index for this group
        # map: key=bus_number value=array index in A
        self.index = {}
        count = 0
        for node in buses:
            self.index[node] = count
            count += 1

        # create A, B, A_line
        self.A = {}
        self.B = {}
        row = 0
        col = 0
        Y_temp = np.array(np.zeros((n,n), dtype=complex))
        for i in buses:
            for j in buses:
                Y_temp[self.index[i]][self.index[j]] = self.Y[i][j] * (12470. * 12470. / 1000000.)
        self.Y_group = Y_temp

    def set_net_load(self, load_act, load_react, cap_act, cap_react):
        self.load_act = load_act
        self.load_react = load_react
        self.cap_act = cap_act
        self.cap_react = cap_react


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
                if ((i,k) in self.lines) or ((k,i) in self.lines):
                    # if this is a line in the network, update the associated lambda
                    print(
                        f'line: ({i},{k}), Wi {W_i[index_i[i]][index_i[k]]}, Wk {W_k[index_k[i]][index_k[k]]}, differenc: {W_i[index_i[i]][index_i[k]] - W_k[index_k[i]][index_k[k]]}')
    
                    lam_ik = self.lam[(i, k)] + alpha * (W_i[index_i[i]][index_i[k]] - W_k[index_k[i]][index_k[k]])
                    lam_ik = max(np.real(lam_ik), np.imag(lam_ik))
                    self.lam[(i, k)] = lam_ik
                    lam[(i, k)] = self.lam[(i, k)]
                    # print(f'edge: {(i,k)}, lam: {lam[(i,k)]}')
        return lam

    def set_lambda(self, key, value):
        self.lam[key] = value


    def calculate_multi(self, W_shared):
        n = len(self.nodes) + len(self.neighbors)
        buses = self.nodes + self.neighbors
        # optimization variable
        W = cp.Variable((n, n), complex=True)
        # constraints
        constraints = [ W >> 0]

        f_obj = []
        for i in range(n):
            constraints += [cp.real(W[i][i]) <= 1.05*1.05]
            constraints += [cp.real(W[i][i]) >= 0.95*0.95]

        for node in self.nodes:
            # print(f"node: {node}, active load: {self.load_act[node]}, reactive load: {self.load_react[node]}, cap: {self.cap_react[node]}")
            p_inj = 0.0
            q_inj = 0.0
            for neighbor in buses:
                if ((node,neighbor) in self.lines) or ((neighbor,node) in self.lines):
                    p_inj += np.real(self.Y_group[self.index[node]][self.index[neighbor]]) * cp.real(W[self.index[node]][self.index[neighbor]])  + np.imag(self.Y_group[self.index[node]][self.index[neighbor]]) * cp.imag(W[self.index[node]][self.index[neighbor]])
                    q_inj += -np.imag(self.Y_group[self.index[node]][self.index[neighbor]]) * cp.real(W[self.index[node]][self.index[neighbor]]) + np.real(self.Y_group[self.index[node]][self.index[neighbor]]) * cp.imag(W[self.index[node]][self.index[neighbor]])
            p_inj += cp.real(W[self.index[node]][self.index[node]]) * np.real(self.Y_group[self.index[node]][self.index[node]])
            q_inj += cp.real(W[self.index[node]][self.index[node]]) * -np.imag(self.Y_group[self.index[node]][self.index[node]])
            constraints += [p_inj <= self.load_act[node] + self.cap_act[node]]
            constraints += [p_inj >= self.load_act[node] - self.cap_act[node]]
            constraints += [q_inj <= self.load_react[node] + self.cap_react[node]]
            constraints += [q_inj >= self.load_react[node] - self.cap_react[node]]
            f_obj += [p_inj]

        # for line in self.lines:
        #     i = line[0]
        #     k = line[1]
        #     line_flow = -np.real(self.Y[i][k]) * cp.real(W[self.index[i]][self.index[i]]) + np.real(
        #         self.Y[i][k]) * cp.real(W[self.index[i]][self.index[k]]) - np.imag(self.Y[i][k]) * cp.imag(
        #         W[self.index[i]][self.index[i]])
        #     constraints += [cp.abs(line_flow) <= 100000000]

        for line in self.neighbor_lines:
            rev_line = (line[1],line[0])
            print(f"line: {line}")
            print(f"lamda = {self.lam[line]}")
            print(f"W_shared:  {W_shared[line]}")
            diff = cp.abs(W[self.index[line[0]]][self.index[line[1]]] - W_shared[line])
            diff_rev = cp.abs(W[self.index[line[1]]][self.index[line[0]]] - W_shared[rev_line])
            # f_obj += [diff * self.lam[line]]
            # f_obj += [diff_rev * self.lam[rev_line]]
        
        
        
        # construct the objective function as per equations (13) from Alejandros paper mangled to fit (8)

        # solve the optimization problem
        prob = cp.Problem(cp.Minimize(sum(f_obj)), constraints)
        prob.solve(verbose=True)

        # Print result.
        # print("The optimal value is", prob.value)
        # print("A solution W is")
        # print(W.value)


        return (self.index, W.value)
