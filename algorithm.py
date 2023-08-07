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
    def __init__(self, nodes, neighbors, lines, neighbor_lines, generator,):
        self.nodes = nodes
        self.neighbors = neighbors
        self.lines = lines
        self.neighbor_lines = neighbor_lines
        self.generator = generator
        self.active_injection_constraints = {}
        self.reactive_injection_constraints = {}
        self.line_constraints = {}
        self.Y = pickle.load(open("Line_Ymatrix.pickle", "rb"))
        allnodes = self.neighbors + self.nodes

        #construct map and matricies
        for node in self.nodes:
            if node != self.generator:
                self.active_injection_constraints[node] = (0,0)
                self.reactive_injection_constraints[node] = (0,0)
            else:
                self.active_injection_constraints[node] = (-100000, 100000)
                self.reactive_injection_constraints[node] = (-100000, 100000)

        for line in self.lines:
            self.line_constraints[line] = 1000000
    
    
    # def __init__(self):
    #     self.nodes = [3, 4, 5, 20, 21,43,44,45,46]
    #     self.neighbors = [0]
    #     self.lines = [(0,3),(3,43),(4,43),(4,5),(43,44),(20,43),(20,21),(21,45),(21,46)]
    #     self.neighbor_lines = [(0,3)]
    #     self.generator = 45
    #     self.active_injection_constraints = {}
    #     self.reactive_injection_constraints = {}
    #     self.line_constraints = {}
    #     self.Y = pickle.load(open("Line_Ymatrix.pickle", "rb"))
    #     allnodes = self.neighbors + self.nodes
    # 
    #     #construct map and matricies
    #     for node in self.nodes:
    #         if node != self.generator:
    #             self.active_injection_constraints[node] = (0,0)
    #             self.reactive_injection_constraints[node] = (0,0)
    #         else:
    #             self.active_injection_constraints[node] = (-100000, 100000)
    #             self.reactive_injection_constraints[node] = (-100000, 100000)
    # 
    #     for line in self.lines:
    #         self.line_constraints[line] = 1000000

    # set constraints
    def set_power_constraints(self, P_up, P_low, Q_up, Q_low, P_line):
        self.P_up = P_up
        self.P_low = P_low
        self.Q_up = P_up
        self.Q_low = Q_low
        self.P_line = P_line

    def calculate_multi(self, v, lam, W_shared):
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
        constraints += [(W[index[node]][index[node]] == v[node] * v[node]) for node in buses]

        # Only for the nodes that are in the group (i.e. not the neighbors) constrain the power injection
        constraints += [(cp.real(cp.trace(A[node] @ W)) <= self.active_injection_constraints[node][1]) for node in self.nodes]
        constraints += [(cp.real(cp.trace(A[node] @ W)) >= self.active_injection_constraints[node][0]) for node in self.nodes]
        constraints += [(cp.real(cp.trace(B[node] @ W)) <= self.reactive_injection_constraints[node][1]) for node in self.nodes]
        constraints += [(cp.real(cp.trace(B[node] @ W)) >= self.reactive_injection_constraints[node][0]) for node in self.nodes]

        # For all lines including neighbor lines constrain the power flow
        #for line in self.lines:
            #constraints += [(cp.abs(cp.trace(A_line[line] @ W)) <= self.line_constraints[line])]


        # construct the objective function as per equations (13) from Alejandros paper mangled to fit (8)
        f = []
        for node in self.nodes:
            f.append(cp.real(cp.trace(A[node]@W)))
        for line in self.neighbor_lines:
            f.append(cp.real(lam[line][0]*(W[index[line[0]]][index[line[1]]]-W_shared[line][0]) + lam[line][1]*(W[index[line[0]]][index[line[1]]]-W_shared[line][1])))

        prob = cp.Problem(cp.Minimize(sum(f)), constraints)
        # prob.solve(verbose=True)
        #
        # Print result.
        # print("The optimal value is", prob.value)
        # print("A solution W is")
        # print(W.value)


        prob.solve(solver=cp.MOSEK)

        # Print result.
        print("MOSEK")
        print("The optimal value is", prob.value)
        print("A solution W is")
        print(W.value)

        #
        #
        # prob.solve(solver=cp.COPT)
        #
        # # Print result.
        # print("COPT")
        # print("The optimal value is", prob.value)
        # print("A solution W is")
        # print(W.value)
        #
        # prob.solve(solver=cp.SDPA)
        #
        # # Print result.
        # print("SDPA")
        # print("The optimal value is", prob.value)
        # print("A solution W is")
        # print(W.value)
        #
        # prob.solve(solver=cp.SCS, max_iters=100, verbose=True)
        #
        # # Print result.
        # print("SCS")
        # print("The optimal value is", prob.value)
        # print("A solution W is")
        # print(W.value)

        # prob.solve(solver=cp.SCS)
        # # Print result.
        # print("SCS")
        # print("The optimal value is", prob.value)
        # print("A solution W is")
        # print(W.value)
        # prob.solve(solver=cp.CVXOPT)
        #
        # # Print result.
        # print("CVXOPT")
        # print("The optimal value is", prob.value)
        # print("A solution W is")
        # print(W.value)



        for node in self.nodes:
            print("-------------------------------")
            print("node: ", node)
            print("constraints")
            print(self.active_injection_constraints[node][0])
            print(self.active_injection_constraints[node][1])
            print(self.reactive_injection_constraints[node][0])
            print(self.reactive_injection_constraints[node][1])
            print("active: ", np.trace(A[node] @ W.value))
            print("reactive", np.trace(B[node]@W.value))



    def calculate(self, Y, v, lam):
        n = 3
        E = np.array(np.zeros((n,n)))
        E[self.node_number][self.node_number] = 1
        A = (0.5)*(np.transpose(np.conjugate(Y))@E + E@Y)
        B = (1.0/(2j)) * (np.transpose(np.conjugate(Y)) @ E - E @ Y)

        # print(A)
        # print(B)

        Ak = []
        for k in range(n-1):
            Ak.append(np.array(np.zeros((n,n),dtype=complex)))
            ind = k
            if k >= self.node_number:
                ind = k + 1
            for l in range(n):
                for m in range(n):
                    if l == m and m == self.node_number:
                        Ak[k][l][m] = np.real(Y[self.node_number][ind])
                    elif l == self.node_number and m == k:
                        Ak[k][l][m] = (-Y[self.node_number][ind])/2
                    elif l == k and m == self.node_number:
                        Ak[k][l][m] = (-np.transpose(np.conjugate(Y[self.node_number][ind])))/ 2
                    else:
                        Ak[k][l][m] = 0

        print(Ak)

        A_til = A
        for i in range(n):
            for j in range(n):
                if i == j:
                    A_til[i][j] = A[i][j];
                elif i < j:
                    A_til[i][j] = A[i][j] + lam[i][j]
                else:
                    A_til[i][j] = A[i][j] - lam[i][j]

        print(A_til)

        W = cp.Variable((n,n), symmetric=True)

        constraints = [W >> 0]
        constraints += [(W[i][i] == v[i]*v[i]) for i in range(n)]
        constraints += [cp.real(cp.trace(A@W)) <= self.P_up]
        constraints += [cp.real(cp.trace(A @ W)) <= self.Q_up]
        constraints += [cp.real(cp.trace(B @ W)) >= self.P_low]
        constraints += [cp.real(cp.trace(B @ W)) >= self.Q_low]
        constraints += [(cp.abs(cp.trace(Ak[i] @ W)) <= self.P_line) for i in range(n-1) ]

        prob = cp.Problem(cp.Minimize(cp.real(cp.trace(A_til @ W))), constraints)
        prob.solve()

        # Print result.
        print("The optimal value is", prob.value)
        print("A solution W is")
        print(W.value)
        P_inj = np.trace(A@W.value)
        Q_inj = np.trace(B@W.value)
        print("Pinj: ",P_inj)
        print("Qinj: ",Q_inj)





