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
        self.Y = np.array(pickle.load(open("Line_Ymatrix.pickle", "rb")) * (12470. * 12470. / 1000000.))
        self.r = np.array(np.zeros((48,48)))
        self.x = np.array(np.zeros((48,48)))
        self.g = np.array(np.zeros((48, 1)))
        self.b = np.array(np.zeros((48, 1)))
        for i in buses:
            for j in buses:
                if i != j:
                    y = -self.Y[i][j]
                    self.r[i][j] = np.real(1/y)
                    self.x[i][j] = np.imag(1/y)
                    #print(f"line ({i},{j}) x: {self.x[i][j]}")
                    #print(f"line ({i},{j}) r: {self.r[i][j]}")
        for i in buses:
            tot = 0.+0.j
            for j in buses:
                if i != j:
                    tot += self.Y[i][j]
            y = self.Y[i][i] + tot
            self.r[i][i] = np.real(1/y)
            self.x[i][i] = np.imag(1/y)
            self.g[i] = np.real(y)
            self.b[i] = -np.imag(y)
            #print(f"line ({i},{i}) x: {self.x[i][i]}")
            #print(f"line ({i},{i}) r: {self.r[i][i]}")
            #print(f"node ({i}) g: {self.g[i]}")
            #print(f"node ({i}) b: {self.b[i]}")
            # print(f"Z = {self.r[i][i]} + i{self.x[i][i]}")
            # print(f"Y = {self.g[i]} + i{self.b[i]}")


    def set_net_load(self, load_act, load_react):
        self.load_act = load_act
        self.load_react = load_react




    def calculate_multi(self):
        n = len(self.nodes) + len(self.neighbors)
        buses = self.nodes + self.neighbors

        p_gen = cp.Variable()
        q_gen = cp.Variable()
        constraints = [p_gen <= 1000]
        constraints += [q_gen <= 1000]
        constraints += [p_gen >= -1000]
        constraints += [q_gen >= -1000]

        #voltages squared
        V = {}
        #currents squared
        I = {}
        # Powers
        Pij = {}
        Qij = {}
        #
        pp_vio = {}
        pn_vio = {}
        qp_vio = {}
        qn_vio = {}
        vp_vio = {}
        vn_vio = {}

        for i in buses:
            V[i] = cp.Variable()
            constraints += [V[i] >= 0.85]
            constraints += [V[i] <= 1.15]

        for line in self.lines:
            I[line] = cp.Variable()
            Pij[line] = cp.Variable()
            Qij[line] = cp.Variable()
            # P[line]
            # Z = self.r[line[0]][line[1]] + 1j*(self.x[line[0]][line[1]])
            # S[line] = Z*I[line]
            # P[line] = cp.real(S[li])

        for node in self.nodes:

            pp_vio[node] = cp.Variable()
            pn_vio[node] = cp.Variable()
            qp_vio[node] = cp.Variable()
            qn_vio[node] = cp.Variable()
            constraints += [pp_vio[node] >= 0]
            constraints += [pn_vio[node] >= 0]
            constraints += [qp_vio[node] >= 0]
            constraints += [qn_vio[node] >= 0]

            Pik = 0
            Qik = 0

            for line in self.lines:
                if node == line[0]:
                    Pik += Pij[line]
                    Qik += Qij[line]
                if node == line[1]:
                    Pik += Pij[line] - self.r[line[0]][line[1]] * I[line]
                    Qik += Qij[line] - self.x[line[0]][line[1]] * I[line]
            if node == self.generator:
                #print(f'generation balance constraint node: {node} ')
                constraints += [V[node] == 1.0]
                constraints += [-self.load_act[node] + p_gen == (Pik  - self.g[node] * V[node]) + pp_vio[node] - pn_vio[node]]
                constraints += [-self.load_react[node] + q_gen == (Qik  - self.b[node] * V[node]) + qp_vio[node] - qn_vio[node]]
            else:
                #print(f"Load balance constraint node : {node}")
                constraints += [-self.load_act[node] == (Pik  - self.g[node] * V[node])+ pp_vio[node] - pn_vio[node]]
                constraints += [-self.load_react[node] == (Qik  - self.b[node] * V[node])+ qp_vio[node] - qn_vio[node]]


        z = {}
        pq = {}
        Velem = {}
        Ielem = {}
        for line in self.lines:
            vn_vio[line] = cp.Variable()
            vp_vio[line] = cp.Variable()


            constraints += [vp_vio[line] >= 0]
            constraints += [vn_vio[line] >= 0]
            r = self.r[line[0]][line[1]]
            x = self.x[line[0]][line[1]]
            constraints += [V[line[1]] == V[line[0]] - 2*(r*Pij[line] + x * Qij[line]) + (r*r + x*x)*I[line] + vp_vio[line] - vn_vio[line]]

            # approximate constraint

            # constraints += [I[line] >= cp.square(Pij[line]) + cp.square(Qij[line])]

            # conic constraint
            z[line] = cp.Variable(1)
            pq[line] = cp.Variable(2)
            Velem[line[0]] = cp.Variable(1)
            Ielem[line] = cp.Variable(1)
            constraints += [Velem[line[0]][0] == V[line[0]]]
            constraints += [Ielem[line][0] == I[line]]
            constraints += [pq[line][0] == Pij[line]]
            constraints += [pq[line][1] == Qij[line]]
            constraints += [cp.PowCone3D(Velem[line[0]], Ielem[line], z[line], 0.5)]
            constraints += [cp.SOC(z[line], pq[line])]



        prob = cp.Problem(cp.Minimize(p_gen + 1000*(sum(vn_vio.values()) + sum(vp_vio.values())+ sum(pp_vio.values())+ sum(qp_vio.values())+ sum(pn_vio.values())+ sum(qn_vio.values()))), constraints)
        prob.solve(verbose=False)

        print(f"pgen : {p_gen.value}")
        print(f"qgen : {q_gen.value}")

        for node in V:
            print(f"voltage at node: {node} = {V[node].value}")

        for key in pp_vio:
            print(f"ppvio node: {key} : {pp_vio[key].value}")
        for key in pn_vio:
            print(f"pnvio node: {key} : {pn_vio[key].value}")
        for key in qp_vio:
            print(f"qpvio node: {key} : {qp_vio[key].value}")
        for key in qn_vio:
            print(f"qnvio node: {key} : {qn_vio[key].value}")
        for key in vp_vio:
            print(f"vpvio node: {key} : {vp_vio[key].value}")
        for key in vn_vio:
            print(f"vnvio node: {key} : {vn_vio[key].value}")




