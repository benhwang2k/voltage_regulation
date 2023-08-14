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
    def __init__(self, group, nodes, neighbors, lines, neighbor_lines, generator):
        self.group = group
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

            # lagrangian multipliers
            self.lam_active = {}
            self.lam_reactive = {}
            self.lam_current = {}
            self.lam_volt = {}


            self.shared_active = {}
            self.shared_reactive = {}
            self.flow_active = {}
            self.flow_reactive = {}
            self.shared_current = {}
            self.shared_volt = {}
            self.current = {}
            self.volt = {}

            for line in self.neighbor_lines:
                self.flow_reactive[line] = 0
                self.flow_active[line] = 0
                self.lam_active[line] = 0
                self.lam_reactive[line] = 0
                #self.shared_active[line] = 0
                #self.shared_reactive[line] = 0
                self.current[line] = 0
                #self.shared_current[line] = 0
                self.lam_current[line] = 0
                self.volt[line[0]] = 0
                self.volt[line[1]] = 0
                #self.shared_volt[line[0]] = 0
                #self.shared_volt[line[1]] =0
                self.lam_volt[line[0]] = 0
                self.lam_volt[line[1]] = 0


    def update_shared_powers(self, shared_active, shared_reactive, shared_current, shared_volt):
        for line in shared_active:
            if line in self.neighbor_lines:
                self.shared_active[line] = shared_active[line]
                self.shared_reactive[line] = shared_reactive[line]
                self.shared_current[line] = shared_current[line]
        for key in shared_volt:
            if (key in self.nodes) or (key in self.neighbors):
                self.shared_volt[key] = shared_volt[key]


    def update_lam(self, alpha_power, alpha_current, alpha_voltage):
        tot = 0
        for line in self.shared_active:
            subtot = 0
            self.lam_active[line] += alpha_power * (self.flow_active[line] - self.shared_active[line])
            self.lam_reactive[line] += alpha_power * (self.flow_reactive[line] - self.shared_reactive[line])
            self.lam_current[line] += alpha_current * (self.current[line] - self.shared_current[line])
            self.lam_volt[line[0]] += alpha_voltage * (self.volt[line[0]] - self.shared_volt[line[0]])
            self.lam_volt[line[1]] += alpha_voltage * (self.volt[line[1]] - self.shared_volt[line[1]])
            subtot_v = abs(self.volt[line[0]] - self.shared_volt[line[0]])
            subtot_v += abs(self.volt[line[1]] - self.shared_volt[line[1]])
            subtot_p = abs(self.flow_active[line] - self.shared_active[line])
            subtot_q = abs(self.flow_reactive[line] - self.shared_reactive[line])
            subtot_I = abs(self.current[line] - self.shared_current[line])
            print(f"difference in Pij active power line : {line} is {subtot_p} | group other : {self.shared_active[line]}, my group ({self.group}) {self.flow_active[line]}")
            print(f"difference in Qij reactive power line : {line} is {subtot_q} | group other : {self.shared_reactive[line]}, my group ({self.group}) {self.flow_reactive[line]}")
            print(f"difference in I current line : {line} is {subtot_I} | group other : {self.shared_current[line]}, my group ({self.group}) {self.current[line]}")
            print(f"difference in voltage in nodes : {line} is {subtot_v} | group other : ({self.shared_volt[line[0]]},{self.shared_volt[line[1]]}), my group ({self.group}) ({self.volt[line[0]]},{self.volt[line[1]]})")


            tot += subtot_p + subtot_q + subtot_I + subtot_v
            # if line == (0,1) :
                # print(
                #     f"line {line} has calculated current: {self.current[line]},  difference = {self.current[line] - self.shared_current[line]}")
            #print(f"node {line[0]} has voltage {self.volt[line[0]]}, difference = {self.volt[line[1]] - self.shared_volt[line[1]]}")
                # print(f"line {line} has calculated active power : {self.flow_active[line]} and {self.shared_active[line]}, difference = {self.flow_active[line] - self.shared_active[line]}")
                # print(f"line {line} has calculated reactive power : {self.flow_reactive[line]} and {self.shared_reactive[line]}, difference = {self.flow_reactive[line] - self.shared_reactive[line]}")


        return (subtot_p,subtot_q, subtot_I, subtot_v)


    def set_lam(self, other_lam_active, other_lam_reactive, other_lam_current, other_lam_volt):
        for key in self.lam_active:
            self.lam_active[key] = other_lam_active[key]
            self.lam_reactive[key] = other_lam_reactive[key]
            self.lam_current[key] = other_lam_current[key]
        for key in self.lam_volt:
            self.lam_volt[key] = other_lam_volt[key]

    def set_net_load(self, load_act, load_react):
        self.load_act = load_act
        self.load_react = load_react

    def print_params(self):
        for line in self.neighbor_lines:
            # print(f"active power in line {line} is {self.flow_active[line]}")
            # print(f"shared active power in line {line} is {self.shared_active[line]}")
            print(f"lambda active power in line {line} is {self.lam_active[line]}")
            # print(f"reactive power in line {line} is {self.flow_reactive[line]}")
            # print(f"shared reactive power in line {line} is {self.shared_reactive[line]}")
            print(f"lambda reactive power in line {line} is {self.lam_reactive[line]}")


    def calculate_multi(self):
        n = len(self.nodes) + len(self.neighbors)
        buses = self.nodes + self.neighbors

        p_gen = cp.Variable()
        q_gen = cp.Variable()
        constraints = [p_gen <= 2000]
        constraints += [q_gen <= 2000]
        constraints += [p_gen >= -2000]
        constraints += [q_gen >= -2000]

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
            constraints += [V[i] >= 0.89]
            constraints += [V[i] <= 1.21]

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

        f_obj = []
        f_obj += [p_gen]
        f_obj += [60000*(sum(vn_vio.values()) + sum(vp_vio.values()))]
        f_obj += [2000*(sum(pp_vio.values())+ sum(qp_vio.values())+ sum(pn_vio.values())+ sum(qn_vio.values()))]


        for line in self.shared_active:
            # print(f"adding line: {line} to the objective function with lambda active {self.lam_active[line]}")
            # print(f"adding line: {line} to the objective function with lambda reactive {self.lam_reactive[line]}")
            # print(f"adding line: {line} to the objective function with lambda current {self.lam_current[line]}")
            # print(f"adding line: {line} to the objective function with lambda voltage ({self.lam_volt[line[0]]} , {self.lam_volt[line[1]]})")
            # print(f"shared active power flow in line {line} = {self.shared_active[line]}")
            f_obj += [self.lam_active[line] * (Pij[line] - self.shared_active[line])]
            f_obj += [self.lam_reactive[line] * (Qij[line] - self.shared_reactive[line])]
            f_obj += [self.lam_current[line] * (I[line] - self.shared_current[line])]
            f_obj += [self.lam_volt[line[0]] * (V[line[0]] - self.shared_volt[line[0]])]
            f_obj += [self.lam_volt[line[1]] * (V[line[1]] - self.shared_volt[line[1]])]

        prob = cp.Problem(cp.Minimize(sum(f_obj)), constraints)
        prob.solve()
        objective = prob.value
        # 
        # print(f"objective value is : {prob.value}")
        # print(f"pgen at node {self.generator} : {p_gen.value}")
        # print(f"qgen at nod {self.generator} : {q_gen.value}")

        # for node in V:
        #     print(f"voltage at node: {node} = {V[node].value}")

        threshold = 1.0E-5

        for key in pp_vio:
            if pp_vio[key].value > threshold:
                print(f"\033[0;31m ppvio node: {key} : {pp_vio[key].value} \033[0m")
        for key in pn_vio:
            if pn_vio[key].value > threshold:
                print(f"\033[0;31m pnvio node: {key} : {pn_vio[key].value} \033[0m")
        for key in qp_vio:
            if qp_vio[key].value > threshold:
                print(f"\033[0;31m qpvio node: {key} : {qp_vio[key].value} \033[0m")
        for key in qn_vio:
            if qn_vio[key].value > threshold:
                print(f"\033[0;31m qnvio node: {key} : {qn_vio[key].value} \033[0m")
        for key in vp_vio:
            if vp_vio[key].value > threshold:
                print(f"\033[0;31m vpvio node: {key} : {vp_vio[key].value} \033[0m")
        for key in vn_vio:
            if vn_vio[key].value > threshold:
                print(f"\033[0;31m vnvio node: {key} : {vn_vio[key].value} \033[0m")

        shared_active = {}
        shared_reactive = {}
        shared_current = {}
        shared_volt = {}
        # print(f"generation at node {self.generator} = ({p_gen.value} + i {q_gen.value})")
        for line in self.neighbor_lines:
            # print(f"P{line} = {Pij[line].value}")
            # print(f"Q{line} = {Qij[line].value}")
            # print(f"I{line} = {I[line].value}")
            shared_active[line] = Pij[line].value
            self.flow_active[line] = Pij[line].value

            shared_reactive[line] = Qij[line].value
            self.flow_reactive[line] = Qij[line].value
            self.current[line] = I[line].value
            shared_current[line] = I[line].value
            shared_volt[line[0]] = V[line[0]].value
            shared_volt[line[1]] = V[line[1]].value
            self.volt[line[0]] = V[line[0]].value
            self.volt[line[1]] = V[line[1]].value

        return (shared_active, shared_reactive, shared_current, shared_volt, objective)


