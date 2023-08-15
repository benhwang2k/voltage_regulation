import cvxpy as cp
import numpy as np
import pickle
import pandas as pd
import mosek # this import is unused, but mosek needs to be installed

n = 48
buses = list(range(n))

# Nodes defining the groups
bus_groups = [
    [7, 31, 33, 32, 47],
    [1, 6, 8, 9, 10, 22, 23, 24, 25, 26, 39, 40],
    [27, 35, 36, 37, 38],
    [0, 2, 14, 15, 16, 17, 18, 19, 41, 42],
    [11, 12, 13, 28, 29, 30, 34],
    [3, 4, 5, 20, 21, 43, 44, 45, 46],
]

N = len(bus_groups)

generators = [47, 23, 37, 15, 13, 45]

# Load the microgrid data
# Scale everything to units of MW (from KW on the sheet)
scale = 1000.0
loads = pd.read_excel("UCSDmicrogrid_iCorev3_info.xlsx",
                      sheet_name="Buses_new").to_numpy()

P = [0]*n
Q = [0]*n
for i in buses:
    P[loads[i+1][1]] = float(loads[i+1][5])/scale
    Q[loads[i+1][1]] = float(loads[i+1][6])/scale
    if not np.isnan(loads[i+1][9]):
        P[loads[i+1][1]] -= float(loads[i+1][9])/scale

network = pd.read_excel("UCSDmicrogrid_iCorev3_info.xlsx",
                        sheet_name="Branches_new").to_numpy()
lines = [(int(network[i][0]), int(network[i][1]))
         for i in range(len(network))]

Y = np.array(pickle.load(
    open("Line_Ymatrix.pickle", "rb")
) * (12470. * 12470. / 1000000.))

r = np.array(np.zeros((n, n)))
x = np.array(np.zeros((n, n)))
g = np.array(np.zeros((n, 1)))
b = np.array(np.zeros((n, 1)))
for i in buses:
    for j in buses:
        y = Y[i][j]
        if y != 0 and i != j:
            r[i][j] = np.real(1/y)
            x[i][j] = np.imag(1/y)

for i in buses:
    tot = 0.+0.j
    for j in buses:
        if i != j:
            tot += Y[i][j]
    y = Y[i][i] + tot
    g[i] = np.real(y)
    b[i] = -np.imag(y)


class Algorithm():
    """
    This is the base functions needed for most algorithms

    Communication
    Topology
    Logging
    """

    # Constructor for group 6
    def __init__(self, group, buses):
        self.group = group
        self.buses = buses
        self.lines = [line for line in lines
                      if line[0] in self.buses or line[1] in self.buses]
        self.neighbors = []
        self.neighbor_lines = []
        for line in lines:
            if line[0] in self.buses and line[1] not in self.buses:
                self.neighbor_lines += [line]
                self.neighbors += [line[1]]
            elif line[1] in self.buses and line[0] not in self.buses:
                self.neighbor_lines += [line]
                self.neighbors += [line[0]]

        # Decision variables are stored here for easy access
        # before and after solving the model

        # Voltage squared
        self.V = [1]*48

        # Current squared
        self.I = {}

        # Power flow on lines
        self.Pij = {}
        self.Qij = {}

        # Power generation
        self.p_gen = [0]*48
        self.q_gen = [0]*48

        # Lagrangian multipliers
        self.lam_P = {}
        self.lam_Q = {}
        self.lam_I = {}
        self.lam_V = {}

        # Shared (coupled) state
        self.P_shr = {}
        self.Q_shr = {}
        self.I_shr = {}
        self.V_shr = {}

        print(
            f"Initialized group {self.group} with\n",
            f"Buses: {self.buses}\n",
            f"Lines: {self.lines}\n",
            f"Neighbors: {self.neighbors}\n",
            f"Neighbor lines: {self.neighbor_lines}\n",
        )

    def build(self):
        constraints = []

        for i in self.buses + self.neighbors:
            if i not in generators:
                self.V[i] = cp.Variable()
                constraints += [self.V[i] >= 0.91]
                constraints += [self.V[i] <= 1.1]
            else:
                self.p_gen[i] = cp.Variable()
                self.q_gen[i] = cp.Variable()
                constraints = [self.p_gen[i] <= 2000]
                constraints += [self.q_gen[i] <= 2000]
                constraints += [self.p_gen[i] >= -2000]
                constraints += [self.q_gen[i] >= -2000]

        for line in self.lines:
            self.I[line] = cp.Variable()
            self.Pij[line] = cp.Variable()
            self.Qij[line] = cp.Variable()

        for i in self.buses:
            Pij_out = 0
            Qij_out = 0
            Pij_in = 0
            Qij_in = 0

            for line in self.lines:
                if i == line[0]:
                    Pij_out += self.Pij[line]
                    Qij_out += self.Qij[line]
                elif i == line[1]:
                    Pij_in += self.Pij[line] - r[line[0]][line[1]] * self.I[line]
                    Qij_in += self.Qij[line] - x[line[0]][line[1]] * self.I[line]

            # Active and reactive power balance
            # The terms g_i*V and b_i*V are zero in this case
            constraints += [-P[i] + self.p_gen[i] == Pij_out - Pij_in]
            constraints += [-Q[i] + self.q_gen[i] == Qij_out - Qij_in]

        self.z = {}
        for line in self.lines:
            r_l = r[line[0]][line[1]]
            x_l = x[line[0]][line[1]]

            constraints += [
                self.V[line[1]] == self.V[line[0]]
                - 2 * (r_l * self.Pij[line] + x_l * self.Qij[line])
                + (r_l*r_l + x_l*x_l) * self.I[line]
               ]

            # Slack variable for constraint VI <= P^2 + Q^2
            self.z[line] = cp.Variable(1)

            # V^.5 * I^.5 <= |z|
            constraints += [cp.PowCone3D(
                cp.vstack([self.V[line[0]]])[0],
                cp.vstack([self.I[line]])[0],
                self.z[line],
                0.5
               )]

            # ||[P;Q]||_2 <= z
            constraints += [cp.SOC(
                self.z[line],
                cp.vstack([self.Pij[line], self.Qij[line]])
               )]

        f_obj = sum(self.p_gen)

        for line in self.neighbor_lines:
            self.lam_P[line] = cp.Parameter(value=0)
            self.lam_Q[line] = cp.Parameter(value=0)
            self.lam_I[line] = cp.Parameter(value=0)
            self.lam_V[line] = cp.Parameter(value=0)
            self.P_shr[line] = cp.Parameter(value=0)
            self.Q_shr[line] = cp.Parameter(value=0)
            self.I_shr[line] = cp.Parameter(value=0)
            self.V_shr[line] = cp.Parameter(value=0)
            lambdas = [
                self.lam_P[line] * (self.Pij[line] - self.P_shr[line]),
                self.lam_Q[line] * (self.Qij[line] - self.Q_shr[line]).
                self.lam_I[line] * (self.I[line] - self.I_shr[line]),
                self.lam_V[line[0]] * (self.V[line[0]] - self.V_shr[line[0]]),
                self.lam_V[line[1]] * (self.V[line[1]] - self.V_shr[line[1]]),
               ]

            f_obj += sum(lambdas)

        self.prob = cp.Problem(cp.Minimize(f_obj), constraints)

    def solve(self):
        self.prob.solve(
            solver=cp.MOSEK,
            verbose=False,
            mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-6}
        )

        print(f"Total generation in group {self.group}: {self.prob.objective.value}")


def test_centralized():
    group_all = Algorithm(0, buses)

    group_all.build()

    group_all.solve()


group = [None]*6
for i in range(6):
    group[i] = Algorithm(
        group=i,
        buses=bus_groups[i],
    )
