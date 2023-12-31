import cvxpy as cp
import numpy as np
import pickle
import pandas as pd
import mosek # this import is unused, but mosek needs to be installed
import pickle
import matplotlib.pyplot as plt

DEBUG = False

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

P = [0.]*n
Q = [0.]*n
for i in buses:
    P[loads[i+1][1]] = float(loads[i+1][5])/scale
    Q[loads[i+1][1]] = float(loads[i+1][6])/scale
    if not np.isnan(loads[i+1][9]):
        P[loads[i+1][1]] -= float(loads[i+1][9])/scale

P = [0.0, 0.0, 0.0, 0.0, 0.5456991, 0.8934548999999999, 0.4053246, 4.0337757000000005, 0.3709854000000001, 2.20467, 1.3755114000000002, 0.016491552, 0.0, 0.0, 0.019836216, 0.03126764, 0.0036618460000000003, 0.01630485, 0.00394044, 0.006545634000000001, 0.394044, 0.0, 0.4783635, 0.3877464, 0.5786881, 1.284752, 0.2581329, 0.0, 0.040866753, 0.018872922, 0.021672420000000005, 1.6243005000000001, 1.576176, 1.199182, 0.011295246000000002, 0.295533, 0.197022, 2.0987310000000003, 0.848088, 0.31702199999999997, 0.755199, 0.0, 0.0200022, 0.0, 0.878088, 0.788088, 0.525066, 0.0]
Q = [0.0, 0.0, 0.0, 0.0, 0.40927440000000004, 0.6700910999999999, 0.3039264, 2.9246367, 0.5024061, 1.500165, 1.0316334, 0.012368672999999998, 0.0, 0.0, 0.014787042, 0.02345073, 0.003786606, 0.013923630000000001, 0.00295533, 0.003930924000000001, 0.295533, 0.0, 0.3587727, 0.2908098, 0.680316, 0.963564, 0.1935996, 0.0, 0.030542064, 0.015078693, 0.01625432, 1.2128253, 1.182132, 0.886599, 0.007574913, 0.2216496, 0.14776499999999998, 1.574048, 0.636066, 0.23776650000000002, 0.561, 0.0, 0.01500165, 0.0, 0.658566, 0.591066, 0.4432955, 0.0]
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

alpha_P = 30e-5
alpha_Q = 2e-4
alpha_I = 20e-10
alpha_V = 1e-2


class Algorithm():
    """
    """

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

        # dict of loads
        self.P = {}
        self.Q = {}

        # Decision variables are stored here for easy access
        # before and after solving the model

        # Voltage squared
        self.V = [1.]*48

        # Current squared
        self.I = {}

        # Power flow on lines
        self.Pij = {}
        self.Qij = {}

        # Power generation
        self.p_gen = [0.]*48
        self.q_gen = [0.]*48

        # Lagrangian multipliers
        self.lam_P = {}
        self.lam_Q = {}
        self.lam_I = {}
        self.lam_V = {}




        # Shared (coupled) state
        self.lam_P_shr = {}
        self.lam_Q_shr = {}
        self.lam_I_shr = {}
        self.lam_V_shr = {}




        if DEBUG:
            print(
                f"Initialized group {self.group} with\n",
                f"Buses: {self.buses}\n",
                f"Lines: {self.lines}\n",
                f"Neighbors: {self.neighbors}\n",
                f"Neighbor lines: {self.neighbor_lines}\n",
            )

    def build(self):
        constraints = []
        # for i in buses:
        #     self.P[loads[i + 1][1]] = cp.Parameter(value=float(loads[i + 1][5]) / scale)
        #     self.Q[loads[i + 1][1]] = cp.Parameter(value=float(loads[i + 1][6]) / scale)
        #     if not np.isnan(loads[i + 1][9]):
        #         self.P[loads[i + 1][1]].value -= float(loads[i + 1][9]) / scale

        P_nom = [0.0, 0.0, 0.0, 0.0, 0.5456991, 0.8934548999999999, 0.4053246, 4.0337757000000005, 0.3709854000000001,
             2.20467, 1.3755114000000002, 0.016491552, 0.0, 0.0, 0.019836216, 0.03126764, 0.0036618460000000003,
             0.01630485, 0.00394044, 0.006545634000000001, 0.394044, 0.0, 0.4783635, 0.3877464, 0.5786881, 1.284752,
             0.2581329, 0.0, 0.040866753, 0.018872922, 0.021672420000000005, 1.6243005000000001, 1.576176, 1.199182,
             0.011295246000000002, 0.295533, 0.197022, 2.0987310000000003, 0.848088, 0.31702199999999997, 0.755199, 0.0,
             0.0200022, 0.0, 0.878088, 0.788088, 0.525066, 0.0]
        Q_nom = [0.0, 0.0, 0.0, 0.0, 0.40927440000000004, 0.6700910999999999, 0.3039264, 2.9246367, 0.5024061, 1.500165,
             1.0316334, 0.012368672999999998, 0.0, 0.0, 0.014787042, 0.02345073, 0.003786606, 0.013923630000000001,
             0.00295533, 0.003930924000000001, 0.295533, 0.0, 0.3587727, 0.2908098, 0.680316, 0.963564, 0.1935996, 0.0,
             0.030542064, 0.015078693, 0.01625432, 1.2128253, 1.182132, 0.886599, 0.007574913, 0.2216496,
             0.14776499999999998, 1.574048, 0.636066, 0.23776650000000002, 0.561, 0.0, 0.01500165, 0.0, 0.658566,
             0.591066, 0.4432955, 0.0]
        for i in range(48):
            self.P[i] = cp.Parameter(value=P_nom[i])
            self.Q[i] = cp.Parameter(value=Q_nom[i])

        for i in self.buses + self.neighbors:
            if i not in generators:
                self.V[i] = cp.Variable()
                constraints += [self.V[i] >= 0.91]
                constraints += [self.V[i] <= 1.1]
            else:
                self.p_gen[i] = cp.Variable()
                self.q_gen[i] = cp.Variable()
                if i != 13:
                    constraints += [self.p_gen[i] <= 2000]
                    constraints += [self.q_gen[i] <= 2000]
                    constraints += [self.p_gen[i] >= -2000]
                    constraints += [self.q_gen[i] >= -2000]
                else:
                    constraints += [self.p_gen[i] <= 0.25]
                    constraints += [self.q_gen[i] <= 0.25]
                    constraints += [self.p_gen[i] >= -0.25]
                    constraints += [self.q_gen[i] >= -0.25]


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
            constraints += [-self.P[i] + self.p_gen[i] == Pij_out - Pij_in]
            constraints += [-self.Q[i] + self.q_gen[i] == Qij_out - Qij_in]

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
            self.lam_V[line[0]] = cp.Parameter(value=0)
            self.lam_V[line[1]] = cp.Parameter(value=0)
            self.lam_P_shr[line] = cp.Parameter(value=0)
            self.lam_Q_shr[line] = cp.Parameter(value=0)
            self.lam_I_shr[line] = cp.Parameter(value=0)
            self.lam_V_shr[line[0]] = cp.Parameter(value=0)
            self.lam_V_shr[line[1]] = cp.Parameter(value=0)

            lambdas = [
                self.lam_P[line] * self.Pij[line] - self.lam_P_shr[line],
                self.lam_Q[line] * self.Qij[line] - self.lam_Q_shr[line],
                self.lam_I[line] * self.I[line] - self.lam_I_shr[line],
                self.lam_V[line[0]] * self.V[line[0]] - self.lam_V_shr[line[0]],
                self.lam_V[line[1]] * self.V[line[1]] - self.lam_V_shr[line[1]],
            ]

            f_obj += sum(lambdas)

        self.prob = cp.Problem(cp.Minimize(f_obj), constraints)

    def rebuild(self, fixed_neighbors=[], group=[]):
        constraints = self.prob.constraints
        for neighbor in fixed_neighbors:
            for line in self.neighbor_lines:
                if line[0] in bus_groups[neighbor]:
                    constraints += [self.Pij[line] == group[neighbor].Pij[line]]
                    constraints += [self.Qij[line] == group[neighbor].Qij[line]]
                    constraints += [self.I[line] == group[neighbor].I[line]]
                    constraints += [self.V[line[0]] == group[neighbor].V[line[0]]]
                elif line[1] in bus_groups[neighbor]:
                    constraints += [self.Pij[line] == group[neighbor].Pij[line]]
                    constraints += [self.Qij[line] == group[neighbor].Qij[line]]
                    constraints += [self.I[line] == group[neighbor].I[line]]
                    constraints += [self.V[line[1]] == group[neighbor].V[line[1]]]
        self.prob = cp.Problem(self.prob.objective, constraints)

    def update_lambdas(self, neighbors, update_vals=False):
        for line in self.neighbor_lines:
            for neighbor in neighbors:
                if line not in neighbor.lines:
                    continue

                # The lam_X_shr parameters hold the value of
                # lam_X*X_shr.  This is done to comply with the
                # disciplined parameterized program (DPP) which is
                # required for CVX to cache the program structure
                # between solves.
                self.lam_P[line].value += alpha_P * (self.Pij[line].value - neighbor.Pij[line].value)
                self.lam_Q[line].value += alpha_Q * (self.Qij[line].value - neighbor.Qij[line].value)
                self.lam_I[line].value += alpha_I * (self.I[line].value - neighbor.I[line].value)
                self.lam_V[line[0]].value += alpha_V * (self.V[line[0]].value - neighbor.V[line[1]].value)
                self.lam_V[line[1]].value += alpha_V * (self.V[line[1]].value - neighbor.V[line[0]].value)

                if update_vals:
                    self.lam_P_shr[line].value = self.lam_P[line].value*neighbor.Pij[line].value
                    self.lam_Q_shr[line].value = self.lam_Q[line].value*neighbor.Qij[line].value
                    self.lam_I_shr[line].value = self.lam_I[line].value*neighbor.I[line].value
                    self.lam_V_shr[line[0]].value = self.lam_V[line[0]].value*neighbor.V[line[0]].value
                    self.lam_V_shr[line[1]].value = self.lam_V[line[1]].value*neighbor.V[line[1]].value


    def get_shared(self, neighbors):
        shared = {}
        for line in self.neighbor_lines:
            for neighbor in neighbors:
                if line not in neighbor.lines:
                    continue
                shared[line] = (self.Pij[line].value, self.Qij[line].value, self.I[line].value, self.V[line[0]].value, self.V[line[1]].value)
                # shared[line] = (0, 0, self.I[line].value, self.V[line[0]].value, self.V[line[1]].value)
                # shared[line] = (0, 0, 0, self.V[line[0]].value, self.V[line[1]].value)

        return shared

    def solve(self):
        self.prob.solve(
            solver=cp.MOSEK,
            verbose=False,
            mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-6}
        )



def test_centralized():
    group_all = Algorithm(0, buses)

    group_all.build()

    group_all.solve()

    print(f"Total generation: {[group_all.p_gen[generators[i]].value[()] for i in range(6)]}")
    for i in range(6):
        print(f"centralized generation at bus{generators[i]} is {group_all.p_gen[generators[i]].value[()]}")
    return group_all


#test_centralized()

group = [None]*N
    #group[i] = Algorithm(
        #group=i,
        #buses=bus_groups[i],
    #)
#
    #group[i].build()

for i in range(N):
    group[i] = Algorithm(group=i, buses=bus_groups[i])
    group[i].build()
    #group[i] = pickle.load(open(f'bin/group{i}.pickle', 'rb'))

s1 = [1]*6
s2 = [0]*6
t = 0

shared = {}
for i in range(6):
    shared[i] = {}

share_rate = 1
fixed_groups = []
while sum([(e1 - e2)**2 for e1, e2 in zip(s1, s2)])**0.5 > 1e-6 and t < 2000:
    s2 = s1

    # solve each sub problem
    for i in range(N):
        group[i].solve()

    # share the values
    update_vals = (t % share_rate == 0)
    for i in range(N):
        shared_vals = group[i].get_shared(group)
        for line in shared_vals:
            if not (line in shared[i]):
                shared[i][line] = [shared_vals[line]]
            else:
                shared[i][line] += [shared_vals[line]]
        group[i].update_lambdas(group, update_vals)

    # check differences
    new_fixed_groups = []
    for i in range(N):
        # get each difference
        total = 0
        for line in group[i].neighbor_lines:
            for j in range(N):
                if j != i:
                    if line in group[j].neighbor_lines:
                        total += abs(group[i].Pij[line].value - group[j].Pij[line].value)
                        total += abs(group[i].Qij[line].value - group[j].Qij[line].value)
                        total += abs(group[i].I[line].value - group[j].I[line].value)
                        total += abs(group[i].V[line[0]].value - group[j].V[line[0]].value)
                        total += abs(group[i].V[line[1]].value - group[j].V[line[1]].value)
        #print(f"total: {total}")
        if total <= 0.1:
            new_fixed_groups += [i]

    if len(new_fixed_groups) > len(fixed_groups):
        fixed_groups = new_fixed_groups
        for i in range(N):
            group[i].rebuild(fixed_groups, group)

    s1 = [g.prob.objective.value for g in group]
    print(f"fixed groups: {fixed_groups}")
    print(f"Iter: {t},  {[(e1 - e2)**2 for e1, e2 in zip(s1, s2)]}", end ='\r' )#[str(g.prob.objective.value)[0:10] for g in group], end='\r')
    t = t + 1
    if fixed_groups == [0, 1, 2, 3, 4, 5]:
        print(f"tollerance: {sum([(e1 - e2)**2 for e1, e2 in zip(s1, s2)])**0.5}")
        break
print(f"t: {t} Generation: {[group[i].p_gen[generators[i]].value[()] for i in range(6)]}")
shared_lines = [(0,1), (0,3), (6,7), (10,27), (41,28)]
diff_p = {}
diff_q = {}
diff_i = {}
diff_v0 = {}
diff_v1 = {}
for line in shared_lines:
    diff_p[line] = []
    diff_q[line] = []
    diff_i[line] = []
    diff_v0[line] = []
    diff_v1[line] = []
for i in range(6):
    j = i + 1
    while j < 6:
        for k in range(t):
            for line in shared[i]:
                if line in shared[j]:
                    diff_p[line].append(abs(shared[i][line][k][0] - shared[j][line][k][0]))
                    diff_q[line].append(abs(shared[i][line][k][1] - shared[j][line][k][1]))
                    diff_i[line].append(abs(shared[i][line][k][2] - shared[j][line][k][2]))
                    diff_v0[line].append(abs(shared[i][line][k][3] - shared[j][line][k][3]))
                    diff_v1[line].append(abs(shared[i][line][k][4] - shared[j][line][k][4]))
        j += 1
test_name = "Fix Params "

plt.figure()
for line in shared_lines:
    plt.plot(diff_p[line][100:t])
plt.title("Shared Params = P,Q,I,V | Active Power Diffs")
plt.legend(["(0,1)", "(0,3)", "(6,7)", "(10,27)", "(41,28)"])
plt.savefig("./Graphs/"+test_name+"Active Power Diffs.png")
plt.figure()
for line in shared_lines:
    plt.plot(diff_q[line][100:t])
plt.legend(["(0,1)", "(0,3)", "(6,7)", "(10,27)", "(41,28)"])
plt.title("Shared Params = P,Q,I,V | Reactive Power Diffs")
plt.savefig("./Graphs/"+test_name+"Reactive Power Diffs.png")
plt.figure()
for line in shared_lines:
    plt.plot(diff_i[line][100:t])
plt.legend(["(0,1)", "(0,3)", "(6,7)", "(10,27)", "(41,28)"])
plt.title("Shared Params = P,Q,I,V | Current Diffs")
plt.savefig("./Graphs/"+test_name+"Current Diffs.png")
plt.figure()
for line in shared_lines:
    plt.plot(diff_v0[line])
plt.legend(["(0,1)", "(0,3)", "(6,7)", "(10,27)", "(41,28)"])
plt.title("Shared Params =  P,Q,I,V | Voltage at first bus Diffs")
plt.savefig("./Graphs/"+test_name+"Voltage ast first bus Diffs.png")
plt.figure()
for line in shared_lines:
    plt.plot(diff_v1[line])
plt.legend(["(0,1)", "(0,3)", "(6,7)", "(10,27)", "(41,28)"])
plt.title("Shared Params =  P,Q,I,V | Voltage at second bus Diffs")
plt.savefig("./Graphs/"+test_name+"Voltage at second bus Diffs.png")
plt.figure()
agg = {}
for line in shared_lines:
    agg[line] = []
    for k in range(t):
        agg[line].append(diff_v1[line][k] + diff_v0[line][k] + diff_i[line][k] + diff_q[line][k] + diff_p[line][k])
for line in shared_lines:
    plt.plot(agg[line])
plt.legend(["(0,1)", "(0,3)", "(6,7)", "(10,27)", "(41,28)"])
plt.title("Shared Params = P,Q,I,V | sum of abs of all diffs (p,q,i,v)")
plt.savefig("./Graphs/"+test_name+"aggregate.png")
plt.figure()
agg = {}
for line in shared_lines:
    agg[line] = []
    for k in range(t):
        agg[line].append(diff_v1[line][k] + diff_v0[line][k] + diff_i[line][k] + diff_q[line][k] + diff_p[line][k])
for line in shared_lines:
    plt.plot(agg[line][100:t])
plt.legend(["(0,1)", "(0,3)", "(6,7)", "(10,27)", "(41,28)"])
plt.title("Shared Params = P,Q,I,V | sum of abs of all diffs (p,q,i,v)")
plt.savefig("./Graphs/"+test_name+"aggregate after 100.png")

# print("--------------")
# print("Total: ", sum([g.prob.objective.value for g in group]))
# print(f"Final generator values (iteration {t}): ", [g.p_gen[generators[g.group]].value[()] for g in group])
# print(f"Final generator values (iteration {t}): ", [g.q_gen[generators[g.group]].value[()] for g in group])
# print("Total: ", sum([g.prob.objective.value for g in group]))