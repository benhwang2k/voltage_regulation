import cvxpy as cp
import numpy as np
import pickle
import pandas as pd
import mosek
import time

n = 48
buses = range(n)

generators = [47, 23, 37, 15, 13, 45]

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


constraints = []
p_gen = [0]*n
q_gen = [0]*n
for i in generators:
    p_gen[i] = cp.Variable()
    q_gen[i] = cp.Variable()
    constraints += [p_gen[i] <= 20000]
    constraints += [q_gen[i] <= 20000]
    constraints += [p_gen[i] >= -20000]
    constraints += [q_gen[i] >= -20000]

# Voltages squared
V = [1]*n
# Currents squared
I = {}

# Power flows on line
Pij = {}
Qij = {}

# Voltage decision variables and constraints
for i in buses:
    if i not in generators:
        V[i] = cp.Variable()
        constraints += [V[i] >= 0.91]
        constraints += [V[i] <= 1.1]

# Current decision variables
for line in lines:
    I[line] = cp.Variable()
    Pij[line] = cp.Variable()
    Qij[line] = cp.Variable()

# Load constraints
for i in buses:
    Pij_out = 0
    Qij_out = 0
    Pij_in = 0
    Qij_in = 0

    for line in lines:
        if i == line[0]:
            Pij_out += Pij[line]
            Qij_out += Qij[line]
        elif i == line[1]:
            Pij_in += Pij[line] - r[line[0]][line[1]] * I[line]
            Qij_in += Qij[line] - x[line[0]][line[1]] * I[line]

    # Active and reactive power balance
    # The terms g_i*V and b_i*V are zero in this case
    constraints += [-P[i] + p_gen[i] == Pij_out - Pij_in]
    constraints += [-Q[i] + q_gen[i] == Qij_out - Qij_in]

z = {}
for line in lines:
    r_l = r[line[0]][line[1]]
    x_l = x[line[0]][line[1]]

    constraints += [V[line[1]] == V[line[0]]
                    - 2*(r_l*Pij[line] + x_l*Qij[line])
                    + (r_l*r_l + x_l*x_l) * I[line]]

    # Slack variable for constraint VI <= P^2 + Q^2
    z[line] = cp.Variable(1)

    # V^.5 * I^.5 >= |z|
    print(f"shape V = {np.shape(cp.vstack([V[line[0]]])[0][0])}")
    print(f"shape I = {np.shape(cp.vstack([I[line]])[0][0])}")
    print(f"shape z = {np.shape(z[line][0])}")
    print(f"shape 0.5 = {np.shape(0.5)}")
    constraints += [cp.PowCone3D(
        cp.vstack([V[line[0]]])[0],
        cp.vstack([I[line]])[0],
        z[line],
        (0.5,)
    )]

    #  ||[P;Q]||_2  <= z 
    constraints += [cp.SOC(
        z[line],
        cp.vstack([Pij[line], Qij[line]])
    )]

# TODO: Need to define a cost for the generators
# Right now all costs are uniform

f_obj = sum([p_gen[i] for i in generators])
#f_obj = sum([I[line]*r[line[0]][line[1]] for line in lines])

prob = cp.Problem(cp.Minimize(f_obj), constraints)
tic = time.time()
prob.solve(solver=cp.MOSEK, verbose=False,
           # Set the relative primal/dual tolerance a bit higher.
           # MOSEK refuses to converge any further on this problem.
           mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-6})
print(f"time elpased = {time.time() - tic}")
objective = [f"gen {i} : {p_gen[i].value}" for i in generators]
print(f"Total Generation: {objective}")
