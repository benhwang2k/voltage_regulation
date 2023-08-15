import cvxpy as cp
import numpy as np
import pickle
import pandas as pd
import mosek

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

network = pd.read_excel("UCSDmicrogrid_iCorev3_info.xlsx",
                        sheet_name="Branches_new").to_numpy()
lines = [(int(network[i][0]), int(network[i][1]))
         for i in range(1, len(network))]

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
    constraints += [p_gen[i] <= 2000]
    constraints += [q_gen[i] <= 2000]
    constraints += [p_gen[i] >= -2000]
    constraints += [q_gen[i] >= -2000]

# Voltages squared
V = [1]*n
# Currents squared
I = {}

# Power flows on lines
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
        if i == line[1]:
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

    # V^.5 * I^.5 <= |z|
    constraints += [cp.PowCone3D(
        cp.vstack([V[line[0]]])[0],
        cp.vstack([I[line]])[0],
        z[line],
        0.5
    )]

    # ||[P;Q]||_2 <= z
    constraints += [cp.SOC(
        z[line],
        cp.vstack([Pij[line], Qij[line]])
    )]

# TODO: Need to define a cost for the generators
# Right now all costs are uniform
f_obj = sum([p_gen[i] for i in generators])


prob = cp.Problem(cp.Minimize(f_obj), constraints)
prob.solve(solver=cp.MOSEK, verbose=True,
           # Set the relative primal/dual tolerance a bit higher.
           # MOSEK refuses to converge any further on this problem.
           mosek_params={mosek.dparam.intpnt_co_tol_rel_gap: 1e-6})

objective = prob.value
print(f"Total Generation: {objective}")
