import algorithm
import numpy as np
import time
import pandas as pd

# P_up = 100
# P_low = -100
# Q_up = 100
# Q_low = -100
# P_line = 100
# v = np.array([[1],[1],[1]])
# Y = np.array([[1j,2+2j,2+3j],[2+2j,2j,3+3j],[2+3j,3+3j,3j]])
# alg = algorithm.Algorithm()
# alg.set_power_constraints(P_up, P_low, Q_up, Q_low, P_line)
# lam = np.array(np.zeros((3,3)))
# alg.calculate(Y, v, lam)
# print(alg)

nodes = []
v = {}
for i in range(48):
    nodes += [i]
    v[i] = 1.
load_act = {0:0.}
load_react = {0:0.}
cap_act = {13:100000,15:100000,23:100000,37:100000,45:100000,47:100000}
cap_react = {13:100000,15:100000,23:100000,37:100000,45:100000,47:100000}
network_data = pd.read_excel("UCSDmicrogrid_iCorev3_info.xlsx", sheet_name="Buses_new")
table = network_data.to_numpy()

batteries = [7,8,31,33,32,47,1,6,9,10,22,23,24]#[13,15,23,37,45,47]

battery_capacity = 10000  # number used for both kVA and kVAR

for row in range(1,49):
    if table[row][1] == 0:
        load_act[table[row][1]] = float(table[row][5])/67000.
        load_react[table[row][1]] = float(table[row][6])/67000.
    else:
        load_act[table[row][1]] = float(table[row][5])/12470.
        load_react[table[row][1]] = float(table[row][6])/12470.
        if table[row][1] in batteries:
            cap_act[table[row][1]] = battery_capacity/12.470
            cap_react[table[row][1]] = battery_capacity/12.470
    if not np.isnan(table[row][9]):
        load_act[table[row][1]] -= float(table[row][9])/12470.

# Now load and capacity in per unit are initialized




#setup the groups and run the optimizations

group1 = algorithm.Algorithm(nodes=[7,8,31,33,32,47], neighbors=[6], lines= [(6,7),(6,8),(7,31),(7,32),(7,33),(33,47)], neighbor_lines=[(6,7),(6,8)], generator = 47)
W_shared = {}
for line in group1.neighbor_lines:
    group1.set_lambda(line,(0,0))
    W_shared[line] = (0,0)
group1.set_net_load(load_act, load_react, cap_act, cap_react)
(index_1, W_1, P_1, Q_1) = group1.calculate_multi(v, W_shared)
print(W_1)
print(P_1)
print(Q_1)


group2 = algorithm.Algorithm(nodes=[1,6,9,10,22,23,24], neighbors=[0,7,8,25,26,27], lines=[(0,1),(1,6),(6,7),(6,8),(6,9),(6,10),(10,22),(10,25),(10,26),(10,27),(22,23),(22,24)], neighbor_lines=[(0,1),(6,7),(6,8),(10,25),(10,26),(10,27)], generator=23)
W_shared = {}
for line in group2.neighbor_lines:
    group2.set_lambda(line,(0,0))
    W_shared[line] = (0, 0)
group2.set_net_load(load_act, load_react, cap_act, cap_react)
(index_2, W_2, P_2, Q_2) = group2.calculate_multi(v, W_shared)


lam = group2.update_lambda( 1000., index_2, W_2, index_1, W_1)
group1.lam = lam

# Write a loop to iterate
for t in range(100):

    W_shared = {}
    for line in group1.neighbor_lines:
        W_shared[line] = (W_2[index_2[line[0]]][index_2[line[1]]], W_2[index_2[line[1]]][index_2[line[0]]])
    (index_1, W_1, P_1, Q_1) = group1.calculate_multi(v, W_shared)

    # in group 2 only update neighbors with group 1
    W_shared = {}
    for line in group2.neighbor_lines:
        if line in group1.neighbor_lines:
            W_shared[line] = (W_1[index_1[line[0]]][index_1[line[1]]], W_1[index_1[line[1]]][index_1[line[0]]])
        else:
            W_shared[line] = (1.,1.)
    group2.set_net_load(load_act, load_react, cap_act, cap_react)
    (index_2, W_2, P_2, Q_2) = group2.calculate_multi(v, W_shared)

    group1.lam = group2.update_lambda(10., index_2, W_2, index_1, W_1)


# group3 = algorithm.Algorithm(nodes=[25,26,27,35,36,37,38,39,40], neighbors=[10], lines=[(10,25),(10,26),(10,27),(26,39),(26,40),(27,35),(27,38),(35,36),(35,37)], neighbor_lines=[(10,25),(10,26),(10,27)], generator=37)
# W_shared = {}
# for line in group3.neighbor_lines:
#     group3.set_lambda(line,(0,0))
#     W_shared[line] = (0, 0)
# group3.set_net_load(load_act, load_react, cap_act, cap_react)
# group3.calculate_multi(v, W_shared)
#
#
# group4 = algorithm.Algorithm(nodes=[0,2,14,15,16,17,18,19,41], neighbors=[1,3,28,42], lines=[(0,1),(0,2),(0,3),(2,41),(14,15),(14,16),(14,17),(14,18),(14,19),(14,41),(28,41),(41,42)], neighbor_lines=[(0,1),(0,3),(28,41),(41,42)], generator=15)
# W_shared = {}
# for line in group4.neighbor_lines:
#     group4.set_lambda(line,(0,0))
#     W_shared[line] = (0, 0)
# group4.set_net_load(load_act, load_react, cap_act, cap_react)
# group4.calculate_multi(v, W_shared)
#
# group5 = algorithm.Algorithm(nodes=[11,12,13,28,29,30,34,42], neighbors=[41], lines=[(11,12),(11,13),(11,28),(28,29),(28,30),(28,41),(29,34),(41,42)], neighbor_lines=[(28,41),(41,42)], generator=13)
# W_shared = {}
# for line in group5.neighbor_lines:
#     group5.set_lambda(line,(0,0))
#     W_shared[line] = (0, 0)
# group5.set_net_load(load_act, load_react, cap_act, cap_react)
# group5.calculate_multi(v, W_shared)
#
# group6 = algorithm.Algorithm(nodes = [3, 4, 5, 20, 21,43,44,45,46],neighbors = [0], lines = [(0,3),(3,43),(4,43),(4,5),(43,44),(20,43),(20,21),(21,45),(21,46)], neighbor_lines = [(0,3)], generator = 45)
# W_shared = {}
# for line in group6.neighbor_lines:
#     group6.set_lambda(line,(0,0))
#     W_shared[line] = (0, 0)
# group6.set_net_load(load_act, load_react, cap_act, cap_react)
# group6.calculate_multi(v, W_shared)








