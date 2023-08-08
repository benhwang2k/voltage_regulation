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

batteries = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]#[13,15,23,37,45,47]

battery_capacity = 10000  # number used for both kVA and kVAR

for row in range(1,49):
    if table[row][1] == 0:
        load_act[table[row][1]] = float(table[row][5])/6700.0
        load_react[table[row][1]] = float(table[row][6])/6700.0
        if table[row][1] in batteries:
            cap_act[table[row][1]] = battery_capacity/12.470
            cap_react[table[row][1]] = battery_capacity/12.470
    else:
        load_act[table[row][1]] = float(table[row][5])/1247.0
        load_react[table[row][1]] = float(table[row][6])/1247.0
        if table[row][1] in batteries:
            cap_act[table[row][1]] = battery_capacity/12.470
            cap_react[table[row][1]] = battery_capacity/12.470
    if not np.isnan(table[row][9]):
        load_act[table[row][1]] -= float(table[row][9])/1247.0

# Now load and capacity in per unit are initialized
# 
# for node in batteries:
#     print(f'node {node} cap {cap_act[node]}')



#setup the groups and run the optimizations

group1 = algorithm.Algorithm(nodes=[7,8,31,33,32,47], neighbors=[6], lines= [(6,7),(6,8),(7,31),(7,32),(7,33),(33,47)], neighbor_lines=[(6,7),(6,8)], generator = 47)
W_shared = {}
for line in group1.neighbor_lines:
    group1.set_lambda(line,(0,0))
    W_shared[line] = (0,0)
group1.set_net_load(load_act, load_react, cap_act, cap_react)
(index_1, W_1, P_1, Q_1) = group1.calculate_multi(v, W_shared)



group2 = algorithm.Algorithm(nodes=[1,6,9,10,22,23,24], neighbors=[0,7,8,25,26,27], lines=[(0,1),(1,6),(6,7),(6,8),(6,9),(6,10),(10,22),(10,25),(10,26),(10,27),(22,23),(22,24)], neighbor_lines=[(0,1),(6,7),(6,8),(10,25),(10,26),(10,27)], generator=23)
W_shared = {}
for line in group2.neighbor_lines:
    group2.set_lambda(line,(0,0))
    W_shared[line] = (0, 0)
group2.set_net_load(load_act, load_react, cap_act, cap_react)
(index_2, W_2, P_2, Q_2) = group2.calculate_multi(v, W_shared)


group3 = algorithm.Algorithm(nodes=[25,26,27,35,36,37,38,39,40], neighbors=[10], lines=[(10,25),(10,26),(10,27),(26,39),(26,40),(27,35),(27,38),(35,36),(35,37)], neighbor_lines=[(10,25),(10,26),(10,27)], generator=37)
W_shared = {}
for line in group3.neighbor_lines:
    group3.set_lambda(line,(0,0))
    W_shared[line] = (0, 0)
group3.set_net_load(load_act, load_react, cap_act, cap_react)
(index_3, W_3, P_3, Q_3) = group3.calculate_multi(v, W_shared)


group4 = algorithm.Algorithm(nodes=[0,2,14,15,16,17,18,19,41], neighbors=[1,3,28,42], lines=[(0,1),(0,2),(0,3),(2,41),(14,15),(14,16),(14,17),(14,18),(14,19),(14,41),(28,41),(41,42)], neighbor_lines=[(0,1),(0,3),(28,41),(41,42)], generator=15)
W_shared = {}
for line in group4.neighbor_lines:
    group4.set_lambda(line,(0,0))
    W_shared[line] = (0, 0)
group4.set_net_load(load_act, load_react, cap_act, cap_react)
(index_4, W_4, P_4, Q_4) = group4.calculate_multi(v, W_shared)

group5 = algorithm.Algorithm(nodes=[11,12,13,28,29,30,34,42], neighbors=[41], lines=[(11,12),(11,13),(11,28),(28,29),(28,30),(28,41),(29,34),(41,42)], neighbor_lines=[(28,41),(41,42)], generator=13)
W_shared = {}
for line in group5.neighbor_lines:
    group5.set_lambda(line,(0,0))
    W_shared[line] = (0, 0)
group5.set_net_load(load_act, load_react, cap_act, cap_react)
(index_5, W_5, P_5, Q_5) = group5.calculate_multi(v, W_shared)

group6 = algorithm.Algorithm(nodes = [3, 4, 5, 20, 21,43,44,45,46],neighbors = [0], lines = [(0,3),(3,43),(4,43),(4,5),(43,44),(20,43),(20,21),(21,45),(21,46)], neighbor_lines = [(0,3)], generator = 45)
W_shared = {}
for line in group6.neighbor_lines:
    group6.set_lambda(line,(0,0))
    W_shared[line] = (0, 0)
group6.set_net_load(load_act, load_react, cap_act, cap_react)
(index_6, W_6, P_6, Q_6) = group6.calculate_multi(v, W_shared)


# update lambdas
alpha = 100.
group1.lam = group2.update_lambda( alpha, index_2, W_2, index_1, W_1)
group3.lam = group2.update_lambda( alpha, index_2, W_2, index_3, W_3)
group5.lam = group4.update_lambda( alpha, index_4, W_4, index_5, W_5)
group6.lam = group4.update_lambda( alpha, index_4, W_4, index_6, W_6)
group4.update_lambda( alpha, index_4, W_4, index_2, W_2)
group2.update_lambda( alpha, index_2, W_2, index_4, W_4)

# Write a loop to iterate
for t in range(100):
    print("----------------------")
    print("t: ", t)
    # GROUP 1
    W_shared = {}
    for line in group1.neighbor_lines:
        W_shared[line] = (W_2[index_2[line[0]]][index_2[line[1]]], W_2[index_2[line[1]]][index_2[line[0]]])
    (index_1, W_1, P_1, Q_1) = group1.calculate_multi(v, W_shared)

    # GROUP 2
    W_shared = {}
    for line in group2.neighbor_lines:
        if line in group1.neighbor_lines:
            W_shared[line] = (W_1[index_1[line[0]]][index_1[line[1]]], W_1[index_1[line[1]]][index_1[line[0]]])
        elif line in group3.neighbor_lines:
            W_shared[line] = (W_3[index_3[line[0]]][index_3[line[1]]], W_3[index_3[line[1]]][index_3[line[0]]])
        elif line in group4.neighbor_lines:
            W_shared[line] = (W_4[index_4[line[0]]][index_4[line[1]]], W_4[index_4[line[1]]][index_4[line[0]]])
    (index_2, W_2, P_2, Q_2) = group2.calculate_multi(v, W_shared)

    # GROUP 3
    W_shared = {}
    for line in group3.neighbor_lines:
        W_shared[line] = (W_2[index_2[line[0]]][index_2[line[1]]], W_2[index_2[line[1]]][index_2[line[0]]])
    (index_3, W_3, P_3, Q_3) = group3.calculate_multi(v, W_shared)

    # GROUP 4

    W_shared = {}
    for line in group4.neighbor_lines:
        if line in group5.neighbor_lines:
            W_shared[line] = (W_5[index_5[line[0]]][index_5[line[1]]], W_5[index_5[line[1]]][index_5[line[0]]])
        elif line in group6.neighbor_lines:
            W_shared[line] = (W_6[index_6[line[0]]][index_6[line[1]]], W_6[index_6[line[1]]][index_6[line[0]]])
        elif line in group2.neighbor_lines:
            W_shared[line] = (W_2[index_2[line[0]]][index_2[line[1]]], W_2[index_2[line[1]]][index_2[line[0]]])
    (index_4, W_4, P_4, Q_4) = group4.calculate_multi(v, W_shared)

    # GROUP 5
    W_shared = {}
    for line in group5.neighbor_lines:
        W_shared[line] = (W_4[index_4[line[0]]][index_4[line[1]]], W_4[index_4[line[1]]][index_4[line[0]]])
    (index_5, W_5, P_5, Q_5) = group5.calculate_multi(v, W_shared)

    # GROUP 6
    W_shared = {}
    for line in group6.neighbor_lines:
        W_shared[line] = (W_4[index_4[line[0]]][index_4[line[1]]], W_4[index_4[line[1]]][index_4[line[0]]])
    (index_6, W_6, P_6, Q_6) = group6.calculate_multi(v, W_shared)

    group1.lam = group2.update_lambda(alpha, index_2, W_2, index_1, W_1)
    group3.lam = group2.update_lambda(alpha, index_2, W_2, index_3, W_3)
    group5.lam = group4.update_lambda(alpha, index_4, W_4, index_5, W_5)
    group6.lam = group4.update_lambda(alpha, index_4, W_4, index_6, W_6)
    group4.update_lambda(alpha, index_4, W_4, index_2, W_2)
    group2.update_lambda(alpha, index_2, W_2, index_4, W_4)











