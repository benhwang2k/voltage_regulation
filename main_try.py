import algorithm_try
import numpy as np
import time
import pandas as pd

nodes = [] # collect all nodes in the network
v = {}  # dicitonary of refrence voltages mapped from bus numbers
for i in range(48):
    nodes += [i]
    v[i] = 1.
# indicate which nodes have generation capcity
batteries = [47,23,37,15,13,45]


# read network data from files
base_power = 1000.0
network_data = pd.read_excel("UCSDmicrogrid_iCorev3_info.xlsx", sheet_name="Buses_new")
table = network_data.to_numpy()
# fill in nominal load and capacity
load_act = {}
load_react = {}
cap_act = {}
cap_react = {}
power = 1

# get nominal net loads
for row in range(1,49):
    load_act[table[row][1]] = float(table[row][5])/base_power
    load_react[table[row][1]] = float(table[row][6])/base_power
    if not np.isnan(table[row][9]):
        load_act[table[row][1]] -= float(table[row][9])/base_power
# get capcity
W_1 = None
print(f"Power : {power}")
for node in nodes:
    if node in batteries:
        cap_act[node] = 500.0 / base_power
        cap_react[node] = 500.0 / base_power
    else:
        cap_act[node] = power/base_power
        cap_react[node] = power/base_power

group1 = algorithm_try.Algorithm(nodes=[7,8,31,33,32,47], neighbors=[6], lines= [(6,7),(6,8),(7,31),(7,32),(7,33),(33,47)], neighbor_lines=[(6,7),(6,8)], generator = 47)
group1.set_net_load(load_act, load_react, cap_act, cap_react)
W_shared = {}
for node in group1.neighbors:
    W_shared[node] = 1
for line in group1.neighbor_lines:
    rev_line = (line[1],line[0])
    W_shared[line] = 1
    W_shared[rev_line] = 1
(index_1, W_1) = group1.calculate_multi(W_shared)



group2 = algorithm_try.Algorithm(nodes=[1,6,9,10,22,23,24], neighbors=[0,7,8,25,26,27], lines=[(0,1),(1,6),(6,7),(6,8),(6,9),(6,10),(10,22),(10,25),(10,26),(10,27),(22,23),(22,24)], neighbor_lines=[(0,1),(6,7),(6,8),(10,25),(10,26),(10,27)], generator=23)
W_shared = {}
for node in group2.neighbors:
    W_shared[node] = 1
for line in group2.neighbor_lines:
    rev_line = (line[1],line[0])
    W_shared[line] = 1
    W_shared[rev_line] = 1
group2.set_net_load(load_act, load_react, cap_act, cap_react)
(index_2, W_2) = group2.calculate_multi(W_shared)


group3 = algorithm_try.Algorithm(nodes=[25,26,27,35,36,37,38,39,40], neighbors=[10], lines=[(10,25),(10,26),(10,27),(26,39),(26,40),(27,35),(27,38),(35,36),(35,37)], neighbor_lines=[(10,25),(10,26),(10,27)], generator=37)
W_shared = {}
for node in group3.neighbors:
    W_shared[node] = 1
for line in group3.neighbor_lines:
    rev_line = (line[1],line[0])
    W_shared[line] = 1
    W_shared[rev_line] = 1
group3.set_net_load(load_act, load_react, cap_act, cap_react)
(index_3, W_3) = group3.calculate_multi(W_shared)


group4 = algorithm_try.Algorithm(nodes=[0,2,14,15,16,17,18,19,41], neighbors=[1,3,28,42], lines=[(0,1),(0,2),(0,3),(2,41),(14,15),(14,16),(14,17),(14,18),(14,19),(14,41),(28,41),(41,42)], neighbor_lines=[(0,1),(0,3),(28,41),(41,42)], generator=15)
W_shared = {}
for node in group4.neighbors:
    W_shared[node] = 1
for line in group4.neighbor_lines:
    rev_line = (line[1],line[0])
    W_shared[line] = 1
    W_shared[rev_line] = 1
group4.set_net_load(load_act, load_react, cap_act, cap_react)
(index_4, W_4) = group4.calculate_multi(W_shared)

group5 = algorithm_try.Algorithm(nodes=[11,12,13,28,29,30,34,42], neighbors=[41], lines=[(11,12),(11,13),(11,28),(28,29),(28,30),(28,41),(29,34),(41,42)], neighbor_lines=[(28,41),(41,42)], generator=13)
W_shared = {}
for node in group5.neighbors:
    W_shared[node] = 1
for line in group5.neighbor_lines:
    rev_line = (line[1],line[0])
    W_shared[line] = 1
    W_shared[rev_line] = 1
group5.set_net_load(load_act, load_react, cap_act, cap_react)
(index_5, W_5) = group5.calculate_multi(W_shared)

group6 = algorithm_try.Algorithm(nodes = [3, 4, 5, 20, 21,43,44,45,46],neighbors = [0], lines = [(0,3),(3,43),(4,43),(4,5),(43,44),(20,43),(20,21),(21,45),(21,46)], neighbor_lines = [(0,3)], generator = 45)
W_shared = {}
for node in group6.neighbors:
    W_shared[node] = 1
for line in group6.neighbor_lines:
    rev_line = (line[1],line[0])
    W_shared[line] = 1
    W_shared[rev_line] = 1
group6.set_net_load(load_act, load_react, cap_act, cap_react)
(index_6, W_6) = group6.calculate_multi(W_shared)

# update lambdas
alpha = 0.0001
group1.lam = group2.update_lambda( alpha, index_2, W_2, index_1, W_1)
group3.lam = group2.update_lambda( alpha, index_2, W_2, index_3, W_3)
group5.lam = group4.update_lambda( alpha, index_4, W_4, index_5, W_5)
group6.lam = group4.update_lambda( alpha, index_4, W_4, index_6, W_6)
group4.update_lambda( alpha, index_4, W_4, index_2, W_2)
group2.update_lambda( alpha, index_2, W_2, index_4, W_4)

for t in range(10):

    W_shared = {}
    for line in group1.neighbor_lines:
        i = line[0]
        k = line[1]
        if line in group2.neighbor_lines:
            W_shared[(i, k)] = W_2[index_2[i]][index_2[k]]
            #W_shared[(k, i)] = W_2[index_2[k]][index_2[i]]
    (index_1, W_1) = group1.calculate_multi(W_shared)

    W_shared = {}
    for line in group2.neighbor_lines:
        i = line[0]
        k = line[1]
        if line in group1.neighbor_lines:
            W_shared[(i, k)] = W_1[index_1[i]][index_1[k]]
            #W_shared[(k, i)] = W_1[index_1[k]][index_1[i]]
        if line in group3.neighbor_lines:
            W_shared[(i, k)] = W_3[index_3[i]][index_3[k]]
            #W_shared[(k, i)] = W_3[index_3[k]][index_3[i]]
        if line in group4.neighbor_lines:
            W_shared[(i, k)] = W_4[index_4[i]][index_4[k]]
            #W_shared[(k, i)] = W_4[index_4[k]][index_4[i]]
    (index_2, W_2) = group2.calculate_multi(W_shared)

    W_shared = {}
    for line in group3.neighbor_lines:
        i = line[0]
        k = line[1]
        if line in group2.neighbor_lines:
            W_shared[(i, k)] = W_2[index_2[i]][index_2[k]]
            #W_shared[(k, i)] = W_2[index_2[k]][index_2[i]]
    (index_3, W_3) = group3.calculate_multi(W_shared)

    W_shared = {}
    for line in group4.neighbor_lines:
        i = line[0]
        k = line[1]
        if line in group2.neighbor_lines:
            W_shared[(i, k)] = W_2[index_2[i]][index_2[k]]
            #W_shared[(k, i)] = W_2[index_2[k]][index_2[i]]
        if line in group5.neighbor_lines:
            W_shared[(i, k)] = W_5[index_5[i]][index_5[k]]
            #W_shared[(k, i)] = W_5[index_5[k]][index_5[i]]
        if line in group6.neighbor_lines:
            W_shared[(i, k)] = W_6[index_6[i]][index_6[k]]
            #W_shared[(k, i)] = W_6[index_6[k]][index_6[i]]
    (index_4, W_4) = group4.calculate_multi(W_shared)

    W_shared = {}
    for line in group5.neighbor_lines:
        i = line[0]
        k = line[1]
        if line in group4.neighbor_lines:
            W_shared[(i, k)] = W_4[index_4[i]][index_4[k]]
            #W_shared[(k, i)] = W_4[index_4[k]][index_4[i]]
    (index_5, W_5) = group5.calculate_multi(W_shared)

    W_shared = {}
    for line in group6.neighbor_lines:
        i = line[0]
        k = line[1]
        if line in group4.neighbor_lines:
            W_shared[(i, k)] = W_4[index_4[i]][index_4[k]]
            #W_shared[(k, i)] = W_4[index_4[k]][index_4[i]]
    (index_6, W_6) = group6.calculate_multi(W_shared)

    group1.lam = group2.update_lambda(alpha, index_2, W_2, index_1, W_1)
    group3.lam = group2.update_lambda(alpha, index_2, W_2, index_3, W_3)
    group5.lam = group4.update_lambda(alpha, index_4, W_4, index_5, W_5)
    group6.lam = group4.update_lambda(alpha, index_4, W_4, index_6, W_6)
    group4.update_lambda(alpha, index_4, W_4, index_2, W_2)
    group2.update_lambda(alpha, index_2, W_2, index_4, W_4)

