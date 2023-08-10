import algorithm
import numpy as np
import time
import pandas as pd

# Negtive injection means power output from the node

# record time elapsed
tic = time.time()


nodes = [] # collect all nodes in the network
v = {}  # dicitonary of refrence voltages mapped from bus numbers
for i in range(48):
    nodes += [i]
    v[i] = 1.

# The following dictionaries map bus numbers to net load (active and reactive)
load_act = {0:0.}
load_react = {0:0.}
cap_act = {}
cap_react = {}

# read network data from files
network_data = pd.read_excel("UCSDmicrogrid_iCorev3_info.xlsx", sheet_name="Buses_new")
table = network_data.to_numpy()


# This is the list of busses that have capacity for generation
#batteries = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]#[13,15,23,37,45,47]
batteries = [47,23,37,15,13,45]

battery_capacity = 1000000000  # number used for both kVA and kVAR (the capacity indicates the upper and lower injection bounds as (nom -cap, nom + cap)

# capacity for non battery buses.
other_capacity = 950.

# base power used in per unit calculations
base_power = 1000.

# This block of code fills in the nominal loads, and capacities at each bus into the dictionaries
for row in range(1,49):
    if table[row][1] == 0:
        load_act[table[row][1]] = float(table[row][5])/base_power
        load_react[table[row][1]] = float(table[row][6])/base_power
        if table[row][1] in batteries:
            cap_act[table[row][1]] = battery_capacity/base_power
            cap_react[table[row][1]] = battery_capacity/base_power
        else:
            cap_act[table[row][1]] = other_capacity / base_power
            cap_react[table[row][1]] = other_capacity / base_power
    else:
        load_act[table[row][1]] = float(table[row][5])/base_power
        load_react[table[row][1]] = float(table[row][6])/base_power
        if table[row][1] in batteries:
            cap_act[table[row][1]] = battery_capacity / base_power
            cap_react[table[row][1]] = battery_capacity / base_power
        else:
            cap_act[table[row][1]] = other_capacity / base_power
            cap_react[table[row][1]] = other_capacity / base_power
    if not np.isnan(table[row][9]):
        load_act[table[row][1]] -= float(table[row][9])/base_power

# Now load and capacity in per unit are initialized


# for node in batteries:
#     print(f'node {node} cap {cap_act[node]}')


#setup the groups and run the optimizations

# nodes are internal nodes only, neighbors are nodes in other groups, lines contains all lines (even neighbors lines)
# neighborlines specifies which lines have associated lambdas, generator indicated which bus the battery is at
group1 = algorithm.Algorithm(nodes=[7,8,31,33,32,47], neighbors=[6], lines= [(6,7),(6,8),(7,31),(7,32),(7,33),(33,47)], neighbor_lines=[(6,7),(6,8)], generator = 47)

# W_shared records the neighborign entries of W. (i.e. W superscript(neighbor group))
# keys for this dictionary are lines , and map to entries of the neighbor group's optimal W
W_shared = {}
for line in group1.neighbor_lines:
    group1.set_lambda(line,(0,0)) # initailze lambda
    W_shared[line] = (1,1) # initialize to some number (1)

# set bounds on power injection
group1.set_net_load(load_act, load_react, cap_act, cap_react)

# RUN THE OPTIMIZATION:
# inputs are the target voltages at each bus and the necesary neighbor W entries
# ouputs are an indexing dictionary (key: bus number, value: index in the matrix W_1),
#       W_1 is the optimal solution (symmetric matrix of squared voltages)
#       P_1, Q_1 are the active and reactive power injections at nodes, access these useing P[index[node]]

(index_1, W_1, P_1, Q_1) = group1.calculate_multi(v, W_shared)



group2 = algorithm.Algorithm(nodes=[1,6,9,10,22,23,24], neighbors=[0,7,8,25,26,27], lines=[(0,1),(1,6),(6,7),(6,8),(6,9),(6,10),(10,22),(10,25),(10,26),(10,27),(22,23),(22,24)], neighbor_lines=[(0,1),(6,7),(6,8),(10,25),(10,26),(10,27)], generator=23)
W_shared = {}
for line in group2.neighbor_lines:
    group2.set_lambda(line,(0,0))
    W_shared[line] = (1,1)
group2.set_net_load(load_act, load_react, cap_act, cap_react)
(index_2, W_2, P_2, Q_2) = group2.calculate_multi(v, W_shared)


group3 = algorithm.Algorithm(nodes=[25,26,27,35,36,37,38,39,40], neighbors=[10], lines=[(10,25),(10,26),(10,27),(26,39),(26,40),(27,35),(27,38),(35,36),(35,37)], neighbor_lines=[(10,25),(10,26),(10,27)], generator=37)
W_shared = {}
for line in group3.neighbor_lines:
    group3.set_lambda(line,(0,0))
    W_shared[line] = (1,1)
group3.set_net_load(load_act, load_react, cap_act, cap_react)
(index_3, W_3, P_3, Q_3) = group3.calculate_multi(v, W_shared)


group4 = algorithm.Algorithm(nodes=[0,2,14,15,16,17,18,19,41], neighbors=[1,3,28,42], lines=[(0,1),(0,2),(0,3),(2,41),(14,15),(14,16),(14,17),(14,18),(14,19),(14,41),(28,41),(41,42)], neighbor_lines=[(0,1),(0,3),(28,41),(41,42)], generator=15)
W_shared = {}
for line in group4.neighbor_lines:
    group4.set_lambda(line,(0,0))
    W_shared[line] = (1,1)
group4.set_net_load(load_act, load_react, cap_act, cap_react)
(index_4, W_4, P_4, Q_4) = group4.calculate_multi(v, W_shared)

group5 = algorithm.Algorithm(nodes=[11,12,13,28,29,30,34,42], neighbors=[41], lines=[(11,12),(11,13),(11,28),(28,29),(28,30),(28,41),(29,34),(41,42)], neighbor_lines=[(28,41),(41,42)], generator=13)
W_shared = {}
for line in group5.neighbor_lines:
    group5.set_lambda(line,(0,0))
    W_shared[line] = (1,1)
group5.set_net_load(load_act, load_react, cap_act, cap_react)
(index_5, W_5, P_5, Q_5) = group5.calculate_multi(v, W_shared)

group6 = algorithm.Algorithm(nodes = [3, 4, 5, 20, 21,43,44,45,46],neighbors = [0], lines = [(0,3),(3,43),(4,43),(4,5),(43,44),(20,43),(20,21),(21,45),(21,46)], neighbor_lines = [(0,3)], generator = 45)
W_shared = {}
for line in group6.neighbor_lines:
    group6.set_lambda(line,(0,0))
    W_shared[line] = (1,1)
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
t = 0
dec = False
while t < 100:
    t += 1
    # if True:
    #     other_capacity /= 2
    # dec = True
    # print(other_capacity)
    # # if other_capacity < 930:
    # #     group1.solver = "SCS"
    # #     group2.solver = "SCS"
    # #     group3.solver = "SCS"
    # #     group4.solver = "SCS"
    # #     group5.solver = "SCS"
    # #     group6.solver = "SCS"

    # cap_act = {}
    # cap_react = {}
    # for node in range(48):
    #     if node in batteries:
    #         cap_act[node] = battery_capacity / base_power
    #         cap_react[node] = battery_capacity / base_power
    #     else:
    #         cap_act[node] = other_capacity / base_power
    #         cap_react[node] = other_capacity / base_power
    #
    # group1.set_net_load(load_act, load_react, cap_act, cap_react)
    # group2.set_net_load(load_act, load_react, cap_act, cap_react)
    # group3.set_net_load(load_act, load_react, cap_act, cap_react)
    # group4.set_net_load(load_act, load_react, cap_act, cap_react)
    # group5.set_net_load(load_act, load_react, cap_act, cap_react)
    # group6.set_net_load(load_act, load_react, cap_act, cap_react)

    print("----------------------")
    print("t: ", t)
    # GROUP 1
    W_shared = {}
    for line in group1.neighbor_lines:
        W_shared[line] = (W_2[index_2[line[0]]][index_2[line[1]]], W_2[index_2[line[1]]][index_2[line[0]]])
    (index_1, W_1_test, P_1, Q_1) = group1.calculate_multi(v, W_shared)
    if not (W_1_test is None):
        W_1 = W_1_test
        dec = dec & True
    else:
        print("1 failed")
        dec = False


    # GROUP 2
    W_shared = {}
    for line in group2.neighbor_lines:
        if line in group1.neighbor_lines:
            W_shared[line] = (W_1[index_1[line[0]]][index_1[line[1]]], W_1[index_1[line[1]]][index_1[line[0]]])
        elif line in group3.neighbor_lines:
            W_shared[line] = (W_3[index_3[line[0]]][index_3[line[1]]], W_3[index_3[line[1]]][index_3[line[0]]])
        elif line in group4.neighbor_lines:
            W_shared[line] = (W_4[index_4[line[0]]][index_4[line[1]]], W_4[index_4[line[1]]][index_4[line[0]]])
    (index_2, W_2_test, P_2, Q_2) = group2.calculate_multi(v, W_shared)
    if not (W_2_test is None):
        W_2 = W_2_test
        dec = dec & True
    else:
        W_2 = np.array(np.ones((13,13)))
        print("2 failed")
        dec = False

    # GROUP 3
    W_shared = {}
    for line in group3.neighbor_lines:
        W_shared[line] = (W_2[index_2[line[0]]][index_2[line[1]]], W_2[index_2[line[1]]][index_2[line[0]]])
    (index_3, W_3_test, P_3, Q_3) = group3.calculate_multi(v, W_shared)
    if not (W_3_test is None):
        W_3 = W_3_test
        dec = dec & True
        print("3 suceed")
    else:
        print("3 failed")
        dec = False

    # GROUP 4

    W_shared = {}
    for line in group4.neighbor_lines:
        if line in group5.neighbor_lines:
            W_shared[line] = (W_5[index_5[line[0]]][index_5[line[1]]], W_5[index_5[line[1]]][index_5[line[0]]])
        elif line in group6.neighbor_lines:
            W_shared[line] = (W_6[index_6[line[0]]][index_6[line[1]]], W_6[index_6[line[1]]][index_6[line[0]]])
        elif line in group2.neighbor_lines:
            W_shared[line] = (W_2[index_2[line[0]]][index_2[line[1]]], W_2[index_2[line[1]]][index_2[line[0]]])
    (index_4, W_4_test, P_4, Q_4) = group4.calculate_multi(v, W_shared)
    if not (W_4_test is None):
        W_4 = W_4_test
        dec = dec & True
    else:
        print("4 failed")
        dec = False

    # GROUP 5
    W_shared = {}
    for line in group5.neighbor_lines:
        W_shared[line] = (W_4[index_4[line[0]]][index_4[line[1]]], W_4[index_4[line[1]]][index_4[line[0]]])
    (index_5, W_5_test, P_5, Q_5) = group5.calculate_multi(v, W_shared)
    if not (W_5_test is None):
        W_5 = W_5_test
        dec = dec & True
    else:
        print("5 failed")
        dec = False

    # GROUP 6
    W_shared = {}
    for line in group6.neighbor_lines:
        W_shared[line] = (W_4[index_4[line[0]]][index_4[line[1]]], W_4[index_4[line[1]]][index_4[line[0]]])
    (index_6, W_6_test, P_6, Q_6) = group6.calculate_multi(v, W_shared)
    if not (W_6_test is None):
        W_6 = W_6_test
        dec = dec & True
    else:
        print("6 failed")
        dec = False

    group1.lam = group2.update_lambda(alpha, index_2, W_2, index_1, W_1)
    group3.lam = group2.update_lambda(alpha, index_2, W_2, index_3, W_3)
    group5.lam = group4.update_lambda(alpha, index_4, W_4, index_5, W_5)
    group6.lam = group4.update_lambda(alpha, index_4, W_4, index_6, W_6)
    group4.update_lambda(alpha, index_4, W_4, index_2, W_2)
    group2.update_lambda(alpha, index_2, W_2, index_4, W_4)


active_power_inj = [P_1, P_2, P_3, P_4, P_5, P_6]
reactive_power_inj = [Q_1, Q_2, Q_3, Q_4, Q_5, Q_6]
for group in range(6):
    for node in active_power_inj[group]:
        print(f"node: {node} | active power injection: {active_power_inj[group][node]-load_act[node]}")

for group in range(6):
    for node in reactive_power_inj[group]:
        print(f"node: {node} | reactive power injection: {reactive_power_inj[group][node]-load_react[node]}")
print("time elapsed: ",time.time() - tic)







