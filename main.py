# import algorithm
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
cap_act = {}
cap_react = {}
network_data = pd.read_excel("UCSDmicrogrid_iCorev3_info.xlsx", sheet_name="Buses_new")
table = network_data.to_numpy()

for row in range(1,49):
    load_act[table[row][1]] = float(table[row][5])
    load_react[table[row][1]] = float(table[row][6])
    if not np.isnan(table[row][11]):
        cap_act[table[row][1]] = float(table[row][12])
        cap_react[table[row][1]] = float(table[row][13])
    if table[row][9]:
        load_act[table[row][1]] -= float(table[row][9])




#setup the groups

group1 = algorithm.Algorithm(nodes=[7,8,31,33,32,47], neighbors=[6], lines= [(6,7),(6,8),(7,31),(7,32),(7,33),(33,47)], neighbor_lines=[(6,7),(6,8)], generator = 47)
W_shared = {}
for line in group1.neighbor_lines:
    group1.set_lambda(line,(0,0))
    W_shared[line] = (0,0)
group1.set_net_load(load_act, load_react, cap_act, cap_react)
group1.calculate_multi(v, W_shared)

group2 = algorithm.Algorithm(nodes=[1,6,9,10,22,23,24], neighbors=[0,7,8,25,26,27], lines=[(0,1),(1,6),(6,7),(6,8),(6,9),(6,10),(10,22),(10,25),(10,26),(10,27),(22,23),(22,24)], neighbor_lines=[(0,1),(6,7),(6,8),(10,25),(10,26),(10,27)], generator=23)
W_shared = {}
for line in group2.neighbor_lines:
    group2.set_lambda(line,(0,0))
    W_shared[line] = (0, 0)
group2.set_net_load(load_act, load_react, cap_act, cap_react)
group2.calculate_multi(v, W_shared)


group3 = algorithm.Algorithm(nodes=[25,26,27,35,36,37,38,39,40], neighbors=[10], lines=[(10,25),(10,26),(10,27),(26,39),(26,40),(27,35),(27,38),(35,36),(35,37)], neighbor_lines=[(10,25),(10,26),(10,27)], generator=37)
W_shared = {}
for line in group3.neighbor_lines:
    group3.set_lambda(line,(0,0))
    W_shared[line] = (0, 0)
group3.set_net_load(load_act, load_react, cap_act, cap_react)
group3.calculate_multi(v, W_shared)


group4 = algorithm.Algorithm(nodes=[0,2,14,15,16,17,18,19,41], neighbors=[1,3,28,42], lines=[(0,1),(0,2),(0,3),(2,41),(14,15),(14,16),(14,17),(14,18),(14,19),(14,41),(28,41),(41,42)], neighbor_lines=[(0,1),(0,3),(28,41),(41,42)], generator=15)
W_shared = {}
for line in group4.neighbor_lines:
    group4.set_lambda(line,(0,0))
    W_shared[line] = (0, 0)
group4.set_net_load(load_act, load_react, cap_act, cap_react)
group4.calculate_multi(v, W_shared)

group5 = algorithm.Algorithm(nodes=[11,12,13,28,29,30,34,42], neighbors=[41], lines=[(11,12),(11,13),(11,28),(28,29),(28,30),(28,41),(29,34),(41,42)], neighbor_lines=[(28,41),(41,42)], generator=13)
W_shared = {}
for line in group5.neighbor_lines:
    group5.set_lambda(line,(0,0))
    W_shared[line] = (0, 0)
group5.set_net_load(load_act, load_react, cap_act, cap_react)
group5.calculate_multi(v, W_shared)

group6 = algorithm.Algorithm(nodes = [3, 4, 5, 20, 21,43,44,45,46],neighbors = [0], lines = [(0,3),(3,43),(4,43),(4,5),(43,44),(20,43),(20,21),(21,45),(21,46)], neighbor_lines = [(0,3)], generator = 45)
group6.set_net_load(load_act, load_react, cap_act, cap_react)
group6.set_lambda((0,3),(0,0))

# calculate things
W_shared = {}
W_shared[(0,3)] = (0,0)
tik = time.time()
group6.calculate_multi(v, W_shared)
print(time.time()-tik)






