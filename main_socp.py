import algorithm_socp
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

group1 = algorithm_socp.Algorithm(nodes=[7,8,31,33,32,47], neighbors=[6], lines= [(6,7),(6,8),(7,31),(7,32),(7,33),(33,47)], neighbor_lines=[(6,7),(6,8)], generator = 47)
group1.set_net_load(load_act, load_react)
group1.calculate_multi()

group2 = algorithm_socp.Algorithm(nodes=[1,6,9,10,22,23,24], neighbors=[0,7,8,25,26,27], lines=[(0,1),(1,6),(6,7),(6,8),(6,9),(6,10),(10,22),(10,25),(10,26),(10,27),(22,23),(22,24)], neighbor_lines=[(0,1),(6,7),(6,8),(10,25),(10,26),(10,27)], generator=23)
group2.set_net_load(load_act, load_react)
group2.calculate_multi()

group3 = algorithm_socp.Algorithm(nodes=[25,26,27,35,36,37,38,39,40], neighbors=[10], lines=[(10,25),(10,26),(10,27),(26,39),(26,40),(27,35),(27,38),(35,36),(35,37)], neighbor_lines=[(10,25),(10,26),(10,27)], generator=37)
group3.set_net_load(load_act, load_react)
group3.calculate_multi()

group4 = algorithm_socp.Algorithm(nodes=[0,2,14,15,16,17,18,19,41], neighbors=[1,3,28,42], lines=[(0,1),(0,2),(0,3),(2,41),(14,15),(14,16),(14,17),(14,18),(14,19),(14,41),(28,41),(41,42)], neighbor_lines=[(0,1),(0,3),(28,41),(41,42)], generator=15)
group4.set_net_load(load_act, load_react)
group4.calculate_multi()

group5 = algorithm_socp.Algorithm(nodes=[11,12,13,28,29,30,34,42], neighbors=[41], lines=[(11,12),(11,13),(11,28),(28,29),(28,30),(28,41),(29,34),(41,42)], neighbor_lines=[(28,41),(41,42)], generator=13)
group5.set_net_load(load_act, load_react)
group5.calculate_multi()

group6 = algorithm_socp.Algorithm(nodes = [3, 4, 5, 20, 21,43,44,45,46],neighbors = [0], lines = [(0,3),(3,43),(4,43),(4,5),(43,44),(20,43),(20,21),(21,45),(21,46)], neighbor_lines = [(0,3)], generator = 45)
group6.set_net_load(load_act, load_react)
group6.calculate_multi()

