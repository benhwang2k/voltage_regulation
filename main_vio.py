import algorithm_vio
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

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

group1 = algorithm_vio.Algorithm(group=1, nodes=[7,31,33,32,47], neighbors=[6], lines= [(6,7),(7,31),(7,32),(7,33),(33,47)], neighbor_lines=[(6,7)], generator = 47)
group1.set_net_load(load_act, load_react)
(shared_active_group1, shared_reactive_group1, shared_current_group1, shared_volt_group1, objective1) = group1.calculate_multi()

group2 = algorithm_vio.Algorithm(group=2, nodes=[1,6,8,9,10,22,23,24,25,26,39,40], neighbors=[0,7,27], lines=[(0,1),(1,6),(6,7),(6,8),(6,9),(6,10),(10,22),(10,25),(10,26),(10,27),(22,23),(22,24),(26,39),(26,40)], neighbor_lines=[(0,1),(6,7),(10,27)], generator=23)
group2.set_net_load(load_act, load_react)
(shared_active_group2, shared_reactive_group2, shared_current_group2, shared_volt_group2, objective2) = group2.calculate_multi()

group3 = algorithm_vio.Algorithm(group=3, nodes=[27,35,36,37,38], neighbors=[10], lines=[(10,27),(27,35),(27,38),(35,36),(35,37)], neighbor_lines=[(10,27)], generator=37)
group3.set_net_load(load_act, load_react)
(shared_active_group3, shared_reactive_group3, shared_current_group3, shared_volt_group3, objective3) = group3.calculate_multi()

group4 = algorithm_vio.Algorithm(group=4, nodes=[0,2,14,15,16,17,18,19,41,42], neighbors=[1,3,28], lines=[(0,1),(0,2),(0,3),(2,41),(14,15),(14,16),(14,17),(14,18),(14,19),(14,41),(28,41),(41,42)], neighbor_lines=[(0,1),(0,3),(28,41)], generator=15)
group4.set_net_load(load_act, load_react)
(shared_active_group4, shared_reactive_group4, shared_current_group4, shared_volt_group4, objective4) = group4.calculate_multi()

group5 = algorithm_vio.Algorithm(group=5, nodes=[11,12,13,28,29,30,34], neighbors=[41], lines=[(11,12),(11,13),(11,28),(28,29),(28,30),(28,41),(29,34)], neighbor_lines=[(28,41)], generator=13)
group5.set_net_load(load_act, load_react)
(shared_active_group5, shared_reactive_group5, shared_current_group5, shared_volt_group5, objective5) = group5.calculate_multi()

group6 = algorithm_vio.Algorithm(group=6, nodes = [3, 4, 5, 20, 21,43,44,45,46],neighbors = [0], lines = [(0,3),(3,43),(4,43),(4,5),(43,44),(20,43),(20,21),(21,45),(21,46)], neighbor_lines = [(0,3)], generator = 45)
group6.set_net_load(load_act, load_react)
(shared_active_group6, shared_reactive_group6, shared_current_group6, shared_volt_group6, objective6) = group6.calculate_multi()

# update parameters
# share all results
t = 0
tot = 2000;
alpha_power = 0.03
alpha_current = 5.0E-8
alpha_voltage = 50.0
# graph1 = [objective1]
# graph2 = [objective2]
graph1 = []
graph_active = []
graph_reactive = []
graph_current = []
graph_voltage = []
graph_lam_current2 = []
graph_lam_active2 = []
graph_lam_reactive2 = []
graph_lam_voltage2 = []
while t < 60:
    t += 1
    print(f"\033[0;32miteration: {t}\033[0m")
    # share the active and reactive power flow in the lines
    group1.update_shared_powers(shared_active_group2, shared_reactive_group2, shared_current_group2, shared_volt_group2)
    group3.update_shared_powers(shared_active_group2, shared_reactive_group2, shared_current_group2, shared_volt_group2)
    group5.update_shared_powers(shared_active_group4, shared_reactive_group4, shared_current_group4, shared_volt_group4)
    group6.update_shared_powers(shared_active_group4, shared_reactive_group4, shared_current_group4, shared_volt_group4)
    group2.update_shared_powers(shared_active_group1, shared_reactive_group1, shared_current_group1, shared_volt_group1)
    group2.update_shared_powers(shared_active_group3, shared_reactive_group3, shared_current_group3, shared_volt_group3)
    group2.update_shared_powers(shared_active_group4, shared_reactive_group4, shared_current_group4, shared_volt_group4)
    group4.update_shared_powers(shared_active_group2, shared_reactive_group2, shared_current_group2, shared_volt_group2)
    group4.update_shared_powers(shared_active_group5, shared_reactive_group5, shared_current_group5, shared_volt_group5)
    group4.update_shared_powers(shared_active_group6, shared_reactive_group6, shared_current_group6, shared_volt_group6)
    # controllers at 2 and 4 update  lambdas
    tot = 0
    (subtot_p2, subtot_q2, subtot_i2,subtot_v2) = group2.update_lam(alpha_power,alpha_current, alpha_voltage)
    (subtot_p4, subtot_q4, subtot_i4,subtot_v4) = group4.update_lam(alpha_power,alpha_current, alpha_voltage)
    graph_active.append(subtot_p2 + subtot_p4)
    graph_reactive.append(subtot_q2 + subtot_q4)
    graph_current.append(subtot_i2 + subtot_i4)
    graph_voltage.append(subtot_v2 + subtot_v4)
    tot += subtot_p2 + subtot_q2 + subtot_i2 + subtot_v2 + subtot_p4 + subtot_q4 + subtot_i4 + subtot_v4
    print("----------------")
    print("1 norm of differences: ", tot)
    print("----------------")
    # share the lambdas to other controllers
    group1.set_lam(group2.lam_active, group2.lam_reactive, group2.lam_current, group2.lam_volt)
    group3.set_lam(group2.lam_active, group2.lam_reactive, group2.lam_current, group2.lam_volt)
    group5.set_lam(group4.lam_active, group4.lam_reactive, group4.lam_current, group4.lam_volt)
    group6.set_lam(group4.lam_active, group4.lam_reactive, group4.lam_current, group4.lam_volt)
    # group1.print_params()
    # group2.print_params()
    # group3.print_params()
    # group4.print_params()
    # group5.print_params()
    # group6.print_params()

    # recalculate
    (shared_active_group1, shared_reactive_group1, shared_current_group1, shared_volt_group1, objective1) = group1.calculate_multi()
    (shared_active_group2, shared_reactive_group2, shared_current_group2, shared_volt_group2, objective2) = group2.calculate_multi()
    (shared_active_group3, shared_reactive_group3, shared_current_group3, shared_volt_group3, objective3) = group3.calculate_multi()
    (shared_active_group4, shared_reactive_group4, shared_current_group4, shared_volt_group4, objective4) = group4.calculate_multi()
    (shared_active_group5, shared_reactive_group5, shared_current_group5, shared_volt_group5, objective5) = group5.calculate_multi()
    (shared_active_group6, shared_reactive_group6, shared_current_group6, shared_volt_group6, objective6) = group6.calculate_multi()
    # graph1.append(objective1)
    # graph2.append(objective2)
    graph1.append(tot)
    graph_lam_current2.append(group2.lam_current[(0,1)])
    graph_lam_active2.append(group2.lam_active[(0, 1)])
    graph_lam_reactive2.append(group2.lam_reactive[(0, 1)])
    graph_lam_voltage2.append(group2.lam_volt[0])
# plt.plot(graph1)
# plt.plot(graph2)
plt.plot(graph1)
plt.plot(graph_current)
plt.title("difference")
plt.xlabel("iteration #")
plt.legend(["total", "current"])
plt.savefig('graph_new.png')
plt.figure()
plt.plot(graph_active)
plt.plot(graph_reactive)
plt.plot(graph_voltage)
plt.title("differences ")
plt.xlabel("iteration #")
plt.legend(["active", "reactive", "voltage"])
plt.savefig('graph_smalldifs.png')
plt.figure()
plt.plot(graph_lam_current2)
plt.plot(graph_lam_active2)
plt.plot(graph_lam_reactive2)
plt.plot(graph_lam_voltage2)
plt.title("lambda of line 0,1")
plt.xlabel("iteration #")
plt.legend(["current", "active", "reactive", "voltage0"])
plt.savefig('graph_lam.png')

