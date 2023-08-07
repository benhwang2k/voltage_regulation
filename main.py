import algorithm
import numpy as np
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
nodes = [0, 3, 4, 5, 20, 21,43,44,45,46]
alg = algorithm.Algorithm()
lam = {}
lam[(0,3)] = (0,0)
v = {}
for node in nodes:
    v[node] = 1.0

W_shared = {}
W_shared[(0,3)] = (1.,1.)
alg.calculate_multi(v, lam, W_shared)




