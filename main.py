import algorithm
import numpy as np
import time
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

self.nodes = [3, 4, 5, 20, 21,43,44,45,46]
    #     self.neighbors = [0]
    #     self.lines = [(0,3),(3,43),(4,43),(4,5),(43,44),(20,43),(20,21),(21,45),(21,46)]
    #     self.neighbor_lines = [(0,3)]
    #     self.generator = 45

lam = {}
lam[(0,3)] = (0,0)
v = {}
for node in nodes:
    v[node] = 1.
v[0] = 1. - 0.05
v[45] = 1. - 0.05



W_shared = {}
W_shared[(0,3)] = (1.,1.)
tik = time.time()
alg.calculate_multi(v, lam, W_shared)
print(time.time()-tik)



