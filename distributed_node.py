import cvxpy as cp
import numpy as np
import pickle
import pandas as pd
import mosek # this import is unused, but mosek needs to be installed
import pickle

import time
import json
import asyncio
import random
import sys


writer = None
reader = None
node_id = 0
neigh_msg_counter = {}
converge_recieved = False
all_converged = False
msg_q = {}
msg_iteration = {}
rec_iteration = {}
start = False

def log(msg):
    print(f"Node {node_id}: " + str(msg))


async def connect_derp(ip, port):
    '''Connect to the DERP server'''
    global reader, writer
    log(f"Connecting to DERP server at {ip}:{port}")

    connected = False
    while not connected:
        try:
            reader, writer = await asyncio.open_connection(ip, port)
            connected = True
            log("Connection successful")
        except:
            log("Failed to connect to server, retrying in 2s")
            await asyncio.sleep(2)

    await send({"id": node_id})


async def handle_responses():
    '''Parse incoming messages from the DERP server'''

    while True:
        response = (await reader.readline()).decode()

        # If the connection is closed, try to reconnect
        if response == "":
            log("Disconnected from server")
            return

        try:
            msg = json.loads(response)
            dispatch(msg)

        except json.JSONDecodeError:
            log("Invalid response from server: " + msg)


async def converge(msg):
    '''
    Send this message when self has converged.
    Derp records the status of all nodes and returns the
    entire system's status
    '''
    global converge_recieved
    converge_recieved = False
    msg['function'] = 'converge'
    msg['src'] = node_id

    await send(msg)
    # wait for the message back
    while not converge_recieved:
        await asyncio.sleep(0.01)

    return await send(msg)


async def broadcast(msg):
    '''Broadcast the local state to all neighbors.
    Messages contents are specified as keyword arguments.
    Content can be anything that can be serialized to JSON.

    Messages will have the form:
    {
        "function": "broadcast"
        "src": id
        "content": ["state_1", ..., "state_N"],
        "state_1": value_1,
        ...
        "state_N": value_N
    }

    Caller is responsible for waiting for response.

    Returns True for success, False for failure.
    '''
    msg['contents'] = ','.join(msg.keys())
    for state, val in msg.items():
        msg[state] = val

    msg['function'] = 'broadcast'
    msg['src'] = node_id

    return await send(msg)

async def send(msg):
    '''Send a JSON message to the DERP over `writer`

    Returns True for success, False for failure.
    '''

    bmsg = (json.dumps(msg) + '\n').encode()

    writer.write(bmsg)

    try:
        await writer.drain()
        return True

    # This happens when the connection is broken
    except ConnectionResetError:
        log("Disconnected from DERP")
        return False


def dispatch(msg):
    global start
    '''Algorithm specific behavior in response to messages'''
    if msg['function'] == 'broadcast':
        get_broadcast(msg)
    elif msg['function'] == 'modbus_read':
        handle_modbus(msg)
    elif msg['function'] == 'converge':
        get_converge(msg)
    elif msg['function'] == 'set_loads':
        set_loads(msg)
    elif msg['function'] == 'start':
        start = True
    elif msg['function'] == 'ack':
        pass
    else:
        raise NotImplementedError()


def handle_modbus(msg):
    log("Modbus: " + str(msg))




DERP_IP = 'localhost'
DERP_PORT = 10000



DEBUG = False

n = 48
buses = list(range(n))

# Nodes defining the groups
bus_groups = [
    [7, 31, 33, 32, 47],
    [1, 6, 8, 9, 10, 22, 23, 24, 25, 26, 39, 40],
    [27, 35, 36, 37, 38],
    [0, 2, 14, 15, 16, 17, 18, 19, 41, 42],
    [11, 12, 13, 28, 29, 30, 34],
    [3, 4, 5, 20, 21, 43, 44, 45, 46],
]

N = len(bus_groups)

generators = [47, 23, 37, 15, 13, 45]

# Load the microgrid data
# Scale everything to units of MW (from KW on the sheet)
scale = 1000.0
loads = pd.read_excel("UCSDmicrogrid_iCorev3_info.xlsx",
                      sheet_name="Buses_new").to_numpy()


P = [0.]*n
Q = [0.]*n
for i in buses:
    P[loads[i+1][1]] = float(loads[i+1][5])/scale
    Q[loads[i+1][1]] = float(loads[i+1][6])/scale
    if not np.isnan(loads[i+1][9]):
        P[loads[i+1][1]] -= float(loads[i+1][9])/scale

P = [0.0, 0.0, 0.0, 0.0, 0.5456991, 0.8934548999999999, 0.4053246, 4.0337757000000005, 0.3709854000000001, 2.20467, 1.3755114000000002, 0.016491552, 0.0, 0.0, 0.019836216, 0.03126764, 0.0036618460000000003, 0.01630485, 0.00394044, 0.006545634000000001, 0.394044, 0.0, 0.4783635, 0.3877464, 0.5786881, 1.284752, 0.2581329, 0.0, 0.040866753, 0.018872922, 0.021672420000000005, 1.6243005000000001, 1.576176, 1.199182, 0.011295246000000002, 0.295533, 0.197022, 2.0987310000000003, 0.848088, 0.31702199999999997, 0.755199, 0.0, 0.0200022, 0.0, 0.878088, 0.788088, 0.525066, 0.0]
Q = [0.0, 0.0, 0.0, 0.0, 0.40927440000000004, 0.6700910999999999, 0.3039264, 2.9246367, 0.5024061, 1.500165, 1.0316334, 0.012368672999999998, 0.0, 0.0, 0.014787042, 0.02345073, 0.003786606, 0.013923630000000001, 0.00295533, 0.003930924000000001, 0.295533, 0.0, 0.3587727, 0.2908098, 0.680316, 0.963564, 0.1935996, 0.0, 0.030542064, 0.015078693, 0.01625432, 1.2128253, 1.182132, 0.886599, 0.007574913, 0.2216496, 0.14776499999999998, 1.574048, 0.636066, 0.23776650000000002, 0.561, 0.0, 0.01500165, 0.0, 0.658566, 0.591066, 0.4432955, 0.0]


network = pd.read_excel("UCSDmicrogrid_iCorev3_info.xlsx",
                        sheet_name="Branches_new").to_numpy()
lines = [(int(network[i][0]), int(network[i][1]))
         for i in range(len(network))]

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


alpha_P = 2e-5
alpha_Q = 2e-5
alpha_I = 2e-11
alpha_V = 2e-5


class Algorithm():
    """
    """

    def __init__(self, group, buses):
        self.group = group
        self.buses = buses
        self.lines = [line for line in lines
                      if line[0] in self.buses or line[1] in self.buses]
        self.neighbors = []
        self.neighbor_lines = []
        for line in lines:
            if line[0] in self.buses and line[1] not in self.buses:
                self.neighbor_lines += [line]
                self.neighbors += [line[1]]
            elif line[1] in self.buses and line[0] not in self.buses:
                self.neighbor_lines += [line]
                self.neighbors += [line[0]]

        # dict of loads
        self.P = {}
        self.Q = {}

        # Decision variables are stored here for easy access
        # before and after solving the model

        # Voltage squared
        self.V = [1.]*48

        # Current squared
        self.I = {}

        # Power flow on lines
        self.Pij = {}
        self.Qij = {}

        # Power generation
        self.p_gen = [0.]*48
        self.q_gen = [0.]*48

        # Lagrangian multipliers
        self.lam_P = {}
        self.lam_Q = {}
        self.lam_I = {}
        self.lam_V = {}

        # Shared (coupled) state
        self.lam_P_shr = {}
        self.lam_Q_shr = {}
        self.lam_I_shr = {}
        self.lam_V_shr = {}

        if DEBUG:
            print(
                f"Initialized group {self.group} with\n",
                f"Buses: {self.buses}\n",
                f"Lines: {self.lines}\n",
                f"Neighbors: {self.neighbors}\n",
                f"Neighbor lines: {self.neighbor_lines}\n",
            )

    def build(self):
        constraints = []
        # for i in buses:
        #     self.P[loads[i + 1][1]] = cp.Parameter(value=float(loads[i + 1][5]) / scale)
        #     self.Q[loads[i + 1][1]] = cp.Parameter(value=float(loads[i + 1][6]) / scale)
        #     if not np.isnan(loads[i + 1][9]):
        #         self.P[loads[i + 1][1]].value -= float(loads[i + 1][9]) / scale

        P_nom = [0.0, 0.0, 0.0, 0.0, 0.5456991, 0.8934548999999999, 0.4053246, 4.0337757000000005, 0.3709854000000001,
             2.20467, 1.3755114000000002, 0.016491552, 0.0, 0.0, 0.019836216, 0.03126764, 0.0036618460000000003,
             0.01630485, 0.00394044, 0.006545634000000001, 0.394044, 0.0, 0.4783635, 0.3877464, 0.5786881, 1.284752,
             0.2581329, 0.0, 0.040866753, 0.018872922, 0.021672420000000005, 1.6243005000000001, 1.576176, 1.199182,
             0.011295246000000002, 0.295533, 0.197022, 2.0987310000000003, 0.848088, 0.31702199999999997, 0.755199, 0.0,
             0.0200022, 0.0, 0.878088, 0.788088, 0.525066, 0.0]
        Q_nom = [0.0, 0.0, 0.0, 0.0, 0.40927440000000004, 0.6700910999999999, 0.3039264, 2.9246367, 0.5024061, 1.500165,
             1.0316334, 0.012368672999999998, 0.0, 0.0, 0.014787042, 0.02345073, 0.003786606, 0.013923630000000001,
             0.00295533, 0.003930924000000001, 0.295533, 0.0, 0.3587727, 0.2908098, 0.680316, 0.963564, 0.1935996, 0.0,
             0.030542064, 0.015078693, 0.01625432, 1.2128253, 1.182132, 0.886599, 0.007574913, 0.2216496,
             0.14776499999999998, 1.574048, 0.636066, 0.23776650000000002, 0.561, 0.0, 0.01500165, 0.0, 0.658566,
             0.591066, 0.4432955, 0.0]
        for i in range(48):
            self.P[i] = cp.Parameter(value=P_nom[i])
            self.Q[i] = cp.Parameter(value=Q_nom[i])

        for i in self.buses + self.neighbors:
            if i not in generators:
                self.V[i] = cp.Variable()
                constraints += [self.V[i] >= 0.91]
                constraints += [self.V[i] <= 1.1]
            else:
                self.p_gen[i] = cp.Variable()
                self.q_gen[i] = cp.Variable()
                if i != 13:
                    constraints += [self.p_gen[i] <= 2000]
                    constraints += [self.q_gen[i] <= 2000]
                    constraints += [self.p_gen[i] >= -2000]
                    constraints += [self.q_gen[i] >= -2000]
                else:
                    constraints += [self.p_gen[i] <= 0.25]
                    constraints += [self.q_gen[i] <= 0.25]
                    constraints += [self.p_gen[i] >= -0.25]
                    constraints += [self.q_gen[i] >= -0.25]


        for line in self.lines:
            self.I[line] = cp.Variable()
            self.Pij[line] = cp.Variable()
            self.Qij[line] = cp.Variable()

        for i in self.buses:
            Pij_out = 0
            Qij_out = 0
            Pij_in = 0
            Qij_in = 0


            for line in self.lines:
                if i == line[0]:
                    Pij_out += self.Pij[line]
                    Qij_out += self.Qij[line]
                elif i == line[1]:
                    Pij_in += self.Pij[line] - r[line[0]][line[1]] * self.I[line]
                    Qij_in += self.Qij[line] - x[line[0]][line[1]] * self.I[line]

            # Active and reactive power balance
            # The terms g_i*V and b_i*V are zero in this case
            constraints += [-self.P[i] + self.p_gen[i] == Pij_out - Pij_in]
            constraints += [-self.Q[i] + self.q_gen[i] == Qij_out - Qij_in]

        self.z = {}
        for line in self.lines:
            r_l = r[line[0]][line[1]]
            x_l = x[line[0]][line[1]]

            constraints += [
                self.V[line[1]] == self.V[line[0]]
                - 2 * (r_l * self.Pij[line] + x_l * self.Qij[line])
                + (r_l*r_l + x_l*x_l) * self.I[line]
            ]

            # Slack variable for constraint VI <= P^2 + Q^2
            self.z[line] = cp.Variable(1)

            # V^.5 * I^.5 <= |z|
            constraints += [cp.PowCone3D(
                cp.vstack([self.V[line[0]]])[0],
                cp.vstack([self.I[line]])[0],
                self.z[line],
                0.5
            )]

            # ||[P;Q]||_2 <= z
            constraints += [cp.SOC(
                self.z[line],
                cp.vstack([self.Pij[line], self.Qij[line]])
            )]

        f_obj = sum(self.p_gen)

        for line in self.neighbor_lines:
            self.lam_P[line] = cp.Parameter(value=0)
            self.lam_Q[line] = cp.Parameter(value=0)
            self.lam_I[line] = cp.Parameter(value=0)
            self.lam_V[line[0]] = cp.Parameter(value=0)
            self.lam_V[line[1]] = cp.Parameter(value=0)
            self.lam_P_shr[line] = cp.Parameter(value=0)
            self.lam_Q_shr[line] = cp.Parameter(value=0)
            self.lam_I_shr[line] = cp.Parameter(value=0)
            self.lam_V_shr[line[0]] = cp.Parameter(value=0)
            self.lam_V_shr[line[1]] = cp.Parameter(value=0)
            lambdas = [
                self.lam_P[line] * self.Pij[line] - self.lam_P_shr[line],
                self.lam_Q[line] * self.Qij[line] - self.lam_Q_shr[line],
                self.lam_I[line] * self.I[line] - self.lam_I_shr[line],
                self.lam_V[line[0]] * self.V[line[0]] - self.lam_V_shr[line[0]],
                self.lam_V[line[1]] * self.V[line[1]] - self.lam_V_shr[line[1]],
            ]

            f_obj += sum(lambdas)

        self.prob = cp.Problem(cp.Minimize(f_obj), constraints)

    def update_lambdas(self, neighbors, update_vals=False):
        for line in self.neighbor_lines:
            for neighbor in neighbors:
                if line not in neighbor.lines:
                    continue

                # The lam_X_shr parameters hold the value of
                # lam_X*X_shr.  This is done to comply with the
                # disciplined parameterized program (DPP) which is
                # required for CVX to cache the program structure
                # between solves.
                self.lam_P[line].value += alpha_P * (self.Pij[line].value - neighbor.Pij[line].value)
                self.lam_Q[line].value += alpha_Q * (self.Qij[line].value - neighbor.Qij[line].value)
                self.lam_I[line].value += alpha_I * (self.I[line].value - neighbor.I[line].value)
                self.lam_V[line[0]].value += alpha_V * (self.V[line[0]].value - neighbor.V[line[1]].value)
                self.lam_V[line[1]].value += alpha_V * (self.V[line[1]].value - neighbor.V[line[0]].value)

                if update_vals:
                    self.lam_P_shr[line].value = self.lam_P[line].value*neighbor.Pij[line].value
                    self.lam_Q_shr[line].value = self.lam_Q[line].value*neighbor.Qij[line].value
                    self.lam_I_shr[line].value = self.lam_I[line].value*neighbor.I[line].value
                    self.lam_V_shr[line[0]].value = self.lam_V[line[0]].value*neighbor.V[line[0]].value
                    self.lam_V_shr[line[1]].value = self.lam_V[line[1]].value*neighbor.V[line[1]].value

    def solve(self):
        self.prob.solve(
            solver=cp.MOSEK,
            verbose=False,
            mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-6}
        )





def test_centralized():
    group_all = Algorithm(0, buses)

    group_all.build()

    group_all.solve()

    print(f"Total generation: {group_all.prob.objective.value}")

    return group_all


group = [None]*N
warm_group = [None]*N
for i in range(N):
    group[i] = pickle.load(open(f'bin/group{i}_other.pickle', 'rb'))
    # print(f"pickle value: {group[i].p_gen[generators[i]].value[()]}")
    # group[i] = Algorithm(i, bus_groups[i])
    # group[i].build()

print(group[0])

def set_loads(msg):
    global group
    ''' set the loads of the groups to the new ones'''
    #pickle the nodes
    # file_name = f"bin/group{node_id}_new.pickle"
    # with open(file_name, 'wb') as filehandler:
    #     pickle.dump(group[node_id], filehandler)
    # pickle.dump(group[node_id], open(f"bin/group{node_id}_other.pickle", 'wb'))
    # print(f"pgen_group{node_id} is {group[node_id].p_gen[generators[node_id]]}")
    print(f"group {node_id} recieved new loads")
    ploads = msg['ploads']
    qloads = msg['qloads']
    for g in range(N):
        if g != 4:
            for key in group[g].P:
                group[g].P[key].value = ploads[str(key)]
                group[g].Q[key].value = qloads[str(key)]



def get_broadcast(msg):
    global neigh_msg_counter, msg_q, msg_iteration
    '''update self's version of shared variables sent by src'''
    msg_q[msg['src']].append(msg)
    # if msg_iteration[msg['src']] >= msg['t']:
    #     # this message is early
    #     msg_q[msg['src']].append(msg)
    # else:
    #     # process the next message
    #     process_msg(msg)

def process_msg(msg):
    global group, neigh_msg_counter, msg_iteration, rec_iteration
    neigh_msg_counter[msg['src']] -= 1
    rec_iteration[msg['src']] = msg['t']
    for line in group[node_id].neighbor_lines:
        if 'Pij' + str(line) in msg:
            #warm_diff = warm_group[msg['src']].Pij[line].value - msg['Pij' + str(line)]
            #if abs(warm_group[msg['src']].I[line].value - msg['I' + str(line)]) < 1e-2:
            #    print(f"big diff in line {line} of group {msg['src']} as calculated by {node_id} = {warm_group[msg['src']].I[line].value - msg['I' + str(line)]}")
            group[msg['src']].Pij[line].value = msg['Pij' + str(line)]
            group[msg['src']].Qij[line].value = msg['Qij' + str(line)]
            group[msg['src']].I[line].value = msg['I' + str(line)]
            group[msg['src']].V[line[0]].value = msg['V' + str(line[0])]
            group[msg['src']].V[line[1]].value = msg['V' + str(line[1])]

def get_converge(msg):
    global converge_recieved, all_converged

    '''update the status of the whole system as converged or not converged'''
    all_converged = msg['status']
    converge_recieved = True

async def send_params(t):
    msg = {}
    for line in group[node_id].neighbor_lines:
        msg['t'] = t
        msg['Pij' + str(line)] = group[node_id].Pij[line].value[()]
        msg['Qij' + str(line)] = group[node_id].Qij[line].value[()]
        msg['I' + str(line)] = group[node_id].I[line].value[()]
        msg['V' + str(line[0])] = group[node_id].V[line[0]].value[()]
        msg['V' + str(line[1])] = group[node_id].V[line[1]].value[()]
    # send the message
    if not await broadcast(msg):
        await connect_derp(DERP_IP, DERP_PORT)


async def main():
    global reader, writer, all_converged, neigh_msg_counter, msg_q, msg_iteration, rec_iteration, start

    if node_id == 0:
        neigh_msg_counter[1] = 0
        msg_q[1] = []
    elif node_id == 1:
        neigh_msg_counter[0] = 0
        neigh_msg_counter[2] = 0
        neigh_msg_counter[3] = 0
        msg_q[0] = []
        msg_q[2] = []
        msg_q[3] = []
    elif node_id == 2:
        neigh_msg_counter[1] = 0
        msg_q[1] = []
    elif node_id == 3:
        neigh_msg_counter[1] = 0
        neigh_msg_counter[4] = 0
        neigh_msg_counter[5] = 0
        msg_q[1] = []
        msg_q[4] = []
        msg_q[5] = []
    elif node_id == 4:
        neigh_msg_counter[3] = 0
        msg_q[3] = []
    elif node_id == 5:
        neigh_msg_counter[3] = 0
        msg_q[3] = []
    for key in neigh_msg_counter:
        msg_iteration[key] = 0
        rec_iteration[key] = -1

    await connect_derp(DERP_IP, DERP_PORT)

    asyncio.create_task(handle_responses())


    while not start:
        await asyncio.sleep(1)

    while True:
        # read loads

        # solve
        s1 = 0
        s2 = 0
        t = 0
        all_converged = False
        while True:
            # read messages from all neighbors
            # keep counter for each node (+1 when you send a message) (-1 when you recieve)
            # if all counters <= 0 then go to next iteration
            s2 = s1
            group[node_id].solve()


            for neighbor in neigh_msg_counter:
                neigh_msg_counter[neighbor] += 1
            await send_params(t)

            # wait for messages
            can_update = False
            while not can_update:
                can_update = True
                for neigh in msg_iteration:
                    msg_iteration[neigh] = t
                    if msg_iteration[neigh] > rec_iteration[neigh] and len(msg_q[neigh]) > 0:
                        process_msg(msg_q[neigh].pop(0))
                    can_update = can_update and (msg_iteration[neigh] == rec_iteration[neigh])
                if not can_update:
                    await asyncio.sleep(0.05)
            # now we have processed the same number of messages as we have sent

            # update lams
            update_vals = (t % 100 == 0)
            group[node_id].update_lambdas(group, update_vals)
            s1 = group[node_id].prob.objective.value
            # send diff

            conv_msg = {}
            conv_msg['diff'] = (s2-s1) ** 2
            conv_msg['pgen'] = group[node_id].prob.objective.value
            conv_msg['qgen'] = group[node_id].q_gen[generators[node_id]].value[()]
            conv_msg['gen'] = generators[node_id]
            for b in group[node_id].buses:
                if not (b in generators):
                    conv_msg[str(b)] = group[node_id].V[b].value ** 0.5
                else:
                    conv_msg[str(b)] = 1.0
            if not await converge(conv_msg):
                await connect_derp(DERP_IP, DERP_PORT)
            t = t + 1
            print(f"t: {t}")





if __name__ == "__main__":

    node_id = int(sys.argv[1])
    asyncio.run(main())
