import asyncio
import json
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from pyModbusTCP.client import ModbusClient
from pyModbusTCP import utils
import random
import pymodbus
import socket
import numpy as np
import pandas as pd
import datetime as dtt
import pytz
from pymodbus.client import ModbusTcpClient

writers = {}

neighbors = [
    [1],         # 0 is neighbors with 1
    [0, 2, 3],   # 1 is neighbors with 0 , 2, 3
    [1],         # 2 is neighbors with 1
    [1, 4, 5],   # 3 is neighbors with 1,4,5
    [3],         # 4 with 3
    [3]         # 5 with 3
]

nodes_diff = [1.0]*6
nodes_converged = [0]*6
p_gen = {}
q_gen = {}
voltage = {}



def connect_to_MySQL():
    load_dotenv()
    return create_engine("mysql://{username}:{password}@{host_name}:{host_port}".format(
        username=os.environ['database_username'],
        password=os.environ['database_password'],
        host_name="minsait-dev.sdsc.edu",
        host_port="3306",)
    )


# Convert single precision floats from 2 registers
def convert_float(a,b):
	h = (a<<16| b)
	sign = h >> 31
	exp = ((h & 0x7f800000) >> 23) - 127
	mant = (h | 0x800000) & 0x00ffffff
	num = mant/(pow(2,23-exp))
	return round(num, 2)


def read_meters_data(dbconn, meter1, meter2, meter3, data_datetime = dtt.datetime.now().replace(microsecond=0)):
    activeA1 = meter1.read_holding_registers(3054-1, 2)
    reactiveA1 = meter1.read_holding_registers(3062-1, 2)
    activeA2 = meter2.read_holding_registers(3054-1, 2)    # active power in W
    reactiveA2 = meter2.read_holding_registers(3062-1, 2)
    activeA3 = meter3.read_holding_registers(197, 2)    # active power in W
    reactiveA3 = meter3.read_holding_registers(207, 2)
    activeA1 = convert_float(activeA1[0], activeA1[1])
    reactiveA1 = convert_float(reactiveA1[0], reactiveA1[1])
    activeA2 = convert_float(activeA2[0], activeA2[1])
    reactiveA2 = convert_float(reactiveA2[0], reactiveA2[1])
    activeA3 = round(((activeA3[0] << 16) | activeA3[1]) * 1000, 2)
    reactiveA3 = round(((reactiveA3[0] << 16) | reactiveA3[1]) * 1000, 2)
    data = (activeA3, reactiveA3, activeA2, reactiveA2, activeA1, reactiveA1)
    df = pd.DataFrame([data], columns=['Pa1', 'Qa1', 'Pa2', 'Qa2', 'Pa3', 'Qa3'])
    df['data_datetime'] = data_datetime
    # df.to_sql("meters", dbconn, "icore", "append", False)
    return df


def read_voltages(dbconn, scada_client, data_datetime = dtt.datetime.now().replace(microsecond=0)):
    matrix = [0, round(0.0001 * (scada_client.read_input_registers(address=0, count=1)).registers[0], 3)]
    for i in range(1,47):
        matrix = np.vstack([matrix, [i, round(0.0001 * (scada_client.read_input_registers(address=i, count=1)).registers[0], 3)]])
    df = pd.DataFrame(matrix, columns=['bus', 'voltage'])
    df.bus = df.bus.astype('int')
    df['data_datetime'] = data_datetime
    # df.to_sql("voltages", dbconn, "icore", "append", False)
    return df


def store_voltages_random(dbconn, data_datetime = dtt.datetime.now().replace(microsecond=0)):
    matrix = [0, round(random.uniform(0.6, 1.5), 3)]
    for i in range(1,47):
        matrix = np.vstack([matrix, [i, round(random.uniform(0.6, 1.5), 3)]])

    df = pd.DataFrame(matrix, columns=['bus', 'voltage'])
    df.bus = df.bus.astype('int')
    df['data_datetime'] = data_datetime
    # df.to_sql("voltages", dbconn, "icore", "append", False)
    return df


async def main():
    # run the server so that the nodes can connect
    asyncio.create_task(run_server('', 10000))


    #interact with database:
    dbconn = connect_db.connect_to_MySQL()
    IPAddr=socket.gethostbyname(socket.gethostname())
    df = pd.DataFrame([('derp', IPAddr, 10000, 'iot')], columns=['name', 'ip', 'port', 'type'])
    df.to_sql("tcp_addresses", dbconn, "icore", "append", False)

    geisel = ModbusClient(host="172.31.213.2", port=502, unit_id=1, auto_open=True, timeout=2.0)
    rady = ModbusClient(host="172.31.212.150", port=502, unit_id=1, auto_open=True)
    mcgill = ModbusClient(host="172.31.212.151", port=502, unit_id=2, auto_open=True)

    scada_ip_query = text("SELECT ip, port from icore.tcp_addresses WHERE name = 'scada' AND type = 'rtds'")
    scada_ip, scada_port = dbconn.connect().execute(scada_ip_query).fetchall()[0]
    scada_client = ModbusTcpClient(scada_ip, scada_port)
    scada_client.connect()

    for i in range(10):
        control_query = text("SELECT voltage_display, algorithm from icore.controls")
        voltage_display, algorithm = dbconn.connect().execute(control_query).fetchall()[0]
        try:
            data_datetime = dtt.datetime.now(pytz.timezone('US/Pacific')).replace(microsecond=0)
            df_meter_reading = read_meters_data(
                dbconn = dbconn,
                data_datetime = data_datetime,
                meter1 = geisel,
                meter2 = rady,
                meter3 = mcgill,
            )
            # algorith == 1  -> centralized
            if algorithm == 1:
                pass
                # solve centralized SOCP
                # send meter data to RTDS
                # send OPT setpoint to RTDS
                df_meter_reading.to_sql("meters", dbconn, "icore", "append", False) # store meter data to DB
                if voltage_display == 'rtds':
                    df_voltages = read_voltages(dbconn = dbconn, data_datetime = data_datetime)
                    df_voltages.to_sql("voltages", dbconn, "icore", "append", False)
                else:
                    df_voltages = store_voltages_random(dbconn = dbconn, data_datetime = data_datetime)
                    df_voltages.to_sql("voltages", dbconn, "icore", "append", False)
            # algorith == 2  -> decentralized
            elif algorithm == 2:
                pass
                # send the current load data for the nodes to solve
                await reset_loads(new_P=new_P, new_Q=new_Q) # new loads should be arrays of length 47 with the P, Q load associated with the bus at the index of the array
                # wait for solution
                while not (all(nodes_converged)):
                    await asyncio.sleep(0.1)
                # read the solution
                setpoints = [(0,0)]*6
                sol_voltages = [0.]*47
                for g in range(6):
                    setpoints[g] = (p_gen[g], q_gen[g])
                for bus in range(48):
                    sol_voltages[bus] = voltage[bus]
                sol = {'setpoints': p_gen, 'voltages': sol_voltage}
                # send power setpoints to RTDS
                
                if voltage_display == 'rtds':
                    read_voltages(dbconn = dbconn, data_datetime = data_datetime)
                else:
                    store_voltages_random(dbconn = dbconn, data_datetime = data_datetime)
            # algorith == other  -> nothing
            else:
                if voltage_display == 'rtds':
                    read_voltages(dbconn = dbconn, data_datetime = data_datetime)
                else:
                    store_voltages_random(dbconn = dbconn, data_datetime = data_datetime)

        except:
            pass
        # time.sleep(1)


def log(msg):
    print("DERP: " + str(msg))


async def run_server(ip, port):
    '''Create and run the DERP server'''
    log(f"Starting DERP server on {ip}:{port}")

    server = await asyncio.start_server(handle_client, ip, port)

    await server.serve_forever()


async def handle_client(reader, writer):
    '''Callback function for new connections to server'''

    # Node ID is always the first message sent
    try:
        data = json.loads((await reader.readline()).decode())
        node_id = data['id']
    except Exception:
        log("Invalid connection")
        return

    log(f"Accepted new connection from node {node_id}")

    # Store the writer for later use
    writers[node_id] = writer

    # Send the start message:
    has_all_clients = True
    for i in range(6):
        has_all_clients = has_all_clients and (i in writers)
    if has_all_clients:
        for i in range(6):
            await send(i, {'function': 'start'})

    # Serve client requests forever
    while True:
        request = (await reader.readline()).decode()

        # Only happens when connection is broken
        if request == "":
            log(f"Disconnected from node {node_id}")
            writers[node_id] = None
            break

        try:
            msg = json.loads(request)
        except json.JSONDecodeError:
            log(f"Invalid request from node {node_id}")
            continue

        response = await dispatch(msg)

        if not await send(node_id, response):
            break


async def send(node_id, msg):
    '''Send a JSON message to `node_id`'''
    writer = writers.get(node_id)

    if not writer:
        log(f"Not connected to node {node_id}")
        return False

    bmsg = (json.dumps(msg) + '\n').encode()

    writer.write(bmsg)

    try:
        await writer.drain()
        return True

    # This happens when the connection is broken
    except ConnectionResetError:
        log(f"Disconnected from node {node_id}")
        writers[node_id] = None
        return False


async def dispatch(msg):
    '''Respond to the contents of the client request'''

    if msg['function'] == 'write_modbus':
        response = write_modbus(msg)

    elif msg['function'] == 'read_modbus':
        response = read_modbus(msg)

    elif msg['function'] == 'broadcast':
        response = await broadcast(msg)

    elif msg['function'] == 'converge':
        response = await converged(msg)
    else:
        raise NotImplementedError()

    return response


def write_modbus(msg):
    '''Write data to modbus outstation'''
    return {}

[P[i] for i in [4,5,20,44,45,46]]
def read_modbus(msg):
    '''Read data from modbus outstation'''
    return {}


async def broadcast(msg):
    '''Forward messages to neighbors of `node_id`'''

    for i in neighbors[msg['src']]:
        await send(i, msg)

    return {'function': 'ack'}


async def reset_loads(new_P, new_Q):
    global nodes_diff, nodes_converged, voltage, p_gen, q_gen
    ''' This function will send the new load to all nodes (groups)'''
    nodes_diff = [1.0]*6
    nodes_converged = [0]*6
    p_gen = {}
    q_gen = {}
    voltage = {}
    ploads = {}
    qloads = {}
    for bus in range(48):
        ploads[bus] = new_P[bus]
        qloads[bus] = new_Q[bus]
    msg = {'function': 'set_loads', 'ploads' : ploads, 'qloads' : qloads}
    for i in range(6):
        await send(i, msg)

    return {'function': 'ack'}

async def converged(msg):
    global nodes_diff, nodes_converged, voltage, p_gen, q_gen
    '''record the converged status of the source. return the status of all nodes'''
    nodes_diff[msg['src']] = msg['diff']
    print(f"{nodes_diff}", end='\r')
    status = (sum(nodes_diff) ** 0.5 < 1e-6)
    response = {'function': 'converge' , 'status' : status}
    if status:
        # record p_gen and voltages
        p_gen[msg['src']] = msg['pgen']
        q_gen[msg['src']] = msg['qgen']
        for i in range(48):
            if str(i) in msg:
                voltage[i] = msg[str(i)]
        nodes_converged[msg['src']] = 1
        # if all(nodes_converged):
        #     await reset_loads()
    return response



if __name__ == "__main__":
    asyncio.run(main())