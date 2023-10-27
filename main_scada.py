import asyncio
import json
import random
import logging
import time
import socket
import numpy as np
import pandas as pd
import datetime as dtt
import pytz
from pymodbus.client import ModbusTcpClient
import pymodbus
import scada
import centralized
from sqlalchemy import create_engine, text

logging.getLogger().setLevel(logging.INFO)

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
voltage = {}

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


async def run_server(ip, port):
    '''Create and run the DERP server'''
    log(f"Starting DERP server on {ip}:{port}")
    server = await asyncio.start_server(handle_client, ip, port)
    await server.serve_forever()


async def dispatch(msg):
    '''Respond to the contents of the client request'''

    # if msg['function'] == 'write_modbus':
    #     response = write_modbus(msg)

    # elif msg['function'] == 'read_modbus':
    #     response = read_modbus(msg)

    if msg['function'] == 'broadcast':
        response = await broadcast(msg)

    elif msg['function'] == 'converge':
        response = await converged(msg)
    else:
        raise NotImplementedError()

    return response


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
    status = (sum(nodes_diff) ** 0.5 < 1e-4)
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


async def main():
    # run the server so that the nodes can connect
    asyncio.create_task(run_server('', 10000))

    dbconn = scada.connect_to_MySQL()

    IPAddr_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    IPAddr_s.connect(("8.8.8.8", 80))
    IPAddr = IPAddr_s.getsockname()[0]
    IPAddr_s.close()
    df_derp = pd.DataFrame([(IPAddr, 10000)], columns=['ip', 'port'])
    df_derp.to_sql("derp_address", dbconn, "icore", "replace", False)

    scada_ip_query = text("SELECT ip, port from icore.tcp_addresses WHERE name = 'scada' AND type = 'rtds'")
    scada_ip, scada_port = dbconn.connect().execute(scada_ip_query).fetchall()[0]
    scada_client = ModbusTcpClient(scada_ip, scada_port)
    scada_client.connect()

    epc_ip_query = text("SELECT ip, port from icore.tcp_addresses WHERE name = 'epc' AND type = 'hardware'")
    epc_ip, epc_port = dbconn.connect().execute(epc_ip_query).fetchall()[0]
    epc_client = ModbusTcpClient(epc_ip, epc_port)
    epc_client.connect()
    epc_client.write_register(246, 0) # disable
    epc_client.write_register(3214, 1) # clear fault
    epc_client.write_register(3215, 0) # command 0 kW
    epc_client.write_register(3216, 0) # command 0 kVar
    epc_client.write_register(246, 1) # enable

    setpoints = [(0. , 0.)] * 6
    voltages = [0] * 48

    for i in range(100):
        control_query = text("SELECT voltage1_display, voltage2_display, algorithm, load_scale, rts_flag FROM icore.controls")
        voltage1_display, voltage2_display, algorithm, load_scale, rts_flag = dbconn.connect().execute(control_query).fetchall()[0]
        try:
            time.sleep(1)
            data_datetime = dtt.datetime.now(pytz.timezone('US/Pacific')).replace(microsecond=0)
            # df_meter_reading = scada.read_meters_data(scale_factor = load_scale)
            # meter_reading = df_meter_reading.values[0][0:6]
            # load_factors = [(meter_reading[0], meter_reading[1]), (meter_reading[2], meter_reading[3]), (meter_reading[4], meter_reading[5])]
            # P, Q = scada.get_loads(load_factors = load_factors, scada_client = scada_client, existing_setpoint_13 = setpoints[4])
            if rts_flag == 1:
                P, Q, load_factors, df_meter_reading = scada.get_loads(scale_factor = load_scale, scada_client = scada_client, existing_setpoint_13 = setpoints[4])
                scada.send_load_values_to_rtds(scada_client, load_factors = load_factors)
            else:
                P, Q, load_factors, df_meter_reading = scada.get_loads(scale_factor = load_scale, existing_setpoint_13 = setpoints[4])

            if voltage1_display == 'rtds':
                df_voltages = scada.read_voltages(scada_client = scada_client)
            else:
                df_voltages = scada.generate_voltages()
                #time.sleep(1)

            df_meter_reading['data_datetime'] = data_datetime
            df_meter_reading.to_sql("meters", dbconn, "icore", "append", False)
            df_voltages['data_datetime'] = data_datetime
            df_voltages.to_sql("voltages", dbconn, "icore", "append", False)
            voltages = list(df_voltages.voltage.values)
            time.sleep(1)
            data_datetime = dtt.datetime.now(pytz.timezone('US/Pacific')).replace(microsecond=0)
            if algorithm == 1:
                setpoints, voltages = centralized.centralized_socp(P, Q, verbose = False)
            elif algorithm == 2:
                await reset_loads(new_P=P, new_Q=Q) # new loads should be arrays of length 47 with the P, Q load associated with the bus at the index of the array
                # wait for solution
                while not (all(nodes_converged)):
                    await asyncio.sleep(0.1)
                # read the solution
                # setpoints = [(0,0)]*6
                # sol_voltages = [0.]*47
                for g in range(6):
                    setpoints[g] = (p_gen[g], q_gen[g])
                for bus in range(48):
                    voltages[bus] = voltage[bus]

            if rts_flag == 1:
                scada.send_setpoints_to_rtds_and_epc(scada_client = scada_client, epc_client = epc_client, setpoints = setpoints)
            if voltage2_display == 'rtds':
                df_voltages = scada.read_voltages(scada_client = scada_client)
            else:
                df_voltages = scada.generate_voltages(voltages)
                #time.sleep(1)
            df_meter_reading['data_datetime'] = data_datetime
            df_meter_reading.to_sql("meters", dbconn, "icore", "append", False)
            df_voltages['data_datetime'] = data_datetime
            df_voltages.to_sql("voltages", dbconn, "icore", "append", False)
            voltages = list(df_voltages.voltage.values)
        except:
            logging.info(f"{data_datetime}: Operation failed. \n") 
        # time.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
