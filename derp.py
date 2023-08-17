import asyncio
import json
import random
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


def read_modbus(msg):
    '''Read data from modbus outstation'''
    return {}


async def broadcast(msg):
    '''Forward messages to neighbors of `node_id`'''

    for i in neighbors[msg['src']]:
        await send(i, msg)

    return {'function': 'ack'}

def read_loads():
    ploads = {}
    qloads = {}
    for i in range(48):
        ploads[i] = random.random()*0.6 + 0.4
        qloads[i] = random.random() * 0.6 + 0.4
    return(ploads, qloads)

async def reset_loads():
    global nodes_diff, nodes_converged, voltage, p_gen
    print(f"voltages: {voltage}")
    print(f"pgen : {p_gen}")
    nodes_diff = [1.0]*6
    nodes_converged = [0]*6
    p_gen = {}
    voltage = {}
    (ploads, qloads) = read_loads()
    msg = {'function': 'set_loads', 'ploads' : ploads, 'qloads' : qloads}
    for i in range(6):
        await send(i, msg)

    return {'function': 'ack'}

async def converged(msg):
    global nodes_diff, nodes_converged, voltage, p_gen
    '''record the converged status of the source. return the status of all nodes'''
    nodes_diff[msg['src']] = msg['diff']
    print(f"{nodes_diff}", end='\r')
    status = (sum(nodes_diff) ** 0.5 < 1e-4)
    response = {'function': 'converge' , 'status' : status}
    if status:
        nodes_converged[msg['src']] = 1
        # record p_gen and voltages
        p_gen[msg['gen']] = msg['pgen']
        for i in range(48):
            if str(i) in msg:
                voltage[i] = msg[str(i)]
        if all(nodes_converged):
            await reset_loads()
    return response

async def main():
    await run_server('', 10000)


if __name__ == "__main__":
    asyncio.run(main())
