

import os
import logging
import random
import numpy as np
import pandas as pd
import datetime as dtt
import pytz
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from pyModbusTCP import utils
from pyModbusTCP.client import ModbusClient
from pymodbus.payload import BinaryPayloadBuilder, Endian, BinaryPayloadDecoder

# from pymodbus.client import ModbusTcpClient


# connect to MySQL database
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


# conversion to binary for EPC
def convert2binary(builder, value):
    builder.reset()
    builder.add_16bit_int(value)
    payload = builder.to_registers()
    return payload[0]

def convert2values(values):
    return np.array(values).astype(np.int16)


# Read the meter data from three meters for three areas
def read_meters_data(scale_factor = 1):
    base_power = 1000.0
    geisel = ModbusClient(host="172.31.213.2", port=502, unit_id=1, auto_open=True, timeout=2.0)
    rady = ModbusClient(host="172.31.212.150", port=502, unit_id=1, auto_open=True)
    mcgill = ModbusClient(host="172.31.212.151", port=502, unit_id=2, auto_open=True)
    activeA1 = geisel.read_holding_registers(3054-1, 2)
    reactiveA1 = geisel.read_holding_registers(3062-1, 2)
    activeA2 = rady.read_holding_registers(3054-1, 2)    # active power in W
    reactiveA2 = rady.read_holding_registers(3062-1, 2)
    activeA3 = mcgill.read_holding_registers(197, 2)    # active power in W
    reactiveA3 = mcgill.read_holding_registers(207, 2)
    activeA1 = convert_float(activeA1[0], activeA1[1]) / base_power
    reactiveA1 = convert_float(reactiveA1[0], reactiveA1[1]) / base_power
    activeA2 = convert_float(activeA2[0], activeA2[1]) / base_power
    reactiveA2 = convert_float(reactiveA2[0], reactiveA2[1]) / base_power
    activeA3 = round(((activeA3[0] << 16) | activeA3[1]) * 1000, 2) / base_power
    reactiveA3 = round(((reactiveA3[0] << 16) | reactiveA3[1]) * 1000, 2) / base_power
    data = [(scale_factor * activeA3, 2 * scale_factor * reactiveA3), (0.00033 * activeA2, 0.00033 * reactiveA2), (scale_factor * activeA1, 2 * scale_factor * reactiveA1)]
    # df = pd.DataFrame([data], columns=['Pa1', 'Qa1', 'Pa2', 'Qa2', 'Pa3', 'Qa3'])
    logging.info(f"Meters were read: {data} \n")
    # return df
    return data


# Read voltages from RTDS
def read_voltages(scada_client):
    matrix = [0, round(0.0001 * (scada_client.read_input_registers(address=0, count=1)).registers[0], 3)]
    for i in range(1,48):
        matrix = np.vstack([matrix, [i, round(0.0001 * (scada_client.read_input_registers(address=i, count=1)).registers[0], 3)]])
    df = pd.DataFrame(matrix, columns=['bus', 'voltage'])
    df.bus = df.bus.astype('int')
    logging.info(f"voltages were read from RTDS. voltages:{list(df.voltage.values)} \n") 
    return df


# generate randomg voltages 
def generate_voltages(voltages = None):
    if voltages is None:
        matrix = [0, round(random.uniform(0.6, 1.5), 3)]
        for i in range(1,48):
            matrix = np.vstack([matrix, [i, round(random.uniform(0.6, 1.5), 3)]])
        logging.info(f"Random voltages were generated.") 
    else:
        matrix = [0, voltages[0]]
        for i, voltage in enumerate(voltages[1:]):
            matrix = np.vstack([matrix, [i+1, voltage]])
        logging.info(f"Voltages were converted to dataframe.") 
    df = pd.DataFrame(matrix, columns=['bus', 'voltage'])
    df.bus = df.bus.astype('int')
    logging.info(f"Voltages:{list(df.voltage.values)} \n") 
    return df


# Write load scale factor values to RTDS
def send_load_values_to_rtds(scada_client, load_factors):
    for idx, load in enumerate(load_factors):
        # set real load
        scada_client.write_register(
            address=2 * idx + 1,
            value=int(10000 * load[0])
        )
        # set reactive load
        scada_client.write_register(
            address=2 * idx + 2,
            value=int(10000 * load[1])
        )
    logging.info(f"loads were sent to RTDS.\n") 


# Write setpoints (Pg, and Qg) to RTDS and EPC
def send_setpoints_to_rtds_and_epc(scada_client, epc_client, setpoints):
    builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Big)
    generators = [47, 23, 37, 15, 13, 45]
    gen_input_map = {
        15:(7,8),
        23:(9,10),
        37:(11,12),
        45:(13,14),
        47:(15,16),
    }
    gen_coil_map = {
        15:(0,1),
        23:(2,3),
        37:(4,7),
        45:(6,7),
        47:(8,9),
    }
    for idx, setpoint in enumerate(setpoints):
        if idx != 4:
            # set real BESS input register
            scada_client.write_register(
                address=gen_input_map[generators[idx]][0],
                value=int(1e2 * abs(setpoint[0]))
            )
            # set real BESS coid register
            scada_client.write_coil(
                address=gen_coil_map[generators[idx]][0],
                value=1 if setpoint[0] >= 0 else 0
            )
            # set reactive BESS input register
            scada_client.write_register(
                address=gen_input_map[generators[idx]][1],
                value=int(1e2 * abs(setpoint[1]))
            )
            scada_client.write_coil(
                address=gen_coil_map[generators[idx]][1],
                value=1 if setpoint[1] >= 0 else 0
            )
        elif idx == 4:
            # Command active power
            epc_client.write_register(3215, convert2binary(builder, int(10000 * setpoint[0])))
            # Command reactive power
            epc_client.write_register(3216, convert2binary(builder, int(10000 * setpoint[1])))
    logging.info(f"setpoints were sent to RTDS and EPC. setpoints: {setpoints} \n")    



# get load (Pl and Ql) values
def get_loads(scale_factor = 1, scada_client = None, existing_setpoint_13 = None):
    load_factors = read_meters_data(scale_factor = scale_factor)
    load_aggregatiosn = [0, 0, 0, 0, 0, 0]
    n = 48
    P = [0] * n
    Q = [0] * n
    scale = 1000
    buses = range(n)
    # Scale everything to units of MW (from KW on the sheet)
    loads = pd.read_excel(
        "grid_data/UCSDmicrogrid_iCorev3_info.xlsx",
        sheet_name="Buses_new"
    ).to_numpy()
    for i in buses:
        P[loads[i+1][1]] = float(loads[i+1][5])/scale
        Q[loads[i+1][1]] = float(loads[i+1][6])/scale
        if not np.isnan(loads[i+1][9]):
            P[loads[i+1][1]] -= float(loads[i+1][9])/scale
    if load_factors is not None:
        logging.info(f"loads are scaled with {load_factors}")
        load_groups = [
            [6, 7, 8, 9, 10, 22, 23, 24, 25, 26, 31, 32, 33, 35, 36, 37, 38, 39, 40], # area 1
            [11, 14, 15, 16, 17, 18, 19, 28, 29, 30, 34, 42], # area 2
            [4, 5, 20, 44, 45, 46] # area 3
        ]
        for idxs, load_factor in enumerate(load_factors):
                for idx in load_groups[idxs]:
                    P[idx] = load_factor[0] * P[idx]
                    Q[idx] = load_factor[1] * Q[idx]
                    load_aggregatiosn[2 * idxs] += P[idx]
                    load_aggregatiosn[2 * idxs + 1] += Q[idx]
    if scada_client is not None and existing_setpoint_13 is not None:
        logging.info("EPC noload current is added to load.")
        P[13] += (scada_client.read_input_registers(address=48, count=1)).registers[0] * 1e-2 / scale - existing_setpoint_13[0]
        Q[13] += (scada_client.read_input_registers(address=49, count=1)).registers[0] * 1e-2 / scale - existing_setpoint_13[1]
    df = scale * pd.DataFrame([load_aggregatiosn], columns=['Pa1', 'Qa1', 'Pa2', 'Qa2', 'Pa3', 'Qa3'])
    logging.info(f"PL: {P}")
    logging.info(f"PL: {Q}")
    logging.info(f"QL: {df} \n")
    return (P, Q, load_factors, df)
