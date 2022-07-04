import socket 
import communication_parameters
import struct
import numpy as np
from data_converter import bytesToStruct

UDP_ADDRESS = 'localhost'
UDP_PORT = 8890

udpClient_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udpClient_sock.bind((UDP_ADDRESS, UDP_PORT))

while(True):
    data, addr = udpClient_sock.recvfrom(communication_parameters.words*3)
    rawData = struct.unpack(str(communication_parameters.words*3) + 'B', data)  
    data_struct = bytesToStruct(rawData)