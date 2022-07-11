import socket
import struct
import communication_parameters
import time
import numpy as np
from math import sin

server_address = ('localhost', 5500)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

data_struct = np.zeros((communication_parameters.samples, communication_parameters.channels, 3))

t0 = time.time()
t = 0
try:
    sock.bind(('localhost', 5501))
    print("ActiviewSimulator connecting to server..")
    sock.connect(server_address)
    print("Connection established!")
    while(True):
        '''
        for i in range(data_struct.shape[0]):
            t = time.time() - t0
            for j in range(data_struct.shape[1]):
                for k in range(data_struct.shape[2]):
                    data_struct[i, j] = j
        '''
        data_struct = "True"
        data = data_struct.encode()

        print("Sending data..")
        sock.sendall(data)
        time.sleep(communication_parameters.samples*1/100)
finally:
    sock.close()