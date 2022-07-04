import numpy as np
import communication_parameters
import socket 

TCPport = 8888 #port configured in Activeview
TCPipaddress = 'localhost' #ip adress of Actiview host

UDPport = 8890
UDPipaddress = 'localhost'

tcpClient_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
udpServer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

tcpClient_sock.bind((TCPipaddress, TCPport))

tcpClient_sock.listen()

connection, client_address = tcpClient_sock.accept()

while(True):
    data = connection.recv(communication_parameters.words*3)
    print(data)
    udpServer_sock.sendto(data, (UDPipaddress, UDPport))

