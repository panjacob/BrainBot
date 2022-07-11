import communication_parameters
import socket
import struct
import random
import sys
import pickle
import data_converter as dc
import numpy as np

TCP_PORT = 8600  # port configured in Activeview
TCP_IP_ADDRESS = 'localhost'  # IP adress of Actiview host

UDP_PORT = 42802
UDP_IP_ADDRESS = 'localhost'

SVM_MODEL_PATH = sys.argv[1]  # Pass as first argument
LDA_MODEL_PATH = sys.argv[2]  # Pass as second argument
MEAN_STD_FILE_PATH = sys.argv[3]  # Pass as third argument


seq_num = random.randint(0, 2 ^ 32 - 1)

with open(SVM_MODEL_PATH, "rb") as svm_file:
    clf = pickle.load(svm_file)
with open(LDA_MODEL_PATH, "rb") as lda_file:
    lda = pickle.load(lda_file)

tcp_client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
udp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #socket.SOCK_DGRAM

tcp_client_sock.bind((TCP_IP_ADDRESS, TCP_PORT))

tcp_client_sock.connect(("localhost", 8888))


udp_server_sock.bind((UDP_IP_ADDRESS, UDP_PORT))

udp_server_sock.connect((UDP_IP_ADDRESS, UDP_PORT))

mean, std = dc.load_mean_std(MEAN_STD_FILE_PATH)

buffer = np.zeros((dc.CHANNELS_COUNT, dc.SPLIT_LENGTH))

while True:
    data = tcp_client_sock.recv(communication_parameters.words * 3)
    rawData = struct.unpack(str(communication_parameters.words * 3) + 'B', data)
    decoded_data = dc.decode_data_from_bytes(rawData)  # [channels, samples]
    prepared_data = dc.prepare_data_for_classification(decoded_data, mean, std)
    buffer = np.roll(buffer, -dc.SPLIT_PADING, axis=1)
    buffer[:, -dc.SPLIT_PADING:] = prepared_data

    label = dc.get_classification(buffer, lda, clf)
    left = True if label == 1 else False
    forward = True
    send_string = "{'left': " + str(left) + ", 'forward': " + str(forward) + "} "
    message_bytes = send_string.encode()
    #result_to_send = struct.pack("I", seq_num) + struct.pack("i", seq_num)
    udp_server_sock.sendall(message_bytes)
    #udp_server_sock.sendto(message_bytes, (UDP_IP_ADDRESS, UDP_PORT))

    seq_num += 1
    if seq_num == 2 ^ 32:
        seq_num = 0
