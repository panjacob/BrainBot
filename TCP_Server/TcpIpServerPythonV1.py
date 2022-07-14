import communication_parameters
import socket
import datetime as dt
import struct
import random
import sys
import pickle
import data_converter as dc
import numpy as np
from scipy.signal import decimate
from scipy.stats import linregress
from communication_parameters import *


if __name__ == '__main__':
    svm_model_path = sys.argv[1]  # Pass as first argument
    lda_model_path = sys.argv[2]  # Pass as second argument
    mean_std_file_path = sys.argv[3]  # Pass as third argument

    seq_num = random.randint(0, 2 ^ 32 - 1)

    with open(svm_model_path, "rb") as svm_file:
        clf = pickle.load(svm_file)
    with open(lda_model_path, "rb") as lda_file:
        lda = pickle.load(lda_file)
    mean, std = dc.load_mean_std(mean_std_file_path)

    tcp_client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    udp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    tcp_client_sock.bind(("localhost", TCP_LOCAL_PORT))
    tcp_client_sock.connect((TCP_AV_ADDRESS, TCP_AV_PORT))
    udp_server_sock.bind((UDP_IP_ADDRESS, UDP_PORT))
    udp_server_sock.connect((UDP_IP_ADDRESS, UDP_PORT))

    buffer = np.zeros((dc.CHANNELS_COUNT, 2048 * 5))

    alpha_beta_buffer = np.zeros(5)

    filters = dc.initialize_filters()
    cnt = 0
    old_slope = 0
    while True:

        data = tcp_client_sock.recv(communication_parameters.words * 3)
        rawData = struct.unpack(str(communication_parameters.words * 3) + 'B', data)
        decoded_data = dc.decode_data_from_bytes(rawData) # [channels, samples]
        # triggers = np.bitwise_and(decoded_data[16, :].astype(int), 2 ** 17 - 1)
        normalized_data = dc.normalize_to_reference(decoded_data[:channels-1, :] , 14)
        #normalized_data = decoded_data[:channels-1, :]
        decimated_signal = np.apply_along_axis(decimate, 1, normalized_data, DECIMATION_FACTOR)  # Accounts for different freqs
        buffer = np.roll(buffer, -samples, axis=1)
        buffer[:, -samples:] = decoded_data[:channels-1, :]
        if cnt % 5 == 0:
            alpha_beta = dc.prepare_data_for_classification(buffer, mean, std, filters, cnt)
            #buffer_no_padding = buffer_no_padding[:, dc.BUFFER_PADDING:]
            #print(len(np.where(buffer[0, :] != 0)[0]))=
            #label = dc.get_classification(buffer_no_padding, lda, clf)
            ##print(label)
            alpha_beta_buffer = np.roll(alpha_beta_buffer, -1)
            alpha_beta_buffer[4] = alpha_beta
            slope = linregress(range(5), alpha_beta_buffer).slope
            left = True if slope < 0 else False
            forward = True
            send_string = '{"left": ' + str(left).lower() + ', "forward": ' + str(forward).lower() + "} "
            message_bytes = send_string.encode()
            result_to_send = struct.pack("I", seq_num) + message_bytes
            udp_server_sock.sendto(result_to_send, (REMOTE_UDP_ADDRESS, REMOTE_UDP_PORT))

        seq_num += 1
        if seq_num == 2 ^ 32:
            seq_num = 0

        alpha_beta_old = alpha_beta

        cnt += 1
