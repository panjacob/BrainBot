import mne
import os
import socket 
import struct
from communication_parameters import *
from itertools import chain
from time import sleep

def filter_data(raw_data):
    raw_data.load_data()
    return raw_data.copy().filter(l_freq = 0.5, h_freq=45.0)
## kopia funkcji z visualize_edf.py
if __name__ == '__main__':

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_ADDRESS_SIMULATOR, UDP_PORT_SIMULATOR))

    file = os.path.join('BrainStudy/data/raw', 'Data1Chiil.bdf')
    data = mne.io.read_raw_bdf(file)
    data = filter_data(data)
    raw_data = data.get_data()
    raw_data = raw_data[:channels, :]
  
    seq = 0
    while (seq < raw_data.shape[1]):
        samples_range = range(seq*samples, seq*samples + samples)
        raw_data_flat = list(chain.from_iterable(raw_data[:channels, samples_range])) #🚂(channel)(channel)(channel)
        message_bytes = struct.pack("%sf" % len(raw_data_flat), *raw_data_flat)
        sock.sendto(message_bytes, (UDP_ADDRESS_VISUALISATION, UDP_PORT_VISUALISATION))
        sleep(128/fs)
        seq = seq + 1