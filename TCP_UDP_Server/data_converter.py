import TCP_UDP_Server.TcpIpServerPythonV1
import communication_parameters
import numpy as np
import pickle
from scipy.signal import decimate
from BrainStudy.svm_preprocessing import eeg_signal_to_dwt
from BrainStudy.signal_parameters import *


# EEG signal ranges and unit
DIGITAL_MIN = -8388608
DIGITAL_MAX = 8388607
PHYSICAL_MIN = -262144
PHYSICAL_MAX = 262144
UNIT = 1e-6  # μV

CAL = (PHYSICAL_MAX - PHYSICAL_MIN) / (DIGITAL_MAX - DIGITAL_MIN)
OFFSET = PHYSICAL_MIN - DIGITAL_MIN * CAL

BIOSEMI_FREQ = 2000
DATASET_FREQ = 500
DECIMATION_FACTOR = BIOSEMI_FREQ / DATASET_FREQ

# TODO: Load it from file in prepare_data_for_classification
with open(TCP_UDP_Server.TcpIpServerPythonV1.MEAN_STD_FILE_PATH, "rb") as mean_std_file:
    MEAN = pickle.load(mean_std_file)
    STD = pickle.load(mean_std_file)

MEAN = np.tile(MEAN, (SPLIT_PADING, 1)).T
STD = np.tile(STD, (SPLIT_PADING, 1)).T


# TODO: I think we also should remove the last channel (triggers), as the client will not need it - check it in the lab
def decode_data_from_bytes(raw_data):
    data_struct = np.zeros((communication_parameters.samples, communication_parameters.channels))

    # 32 bit unsigned words reorder
    raw_data_array = np.array(raw_data)
    raw_data_array = raw_data_array.reshape((communication_parameters.words, 3))
    raw_data_array = np.transpose(raw_data_array)
    normal_data = raw_data_array[2, :]*(256**3) + raw_data_array[1, :]*(256**2) + raw_data_array[0, :]*256 + 0
    for i in range(communication_parameters.samples):
        for j in range(communication_parameters.channels):
            data_struct[j, i] = normal_data[i + j].astype('float32')
            data_struct[j, i] *= CAL
            data_struct[j, i] += OFFSET
            data_struct[j, i] *= UNIT
    return data_struct


def prepare_data_for_classification(decoded_data):
    decimated_signal = np.apply_along_axis(decimate, 1, decoded_data, DECIMATION_FACTOR)  # Accounts for different freqs
    decimated_signal = decimated_signal[:, :SPLIT_PADING]
    decimated_signal -= MEAN
    decimated_signal /= STD


def get_classification(buffer, lda, clf):
    dwt = eeg_signal_to_dwt(buffer)
    data_reduced = lda.transform(dwt)
    prediction = clf.predict(data_reduced)
    return int(prediction)
