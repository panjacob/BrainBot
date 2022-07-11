import pywt
import communication_parameters
import numpy as np
import pickle
from scipy.signal import decimate
from BrainStudy.signal_parameters import *


def eeg_signal_to_dwt(data):
    c_allchannels = np.empty(0)
    for channel in data:
        ca1, cd1 = pywt.dwt(channel, 'db1')
        c_allchannels = np.append(c_allchannels, ca1)
        c_allchannels = np.append(c_allchannels, cd1)
    return c_allchannels


def load_mean_std(mean_std_file_path):
    with open(mean_std_file_path, "rb") as mean_std_file:
        mean = pickle.load(mean_std_file)
        std = pickle.load(mean_std_file)

    mean = mean[:, :SPLIT_PADING]
    std = std[:, :SPLIT_PADING]
    return mean, std


# TODO: I think we also should remove the last channel (triggers), as the client will not need it - check it in the lab
def decode_data_from_bytes(raw_data):
    data_struct = np.zeros((communication_parameters.channels, communication_parameters.samples))

    # 32 bit unsigned words reorder
    raw_data_array = np.array(raw_data)
    raw_data_array = raw_data_array.reshape((communication_parameters.words, 3))
    raw_data_array = raw_data_array.astype("int32")
    raw_data_array = np.flip(raw_data_array, 0)
    raw_data_array = ((raw_data_array[:, 0]) +
                      (raw_data_array[:, 1] << 8) +
                      (raw_data_array[:, 2] << 16))
    raw_data_array[raw_data_array >= (1 << 23)] -= (1 << 24)
    normal_data = raw_data_array

    for j in range(communication_parameters.channels):
        for i in range(communication_parameters.samples):
            data_struct[j, i] = normal_data[i * communication_parameters.channels + j].astype('float32')
            data_struct[j, i] *= CAL
            data_struct[j, i] += OFFSET
            data_struct[j, i] *= UNIT

    return np.flip(data_struct, 0)


def prepare_data_for_classification(decoded_data, mean, std):
    decimated_signal = np.apply_along_axis(decimate, 1, decoded_data, DECIMATION_FACTOR)  # Accounts for different freqs
    decimated_signal = decimated_signal[:, :SPLIT_PADING]
    decimated_signal -= mean
    decimated_signal /= std
    return decimated_signal


def get_classification(buffer, lda, clf):
    dwt = eeg_signal_to_dwt(buffer)
    dwt = dwt.reshape(1, -1)
    data_reduced = lda.transform(dwt)
    prediction = clf.predict(data_reduced)
    return int(prediction)
