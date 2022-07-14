import pywt
import numpy as np
import pickle
from scipy.signal import decimate
from BrainStudy.signal_parameters import *
from communication_parameters import *
from scipy.signal import lfilter, butter, buttord


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
    data_struct = np.zeros((channels, samples))

    # 32 bit unsigned words reorder
    raw_data_array = np.array(raw_data)
    raw_data_array = raw_data_array.reshape((words, 3))
    raw_data_array = raw_data_array.astype("int32")
    raw_data_array = np.flip(raw_data_array, 0)
    raw_data_array = ((raw_data_array[:, 0]) +
                      (raw_data_array[:, 1] << 8) +
                      (raw_data_array[:, 2] << 16))
    raw_data_array[raw_data_array >= (1 << 23)] -= (1 << 24)
    normal_data = raw_data_array

    for j in range(channels):
        for i in range(samples):
            data_struct[j, i] = normal_data[i * channels + j].astype('float32')
            data_struct[j, i] *= CAL
            data_struct[j, i] += OFFSET
            data_struct[j, i] *= UNIT

    return np.flip(data_struct, 0)


def initialize_filters():
    eeg_filter = dict()
    o, wn = buttord(LOW_PASS_FREQ_PB, LOW_PASS_FREQ_SB, MAX_LOSS_PB, MIN_ATT_SB, False, DATASET_FREQ)
    eeg_filter["b_low"], eeg_filter["a_low"] = butter(o, wn, 'lowpass', False, 'ba', DATASET_FREQ)

    o, wn = buttord(HIGH_PASS_FREQ_PB, HIGH_PASS_FREQ_SB, MAX_LOSS_PB, MIN_ATT_SB, False, DATASET_FREQ)
    eeg_filter["b_high"], eeg_filter["a_high"] = butter(o, wn, 'highpass', False, 'ba', DATASET_FREQ)

    return eeg_filter


def normalize_to_reference(decoded_data, ref_channel):
        return decoded_data - np.tile(decoded_data[ref_channel, :], (channels, 1))


def prepare_data_for_classification(decoded_data, mean, std, eeg_filters):
    decimated_signal = np.apply_along_axis(decimate, 1, decoded_data, DECIMATION_FACTOR)  # Accounts for different freqs
    mean_vec = mean[:, 0]
    std_vec = std[:, 0]
    mean = np.tile(mean_vec, (SAMPLES_DECIMATED, 1)).T
    std = np.tile(std_vec, (SAMPLES_DECIMATED, 1)).T

    low_filtered = lfilter(eeg_filters["b_low"], eeg_filters["a_low"], decimated_signal)
    filtered = lfilter(eeg_filters["b_high"], eeg_filters["a_high"], low_filtered)

    filtered -= mean
    filtered /= std

    return filtered


def get_classification(buffer, lda, clf):
    dwt = eeg_signal_to_dwt(buffer)
    dwt = dwt.reshape(1, -1)
    data_reduced = lda.transform(dwt)
    prediction = clf.predict(data_reduced)
    return int(prediction)
