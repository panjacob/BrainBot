import communication_parameters
import numpy as np
from scipy.signal import decimate
from BrainStudy.svm_preprocessing import eeg_signal_to_dwt


# EEG signal ranges and unit
DIGITAL_MIN = -8388608
DIGITAL_MAX = 8388607
PHYSICAL_MIN = -262144
PHYSICAL_MAX = 262144
UNIT = 1e-6  # μV

CAL = (PHYSICAL_MAX - PHYSICAL_MIN) / (DIGITAL_MAX - DIGITAL_MIN)
OFFSET = PHYSICAL_MIN - DIGITAL_MIN * CAL

# TODO: Make common constants
SPLIT_PADING = 25
SPLIT_LENGTH = 200
CHANNELS_COUNT = 16

BIOSEMI_FREQ = 2000
DATASET_FREQ = 500
DECIMATION_FACTOR = BIOSEMI_FREQ / DATASET_FREQ

# TODO: Load it from file in prepare_data_for_classification
MEAN = [-1.53403974e-08, -6.20724583e-09,  2.46784659e-09,  6.78021528e-10,
        -3.66308983e-09, -1.61879630e-08,  5.27457411e-09,  2.57755195e-09,
        -1.01040387e-09, -8.43589110e-09,  1.17398935e-08,  4.17136015e-08,
        2.66637965e-08,  3.27091492e-08, -2.20020517e-08, -1.55450426e-08]


STD = [1.08722106e-05, 1.15112371e-05, 1.00754341e-05, 1.03625598e-05,
       9.87264593e-06, 8.93015840e-06, 1.03201528e-05, 1.07836222e-05,
       1.05311265e-05, 1.01829701e-05, 1.21622943e-05, 1.26108616e-05,
       1.14071245e-05, 1.25667975e-05, 9.41451344e-06, 1.40940447e-05]

MEAN = np.tile(MEAN, (SPLIT_PADING, 1)).T
STD = np.tile(STD, (SPLIT_PADING, 1)).T


# TODO: I think we also should remove the last channel (triggers), as the client will not need it - check it in the lab
def decode_data_from_bytes(raw_data):
    data_struct = np.zeros((communication_parameters.samples, communication_parameters.channels))

    # 32 bit unsigned words reorder
    raw_data_array = np.array(raw_data)
    raw_data_array = raw_data_array.reshape((communication_parameters.words, 3))
    raw_data_array = np.transpose(raw_data_array)
    normaldata = raw_data_array[2, :]*(256**3) + raw_data_array[1, :]*(256**2) + raw_data_array[0, :]*256 + 0
    for i in range(communication_parameters.samples):
        for j in range(communication_parameters.channels):
            data_struct[j, i] = normaldata[i + j].astype('float32')
            data_struct[j, i] *= CAL
            data_struct[j, i] += OFFSET
            data_struct[j, i] *= UNIT
    return data_struct


def prepare_data_for_classification(decoded_data, mean_std_file_path):
    decimated_signal = np.apply_along_axis(decimate, 1, decoded_data, DECIMATION_FACTOR)  # Accounts for different freqs
    decimated_signal = decimated_signal[:, :SPLIT_PADING]
    decimated_signal -= MEAN
    decimated_signal /= STD


def get_classification(buffer, lda, clf):
    dwt = eeg_signal_to_dwt(buffer)
    data_reduced = lda.transform(dwt)
    prediction = clf.predict(data_reduced)
    return int(prediction)
