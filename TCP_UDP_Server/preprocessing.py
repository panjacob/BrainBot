from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def filter_data(data, fs,  fc):
    data_filtered = np.zeros(data.shape)
    w = fc / (fs/2)
    b, a = butter(5, w, 'low')
    for i in range(data.shape[1]):
        data_filtered[:,i] = filtfilt(b, a, data[:, i])
    return data_filtered

# 