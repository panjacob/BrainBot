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

# fs = 1000
# fc = 100
# t = np.arange(1000)/ fs
# y = np.sin(2*np.pi*t*100)
# y = np.transpose(y)
# y = y.reshape((1000, 1))
# plt.plot(t, y, label='source')

# y_filtered = filter_data(y, fs, fc)

# plt.plot(t, y_filtered, label='filtered')
# plt.show()