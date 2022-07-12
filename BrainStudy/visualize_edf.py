import mne
import os
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

def plot_raw_signal(raw_data):
    x = raw_data[2, :]
    print('len:', len(x))
    # S{numer_osoby}R{numer_zadania}_{numer_sygnalu_eeg(numer sondy?)-64_{nazwa_zadania}}
    title = "S001R01_1-64_Baseline, eyes open"
    plt.title(title)
    plt.plot(x)
    plt.show()
    plt.savefig(os.path.join('plots', "S001R01_1-64_Baseline, eyes open"))

def filter_data(raw_data):
    raw_data.load_data()
    return raw_data.copy().filter(l_freq = 0.5, h_freq=45.0)

def PSD(data_filtered):
    kwargs = dict(fmin=0.5, fmax=45, n_jobs=1)
    print(data_filtered)
    psds, freqs = mne.time_frequency.psd_welch(data_filtered, average='mean', n_fft=10000, n_per_seg=8192, **kwargs)
    psds_welch_mean  = 10*np.log10(psds)
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [dB]")
    for i in range(16):
        plt.plot(freqs, psds_welch_mean[i], label='A%s' % i)
        print(freqs)
    plt.legend()
    plt.savefig(os.path.join('BrainStudy', 'data/plots/s1_2'))
    plt.show()

if __name__ == '__main__':
    #file = os.path.join('BrainStudy/data/raw', 'Data1Chiil.bdf') #
    file = os.path.join('BrainStudy/data/raw1', 'Subject01_2.edf')
    data = mne.io.read_raw_edf(file)
    data_filtered = filter_data(data)
    #raw_data = data.get_data()
    #info = data.info
    #print(info)
    #channels = data.ch_names
    PSD(data_filtered)
    print(data)
    print(data.info)
    #data.plot(n_channels=16, highpass=0.5, lowpass=45, theme="light", block=True, color=dict(eeg='blue'))
