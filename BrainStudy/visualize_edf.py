import mne
import os
import matplotlib.pyplot as plt


def plotRawSignal(raw_data):
    x = raw_data[2, :]
    print('len:', len(x))
    # S{numer_osoby}R{numer_zadania}_{numer_sygnalu_eeg(numer sondy?)-64_{nazwa_zadania}}
    title = "S001R01_1-64_Baseline, eyes open"
    plt.title(title)
    plt.plot(x)
    plt.show()
    plt.savefig(os.path.join('plots', "S001R01_1-64_Baseline, eyes open"))



file = os.path.join('BrainStudy/data/raw', 'Data11Zagadka.bdf') #file = os.path.join('data/mentalload/raw', 'Subject01_2.edf')
data = mne.io.read_raw_bdf(file)
#raw_data = data.get_data()
#info = data.info
#print(info)
#channels = data.ch_names
print(data)
print(data.info)
data.plot(n_channels=16, highpass=0.5, lowpass=45, theme="light", block=True, color=dict(eeg='blue'))
