import mne
import os
import matplotlib.pyplot as plt

file = os.path.join('files2', 'Subject00_2.edf')
data = mne.io.read_raw_edf(file)
raw_data = data.get_data()
info = data.info
print(info)
channels = data.ch_names

x = raw_data[1, :]
# S{numer_osoby}R{numer_zadania}_{numer_sygnalu_eeg(numer sondy?)-64_{nazwa_zadania}}
title = "S001R01_1-64_Baseline, eyes open"
plt.title(title)
plt.plot(x)
plt.show()
plt.savefig(os.path.join('plots', "S001R01_1-64_Baseline, eyes open"))
