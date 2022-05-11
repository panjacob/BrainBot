import mne
import os
import matplotlib.pyplot as plt

DATA_PATH = "data/mentalload/"
FILENAME = 'Subject02_2.edf'
class_idx = int(FILENAME.split('_')[-1][0])
file = os.path.join(DATA_PATH, FILENAME)
data = mne.io.read_raw_edf(file)
raw_data = data.get_data()
info = data.info
print(info)
channels = data.ch_names

for i in range(20,21):
    x = raw_data[i, :-1000]
    print('len:', len(x))
    # S{numer_osoby}R{numer_zadania}_{numer_sygnalu_eeg(numer sondy?)-64_{nazwa_zadania}}
    #title = "S001R01_1-64_Baseline, eyes open "
    title = FILENAME +  " " +str(i) + " motion = "+ str(class_idx-1)
    plt.title(title)
    plt.plot(x)
    plt.show()
    plt.savefig(os.path.join('plots', "S001R01_1-64_Baseline, eyes open"))
