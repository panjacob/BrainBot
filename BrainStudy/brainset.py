"""
Loading the data.
"""

import os
import random
import mne
import json
import math
from sklearn.decomposition import FastICA
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter, buttord
from scipy import fft
from torch.utils.data import Dataset, DataLoader
from signal_parameters import *
from brainset_parameters import *


def select_train_test_files():
    #files_idx = list(range(0, 30))
    #files_idx = random.sample(range(0, 35), size)
    train_files_idx = list(range(0, 5))
    train = []
    for idx in train_files_idx:
        az = "0" if idx < 10 else ""  # additional zero to print numbers like this: 00 01 09 and 10 22 34.
        train.append("Subject" + az + str(idx) + "_1.edf")
        train.append("Subject" + az + str(idx) + "_2.edf")

    test_files_idx = list(range(5, 10))
    test = []
    for idx in test_files_idx:
        az = "0" if idx < 10 else ""  # additional zero to print numbers like this: 00 01 09 and 10 22 34.
        test.append("Subject" + az + str(idx) + "_1.edf")
        test.append("Subject" + az + str(idx) + "_2.edf")

    return train, test


def load_data(train_batch_size=8, test_batch_size=2):
    path = os.path.join(DATA_PATH)
    brainset_train = Brainset(path, is_trainset=True, load_pickled=DATA_PICKLED)
    brainset_test = Brainset(path, is_trainset=False, load_pickled=DATA_PICKLED)
    train_loader = DataLoader(brainset_train, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(brainset_test, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def select_train_files():
    train_percentage = 0.7
    files_idx = random.sample(range(0, 35), math.ceil(35 * train_percentage))
    result = []
    for idx in files_idx:
        az = "0" if idx < 10 else ""  # additional zero to print numbers like this: 00 01 09 and 10 22 34.
        result.append("Subject" + az + str(idx) + "_1.edf")
        result.append("Subject" + az + str(idx) + "_2.edf")
    return result


class Brainset(Dataset):
    """
        Dataset to load EEG signal data from edf files (or pickled files).
    """


    def __init__(self, path, is_trainset, load_pickled=False, mean=None, std=None, use_filter=False):
        np.seterr(all='raise')
        # List containig the data:
        self.brain_set = []

        self.mean = mean
        self.std = std
        self.total_sample_count = 0
        self.normalization_sum = np.zeros(CHANNELS_COUNT)
        self.normalization_sq_sum = np.zeros(CHANNELS_COUNT)
        self.filter = None

        if is_trainset:
            pickle_path = PICKLE_PATH_TRAIN
        else:
            pickle_path = PICKLE_PATH_TEST

        with open("channel_mapping.json", 'r') as json_mapping_file:
            self.channel_ordering = json.load(json_mapping_file)['mapping']

        if use_filter:
            self.filter = dict()
            ord, wn = buttord(LOW_PASS_FREQ_PB, LOW_PASS_FREQ_SB, MAX_LOSS_PB, MIN_ATT_SB, False, DATASET_FREQ)
            self.filter["b_low"], self.filter["a_low"] = butter(ord, wn, 'lowpass', False, 'ba', DATASET_FREQ)

            ord, wn = buttord(HIGH_PASS_FREQ_PB , HIGH_PASS_FREQ_SB, MAX_LOSS_PB, MIN_ATT_SB, False, DATASET_FREQ)
            self.filter["b_high"], self.filter["a_high"] = butter(ord, wn, 'highpass', False, 'ba', DATASET_FREQ)

            #t = np.linspace(0, 1, 2000, False)  # 1 second
            #sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 85 * t)
            #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            #ax1.plot(t, sig)
            #ax1.set_title('10 Hz and 20 Hz sinusoids')
            #ax1.axis([0, 1, -2, 2])
            #filtered = self.__filter(sig)
            #ax2.plot(t, filtered)
            #ax2.set_title('After 15 Hz high-pass filter')
            #ax2.axis([0, 1, -2, 2])
            #ax2.set_xlabel('Time [seconds]')
            #plt.tight_layout()
            #plt.show()

        if not load_pickled:
            # Load data from the source (edf files)
            files = sorted(os.listdir(path))
            #files = files[:36]
            #files = filter(lambda x: x.endswith(".edf"), files)
            
            # Split files to test and train files:
            train_files = select_train_files()
            test_files = [x for x in files if x not in train_files]
            #train_files, test_files = select_train_test_files()
            
            # Set dataset files (either test or train)
            #files = train_files if is_trainset else test_files

            y_all = np.empty((CHANNELS_COUNT, 0), dtype=np.float32)
            y_all_list = []

            # Get data signals form files:
            for index, filename in enumerate(files[2:]):
                # Extract labels:
                #class_idx = int(filename.split('_')[-1][0])  # labels are saved inside file_name at the end after "_"
                #label = np.float32(CLASSES[class_idx])  # labels are shifted 0,1 <- 1,2 in files
                # Extract data:
                file = os.path.join(DATA_PATH, filename)
                data = mne.io.read_raw_bdf(file, verbose=False)
                raw_data = data.get_data()
                # Cut data to unified size:
                # TODO: Discuss why should we even cut the data, for now only cut it so it is divisible for splitting
                y_length = raw_data.shape[1] - (raw_data.shape[1] % SPLIT_LENGTH)
                y_unordered = raw_data[:, :y_length].astype(np.float32)

                if use_filter:
                    y_filtered = np.apply_along_axis(self.__filter, 1, y_unordered)
                else:
                    y_filtered = y_unordered

                ica_transformer = FastICA(16, whiten='unit-variance', max_iter=10000)
                after_ica = ica_transformer.fit_transform(y_filtered[:, 13000:26000].T).T

                fig, axs = plt.subplots(16)
                for i, ax in enumerate(axs):
                    ax.plot(after_ica[i, :])
                plt.show()

                X = fft.fft(y_filtered.flatten())
                N = len(X)
                n = np.arange(N)
                T = N / DATASET_FREQ
                freq = n / T

                plt.figure(figsize=(12, 6))
                plt.subplot(121)

                plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
                plt.xlabel('Freq (Hz)')
                plt.ylabel('FFT Amplitude |X(freq)|')
                plt.xlim(0, 100)

                plt.show()

                # Pick and order channels like in our Biosemi EEG
                y = self.__order_channels(y_filtered)
                y = y[:, REMOVE_PADDING:-REMOVE_PADDING]
                y_all = np.append(y_all, y, axis=1)
                y_all_list.append(y)

                # Cut signals into small signals (to get more data and to fit it into neural net input):
                y_split = []
                for i in range(SPLIT_LENGTH, y.shape[1], SPLIT_PADING):
                    y_sample = y[:, i-SPLIT_LENGTH:i]
                    y_split.append(y_sample)
                for ysx in y_split:
                    self.brain_set.append([ysx, label, filename])

            # Normalize using precalculated sums
            self.__normalize(y_all, y_all_list)

            # Save to pickled files for later use:
            with open(pickle_path, "wb") as pickle_file:
                pickle.dump(self.brain_set, pickle_file)
                if is_trainset:
                    print("Train ", end="")
                else:
                    print("Test ", end="")
                print("dataset normalized and saved")

            # Save train mean and std to text file (for prediction):
            if is_trainset:
                with open(MEAN_STD_PATH, "w") as mean_std_file:
                    with np.printoptions(threshold=np.inf):
                        mean_std_file.write(f"Mean:\n{self.mean[:, 1]}\n\n\nStd:\n{self.std[:, 1]}")

        # Load normalized data, that was previously pickled
        else:
            with open(pickle_path, "rb") as pickle_file:
                self.brain_set = pickle.load(pickle_file)

        # Shuffle the data
        random.shuffle(self.brain_set)

    def __normalize(self, y_all, y_all_list):
        # Normalize using the calcualted sums
        if self.mean is None or self.std is None:
            self.mean = np.zeros(CHANNELS_COUNT)
            self.std = np.zeros(CHANNELS_COUNT)
            for i in range(CHANNELS_COUNT):
                self.mean[i] = y_all[i, :].mean()
                self.std[i] = y_all[i, :].std()

            self.mean = np.tile(self.mean, (SPLIT_LENGTH, 1)).T
            self.std = np.tile(self.std, (SPLIT_LENGTH, 1)).T

        for y in y_all_list:
            for i in range(0, y.shape[1], SPLIT_LENGTH):
                y[:, i:i+SPLIT_LENGTH] -= self.mean
                y[:, i:i+SPLIT_LENGTH] /= self.std

    def __order_channels(self, unordered):
        ordered = unordered[self.channel_ordering, :]
        return ordered

    def __filter(self, data):
        low_filtered = lfilter(self.filter["b_low"], self.filter["a_low"], data)
        filtered = lfilter(self.filter["b_high"], self.filter["a_high"], low_filtered)
        return filtered

    def __len__(self):
        return len(self.brain_set)

    def __getitem__(self, idx):
        return self.brain_set[idx]
