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

'''
DIR_PATH = "data/mentalload"
DATA_PATH = DIR_PATH + "/raw"
PICKLE_PATH_TRAIN = DIR_PATH + "/train.pickle"
PICKLE_PATH_TEST = DIR_PATH + "/test.pickle"
MEAN_STD_PATH = DIR_PATH + "/mean_std.txt"

CLASSES = {
    1: 0,
    2: 1
}
'''


def selectMentalArithmeticFiles(randomise = False):
    train_files = []
    test_files = []
    max_file_index = 36 #amount of subjects
    #Select test files (indexes)
    if randomise:
        test_percentage = 0.3
        test_files_idx = random.sample(range(0, max_file_index), math.ceil(max_file_index * test_percentage))
    else:
        #test_files_idx = list(range(30, 36))
        test_files_idx = [17, 7, 12, 28, 22, 29, 2, 18, 30, 19, 1]
    #Select train files (complement)
    train_files_idx =  [t for t in list(range(0, max_file_index)) if t not in test_files_idx]

    def fileIdxString(idx):
        result = []
        az = "0" if idx < 10 else ""  # additional zero to print numbers like this: 00 01 09 and 10 22 34.
        result.append("Subject" + az + str(idx) + "_1.edf")
        result.append("Subject" + az + str(idx) + "_2.edf")
        return result

    for idx in train_files_idx:
        train_files.extend(fileIdxString(idx))

    for idx in test_files_idx:
        test_files.extend(fileIdxString(idx))

    return train_files, test_files




def load_data(train_batch_size=8,test_batch_size=2,load_pickled_data=True, use_filter=False):
    path = os.path.join(DATA_PATH)
    brainset_train = Brainset(path, is_trainset=True, load_pickled=load_pickled_data, use_filter=use_filter)
    brainset_test = Brainset(path, is_trainset=False, load_pickled=load_pickled_data, use_filter=use_filter)
    train_loader = DataLoader(brainset_train, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(brainset_test, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader



class Brainset(Dataset):
    """
        Dataset to load EEG signal data from edf files (or pickled files).
        is_trainset - Enable if this dataset will be used for training
        load_pickled - Enable if Data has been saved previously saved in pickled files
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
        self.is_trainset = is_trainset
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

        if not load_pickled:
            # Load data from the source (edf files)
            files = sorted(os.listdir(path))
            files = files[:36]
            files = filter(lambda x: x.endswith(".edf"), files)
            
            # Split files to test and train files:
            train_files, test_files = selectMentalArithmeticFiles(randomise=False)
            
            # Set dataset files (either test or train)
            files = train_files if is_trainset else test_files

            y_all = np.empty((CHANNELS_COUNT, 0), dtype=np.float32)
            y_all_list = []

            # Get data signals form files:
            for index, filename in enumerate(files):
                # Extract labels:
                class_idx = int(filename.split('_')[-1][0])  # labels are saved inside file_name at the end after "_"
                label = np.float32(CLASSES[class_idx])  # labels are shifted 0,1 <- 1,2 in files
                # Extract data:
                file = os.path.join(DATA_PATH, filename)
                data = mne.io.read_raw_edf(file, verbose=False)
                raw_data = data.get_data()
                # Cut data to unified size (and divisible into neat splits):
                y_length = MAX_LENGTH - (MAX_LENGTH % SPLIT_LENGTH)
                y_unordered = raw_data[:, :y_length].astype(np.float32)

                if use_filter:
                    y_filtered = np.apply_along_axis(self.__filter, 1, y_unordered)
                else:
                    y_filtered = y_unordered

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
            print("Dataset Loaded from pickled file")

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

    def stats(self):
        if self.is_trainset :
            print("Train",end=' ')
        else:
            print("Test", end=' ')
        print("Dataset Stats:")
        print("Samples: ", len(self.brain_set))
        ones = 0
        zeroes = 0
        for data in self.brain_set:
            if data[1] == 1:
                ones = ones + 1
            else:
                zeroes = zeroes + 1
        print("Mental Load Samples: ",ones)
        print("Quiet Samples: ", zeroes)
