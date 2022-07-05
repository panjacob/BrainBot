"""
Loading the data.
"""

import os
import random
import mne
import json
import math
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from signal_parameters import *


DIR_PATH = "data/mentalload"
DATA_PATH = DIR_PATH + "/raw"
PICKLE_PATH_TRAIN = DIR_PATH + "/train.pickle"
PICKLE_PATH_TEST = DIR_PATH + "/test.pickle"
MEAN_STD_PATH = DIR_PATH + "/mean_std.txt"

DATA_PICKLED = False  # Enable if Data has been saved previously saved in pickled files

CLASSES = {
    1: 0,
    2: 1
}


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


def load_data(train_batch_size=8,test_batch_size=2):
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


    def __init__(self, path, is_trainset, load_pickled=False, mean=None, std=None):
        np.seterr(all='raise')
        # List containig the data:
        self.mean = mean
        self.std = std
        self.brain_set = []
        self.total_sample_count = 0
        self.normalization_sum = np.zeros(CHANNELS_COUNT)
        self.normalization_sq_sum = np.zeros(CHANNELS_COUNT)
        if is_trainset:
            pickle_path = PICKLE_PATH_TRAIN
        else:
            pickle_path = PICKLE_PATH_TEST

        with open("channel_mapping.json", 'r') as json_mapping_file:
            self.channel_ordering = json.load(json_mapping_file)['mapping']

        if not load_pickled:
            # Load data from the source (edf files)
            files = sorted(os.listdir(path))
            files = files[:36]
            files = filter(lambda x: x.endswith(".edf"), files)
            
            # Split files to test and train files:
            train_files = select_train_files()
            test_files = [x for x in files if x not in train_files]
            #train_files, test_files = select_train_test_files()
            
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
                # Cut data to unified size:
                # TODO: Discuss why should we even cut the data, for now only cut it so it is divisible for splitting
                y_length = raw_data.shape[1] - (raw_data.shape[1] % SPLIT_LENGTH)
                y_unordered = raw_data[:, :y_length].astype(np.float32)

                # Pick and order channels like in our Biosemi EEG
                y = self.__order_channels(y_unordered)
                y = y[REMOVE_PADDING:-REMOVE_PADDING]
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

    def __len__(self):
        return len(self.brain_set)

    def __getitem__(self, idx):
        return self.brain_set[idx]
