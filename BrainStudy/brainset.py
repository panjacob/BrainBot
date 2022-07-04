"""
Loading the data.
"""

import os
import random
import mne
import math
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader


DIR_PATH = "data/mentalload"
DATA_PATH = DIR_PATH + "/raw"
PICKLE_PATH_TRAIN = DIR_PATH + "/train.pickle"
PICKLE_PATH_TEST = DIR_PATH + "/test.pickle"

DATA_PICKLED = True  # Enable if Data has been saved previously saved in pickled files

CLASSES = {
    1: 0,
    2: 1
}


def select_train_files():
    files_idx = list(range(0, 30))  # random.sample(range(0, 35), size)
    result = []
    for idx in files_idx:
        az = "0" if idx < 10 else ""  # additional zero to print numbers like this: 00 01 09 and 10 22 34.
        result.append("Subject" + az + str(idx) + "_1.edf")
        result.append("Subject" + az + str(idx) + "_2.edf")
    return result


def load_data(train_batch_size=4,test_batch_size=2):
    path = os.path.join(DATA_PATH)
    brainset_train = Brainset(path, is_trainset=True, load_pickled=DATA_PICKLED)
    brainset_test = Brainset(path, is_trainset=False, load_pickled=DATA_PICKLED)
    train_loader = DataLoader(brainset_train, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(brainset_test, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


class Brainset(Dataset):
    """
        Dataset to load EEG signal data from edf files (or pickled files).
    """
    SIGNAL_LENGTH = 30477
    SPLIT_DATA = True
    SPLIT_LENGTH = 200
    SPLIT_PADING = 5

    def __init__(self, path, is_trainset, load_pickled=False):
        # List containig the data:
        self.brain_set = []
        self.sample_number = 0
        if is_trainset:
            pickle_path = PICKLE_PATH_TRAIN
        else:
            pickle_path = PICKLE_PATH_TEST

        if not load_pickled:
            # Load data from the source (edf files)
            files = os.listdir(path)
            files = filter(lambda x: x.endswith(".edf"), files)
            # Split files to test and train files:
            test_files = select_train_files()
            train_files = [x for x in files if x not in test_files]
            # Set dataset files (either test or train)
            files = train_files if is_trainset else test_files

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
                y = raw_data[:, :self.SIGNAL_LENGTH].astype(np.float32)
                self.__normalize2(y)
                # Check if there is a need to cut signals:
                # if self.SPLIT_DATA is True:

                # Cut signals into small signals (to get more data and to fit it into neural net input):
                #
                y_split = []
                for i in range(self.SPLIT_LENGTH, y.shape[1], self.SPLIT_PADING):
                    y_sample = y[:,i-self.SPLIT_LENGTH:i]
                    y_split.append(y_sample)
                    self.sample_number = self.sample_number + 1
                #y_split = np.array(y_split,np.float32)
                #y_ind = [i for i in range(self.SPLIT_LENGTH, y.shape[1], self.SPLIT_LENGTH)]
                #y_split = np.split(y, y_ind, axis=1)[:-1]
                for ysx in y_split:
                    self.brain_set.append([ysx, label, filename])

                #else:
                #    self.brain_set.append([y, label, filename])

            # Normalize data:
            #self.__normalize()
            # Save to pickled files for later use:
            with open(pickle_path, "wb") as pickle_file:
                pickle.dump(self.brain_set, pickle_file)
                if is_trainset:
                    print("Train ", end="")
                else:
                    print("Test ", end="")
                print("dataset normalized and saved")

        # Load normalized data, that was previously pickled
        else:
            with open(pickle_path, "rb") as pickle_file:
                self.brain_set = pickle.load(pickle_file)

        # Shuffle the data
        random.shuffle(self.brain_set)

    def __normalize(self):
        # Data per channel normalisation:
        sample_count = self.sample_number #self.SPLIT_LENGTH * len(self.brain_set)
        for channel in range(21):
            tmp_sum = 0.0

            for y, _, _ in self.brain_set:
                tmp_sum += sum(y[channel])
            mean = tmp_sum / sample_count
            tmp_sum = 0.0

            for y, _, _ in self.brain_set:
                tmp_sum += sum([(sample_y_val - mean) ** 2 for sample_y_val in y[channel]])
            std = math.sqrt(tmp_sum / sample_count)

            for i, (y, _, _) in enumerate(self.brain_set):
                self.brain_set[i][0][channel] = [(old_val - mean) / std for old_val in y[channel]]

    def __normalize2(self,data):
        # Data per channel normalisation:
        sample_count = data.shape[1]
        for channel in range(data.shape[0]):


            mean = sum(data[channel]) / sample_count

            tmp_sum = 0.0
            for y in data.T:
                tmp_sum += (y[channel] - mean) ** 2
            std = math.sqrt(tmp_sum / sample_count)

            for i, y in enumerate( data.T):
                data[channel][i] = (y[channel] - mean) / std


    def __len__(self):
        return len(self.brain_set)

    def __getitem__(self, idx):
        return self.brain_set[idx]
