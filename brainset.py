"""
This is a script with Dataset class
and loading data function
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

DATA_PICKLED =  True # If selected will pull data form pickle files

classes = {
    1: 0,
    2: 1,
}


def select_train_files(size=4):
    files_idx = list(range(0,30)) #random.sample(range(0, 35), size)
    result = []
    for idx in files_idx:
        az = "0" if idx < 10 else "" #additional zero to print numbers like this: 00 01 09 and 10 22 34 etc
        result.append("Subject"+az+str(idx)+"_1.edf")
        result.append("Subject"+az+str(idx)+"_2.edf")
    return result


class Brainset(Dataset):
    """
        This is dataset class, designed to load EEG signal data from edf files (or pickled files)
        and create usable with pytorch - dataset
    """
    #Parameters:
    y_length = 30477  #initial signal cut size
    split_data_flag = True  #split data to smaller batches
    split_amount = 200  # size of the smaller batch

    def __init__(self, path, is_train_pretty, pickled=False):
        #Data will be in this list:
        self.brain_set = []

        #Define if this is train or test dataset:
        if is_train_pretty:
            pickle_path = PICKLE_PATH_TRAIN
        else:
            pickle_path = PICKLE_PATH_TEST

        #If data was not normalized and saved before:
        if not pickled:
            #Load data from source (edf files)
            files = os.listdir(path)
            files = filter(lambda x: x.endswith(".edf"), files)
            #Split files to test and train files:
            test_files = select_train_files(size=4)
            train_files = [x for x in files if x not in test_files]
            #Set dataset files (either test or train)
            files = train_files if is_train_pretty else test_files

            #Get data signals form files:
            for index, filename in enumerate(files):
                #Extract labels:
                class_idx = int(filename.split('_')[-1][0])   # labels are saved inside file_name at the end after "_"
                label = np.float32(classes[class_idx])  # labels are shifted 0,1 <- 1,2 in files
                # Extract data:
                file = os.path.join(DATA_PATH, filename)
                data = mne.io.read_raw_edf(file, verbose=False)
                raw_data = data.get_data()
                #Cut data to unified size:
                y = raw_data[:, :self.y_length].astype(np.float32)

                #Check if there is a need to cut signals:
                if self.split_data_flag is True:
                    # Cut signals into small signals (to get more data and to fit it into neural net input):
                    y_ind = [i for i in range(self.split_amount,y.shape[1], self.split_amount)]
                    y_split = np.split(y,y_ind,axis=1)[:-1]
                    for ysx in y_split :
                        self.brain_set.append([ysx, label, filename])
                else:
                    self.brain_set.append([y, label, filename])

            #Normalize data:
            self.__normalize()
            #Save to pickled files for later use:
            with open(pickle_path, "wb") as pickle_file:
                pickle.dump(self.brain_set, pickle_file)
                if is_train_pretty:
                    print("Train ",end ="")
                else:
                    print("Test ",end ="")
                print("dataset normalized and saved")

        # When Data was normalized and saved to pickled files
        else:
            with open(pickle_path, "rb") as pickle_file:
                self.brain_set = pickle.load(pickle_file)

        # Mix up the data data
        random.shuffle(self.brain_set)

    def __normalize(self):
        # Data per channel normalisation:
        sample_count = self.split_amount * len(self.brain_set)
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

    def __len__(self):
        return len(self.brain_set)

    def __getitem__(self, idx):
        return self.brain_set[idx]


def loadData(single_batch_test=False):
    path = os.path.join(DATA_PATH)
    brainset_train = Brainset(path, is_train_pretty = True, pickled = DATA_PICKLED)
    brainset_test = Brainset(path, is_train_pretty = False, pickled = DATA_PICKLED)
    train_loader = DataLoader(brainset_train, batch_size=4, shuffle=True)
    test_loader = DataLoader(brainset_test, batch_size=2, shuffle=False)
    return train_loader, test_loader

