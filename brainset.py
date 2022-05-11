import os
import random
import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "data/mentalload"

classes = {
    1: 0,
    2: 1,
}


def select_train_files(size=4):
    files_idx = list(range(0,30)) #random.sample(range(0, 35), size)
    result = []
    for idx in files_idx:
        result.append(f"Subject{idx}_1.edf")
        result.append(f"Subject{idx}_2.edf")
    return result


class Brainset(Dataset):
    y_length = 30477
    split_data_flag = True
    split_amount = 200

    def __init__(self, path, is_train_pretty):
        files = os.listdir(path)
        self.brain_set = []
        self.is_train_pretty = type

        test_files = select_train_files(size=4)
        train_files = [x for x in files if x not in test_files]

        files = train_files if self.is_train_pretty else test_files

        for index, filename in enumerate(files):
            class_idx = int(filename.split('_')[-1][0])
            file = os.path.join(DATA_PATH, filename)
            data = mne.io.read_raw_edf(file, verbose=False)
            raw_data = data.get_data()
            y = raw_data[:, :self.y_length].astype(np.float32)
            label = np.float32(classes[class_idx])

            if self.split_data_flag is True:
                y_ind = [i for i in range(self.split_amount,y.shape[1], self.split_amount)]
                y_split = np.split(y,y_ind,axis=1)[:-1]
                for ysx in y_split :
                    self.brain_set.append([ysx, label, filename])
            else:
                self.brain_set.append([y, label, filename])

        random.shuffle(self.brain_set)
        # self.brain_set.set_format("torch", columns=21)

    def __len__(self):
        return len(self.brain_set)

    def __getitem__(self, idx):
        return self.brain_set[idx]


def loadData(single_batch_test=False):
    path = os.path.join(DATA_PATH)
    brainset_train = Brainset(path, True)
    brainset_test = Brainset(path, False)
    train_loader = DataLoader(brainset_train, batch_size=4, shuffle=True)
    test_loader = DataLoader(brainset_test, batch_size=2, shuffle=False)
    return train_loader, test_loader

