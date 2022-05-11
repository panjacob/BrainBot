import os
import random
import mne
import math
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
            y = raw_data[:, :30477].astype(np.float32)
            label = np.float32(classes[class_idx])
            # y = np.full(10, 10).astype(np.double)
            #y2 = np.array(y).astype(np.float)

            self.brain_set.append([y, label, filename])

        sample_count = 30477 * len(self.brain_set)
        for channel in range(21):
            tmp_sum = 0.0

            for y, _, _ in self.brain_set:
                tmp_sum += sum(y[channel])
            mean = tmp_sum / sample_count

            tmp_sum = 0.0

            for y, _, _ in self.brain_set:
                tmp_sum += sum([(sample_y_val - mean)**2 for sample_y_val in y[channel]])
            std = math.sqrt(tmp_sum / sample_count)

            for i, (y, _, _) in enumerate(self.brain_set):
                self.brain_set[i][0][channel] = [(old_val - mean) / std for old_val in y[channel]]

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
