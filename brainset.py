import os
import random
import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

classes = {
    1: 0.0,
    2: 1.0,
}


def select_train_files(size=4):
    files_idx = random.sample(range(0, 35), size)
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
            file = os.path.join('files2', filename)
            data = mne.io.read_raw_edf(file, verbose=False)
            raw_data = data.get_data()
            y = raw_data[:, :30000]
            # y = np.full(10, 10).astype(np.double)
            y2 = np.array(y).astype(np.float)

            self.brain_set.append([y2, classes[class_idx], filename])

        random.shuffle(self.brain_set)
        # self.brain_set.set_format("torch", columns=21)

    def __len__(self):
        return len(self.brain_set)

    def __getitem__(self, idx):
        return self.brain_set[idx]


def loadData():
    path = os.path.join('files2')
    brainset_train = Brainset(path, True)
    brainset_test = Brainset(path, False)
    train_loader = DataLoader(brainset_train, batch_size=8, shuffle=True)
    test_loader = DataLoader(brainset_test, batch_size=2, shuffle=False)
    return train_loader, test_loader

# brainloader, testloader = loadData()
# device = torch.device("cpu")
# # Szkielet pętli treningowej!
# for batch in brainloader:
#     inputs = batch[0]
#     labels = batch[1]
#     filenames = batch[2]
