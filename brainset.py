import os
import random
from pprint import pprint

import mne
import torch
from torch.utils.data import Dataset, DataLoader

classes = {
    1: "False",
    2: 'True',
}


def select_train_files(size=4):
    files_idx = random.sample(range(0, 35), size)
    result = []
    for idx in files_idx:
        result.append(f"Subject{idx}_1.edf")
        result.append(f"Subject{idx}_2.edf")
    return result


class Brainset(Dataset):

    def __init__(self, path, type):
        files = os.listdir(path)
        self.brain_set = []
        self.type = type

        test_files = select_train_files(size=4)
        train_files = [x for x in files if x not in test_files]

        files = train_files if self.type else test_files

        for index, filename in enumerate(files):
            class_idx = int(filename.split('_')[-1][0])
            file = os.path.join('files2', filename)
            data = mne.io.read_raw_edf(file)
            raw_data = data.get_data()
            y = raw_data[:, :30000]

            self.brain_set.append([y, classes[class_idx], filename])

        random.shuffle(self.brain_set)
        # self.brain_set.set_format("torch", columns=21)

    def __len__(self):
        return len(self.brain_set)

    def __getitem__(self, idx):
        return self.brain_set[idx]


def loadData():
    path = os.path.join('files2')
    brainset = Brainset(path)
    brainloader = DataLoader(brainset, batch_size=8, shuffle=True)
    testloader = DataLoader(brainset, batch_size=2, shuffle=Talse)
    return brainloader , testloader


brainloader, testloader = loadData()
device = torch.device("cpu")
# Szkielet pętli treningowej!
for batch in brainloader:
    inputs = batch[0]
    labels = batch[1]
    filenames = batch[2]
