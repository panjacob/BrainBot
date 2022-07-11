import gc

from BrainStudy.brainset import *
from BrainStudy.model import *
import pywt
import pickle


DATA_PATH = "data/mentalload/raw"
OUTPUT_PATH_TRAIN = "data/mentalload/svm/svm_train.npy"
OUTPUT_PATH_TRAIN_LABELS = "data/mentalload/svm/svm_train_labels.pkl"
OUTPUT_PATH_TEST = "data/mentalload/svm/svm_test.npy"
OUTPUT_PATH_TEST_LABELS = "data/mentalload/svm/svm_test_labels.pkl"


def eeg_signal_to_dwt(data):
    c_allchannels = np.empty(0)
    for channel in data:
        ca1, cd1 = pywt.dwt(channel, 'db1')
        c_allchannels = np.append(c_allchannels, ca1)
        c_allchannels = np.append(c_allchannels, cd1)
    return c_allchannels


if __name__ == '__main__':
    path = os.path.join(DATA_PATH)
    train = Brainset(path, is_trainset=True, load_pickled=False)
    # Test set should be normalized with TRAINING mean and std, just like real data:
    test = Brainset(path, is_trainset=False, load_pickled=False, mean=train.mean, std=train.std).brain_set
    train = train.brain_set

    data_size = train[0][0].shape

    data_test = [test_list[0] for test_list in test]
    data_train = [train_list[0] for train_list in train]

    label_test = [test_list[1] for test_list in test]
    label_train = [train_list[1] for train_list in train]

    del test
    del train

    data_train_transformed = np.empty((len(data_train), data_size[0] * data_size[1]), dtype="float32")
    data_test_transformed = np.empty((len(data_test), data_size[0] * data_size[1]), dtype="float32")

    print("Starting...")

    for index, sample in enumerate(data_test):
        data_test_transformed[index, :] = eeg_signal_to_dwt(sample)
        data_test[index] = 0

    del data_test

    print("Writing to files")

    with open(OUTPUT_PATH_TEST, "wb", buffering=0) as test_file:
        np.save(test_file, data_test_transformed, allow_pickle=False)

    del data_test_transformed

    with open(OUTPUT_PATH_TEST_LABELS, "wb") as test_file_labels:
        pickle.dump(label_test, test_file_labels)

    del label_test

    gc.collect()

    print("Starting...")

    for index, sample in enumerate(data_train):
        data_train_transformed[index, :] = eeg_signal_to_dwt(sample)
        data_train[index] = None

    del data_train

    print("Writing to files")

    with open(OUTPUT_PATH_TRAIN, "wb", buffering=0) as train_file:
        np.save(train_file, data_train_transformed, allow_pickle=False)

    del data_train_transformed

    with open(OUTPUT_PATH_TRAIN_LABELS, "wb") as train_file_labels:
        pickle.dump(label_train, train_file_labels)

    del label_train
