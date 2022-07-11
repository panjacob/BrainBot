PATH_PREFIX = "data/mentalload/"
DATA_PATH = PATH_PREFIX + "raw"
PICKLE_PATH_TRAIN = PATH_PREFIX + "/train.pickle"
PICKLE_PATH_TEST = PATH_PREFIX + "/test.pickle"
MEAN_STD_PATH = PATH_PREFIX + "/mean_std.txt"

DATA_PICKLED = False  # Enable if Data has been saved previously saved in pickled files

MAX_LENGTH = 30000

CLASSES = {
    1: 0,
    2: 1
}
