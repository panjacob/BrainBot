from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import svm
from brainset import *
from model import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pywt
from datetime import datetime

DATA_PATH = "data/mentalload/raw"


def eeg_signal_to_dwt(data):
    c_allchannels = []
    for channel in data:
        ca1, cd1 = pywt.dwt(channel, 'db1')
        for element in ca1:
            c_allchannels.append(element)
        for element in cd1:
            c_allchannels.append(element)
    return c_allchannels


path = os.path.join(DATA_PATH)
train = Brainset(path, True, load_pickled=True).brain_set
test = Brainset(path, False, load_pickled=True).brain_set

data_test = [test_list[0] for test_list in test]
data_train = [train_list[0] for train_list in train]

label_test = [test_list[1] for test_list in test]
label_train = [train_list[1] for train_list in train]

data_train_transformed = []
data_test_transformed = []

for index, sample in enumerate(data_train):
    data_train_transformed.append([element for element in eeg_signal_to_dwt(sample)])
for index, sample in enumerate(data_test):
    data_test_transformed.append([element for element in eeg_signal_to_dwt(sample)])

lda = LinearDiscriminantAnalysis()
lda.fit(data_train_transformed, label_train)
data_train_reduced = lda.transform(data_train_transformed)

lda.fit(data_test_transformed, label_test)
data_test_reduced = lda.transform(data_test_transformed)

clf = svm.SVC(kernel='rbf', C=0.01)
clf.fit(data_train_reduced, label_train)
label_pred = clf.predict(data_test_reduced)

print("Miara accuracy:", metrics.accuracy_score(label_test, label_pred))
# y_score = clf.decision_function(X_test)

#save model
save_path = "models/Model_SVM_" + datetime.now().strftime("%d.%m.%Y_%H:%M") + ".pkl"
with open(save_path,'wb') as save_file:
    pickle.dump(clf,save_file)
