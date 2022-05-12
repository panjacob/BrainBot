
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import svm
from brainset import *
from model import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pywt

def EEG_to_dwt(data):
    c_allchannels = []
    for channel in data:
        coeffs = pywt.dwt(channel, 'db1')
        cA1, cD1 = coeffs
        for element in cA1:
            c_allchannels.append(element)
        for element in cD1:
            c_allchannels.append(element)
    #print(c_allchannels)
    return c_allchannels



DATA_PATH = "data/mentalload/raw"

path = os.path.join(DATA_PATH)
train = Brainset(path, True, pickled=True).brain_set
test = Brainset(path, False, pickled=True).brain_set

data_test = [list[0] for list in test]
data_train = [list[0] for list in train]

label_test = [list[1] for list in test]
label_train = [list[1] for list in train]

data_train_transformed = []
data_test_transformed = []

for index, sample in enumerate(data_train):
    data_train_transformed.append([element for element in EEG_to_dwt(sample)])
for index, sample in enumerate(data_test):
    data_test_transformed.append([element for element in EEG_to_dwt(sample)])


# lda = LinearDiscriminantAnalysis()
# lda.fit(data_train_transformed, label_train)
# label_pred = lda.predict(data_test_transformed)
# data_train_reduced = lda.transform(data_train_transformed)

# lda.fit(data_test_transformed, label_test)
# data_test_reduced = lda.transform(data_test_transformed)

clf = svm.SVC(kernel= 'rbf')
clf.fit(data_train_transformed, label_train)
label_pred = clf.predict(data_test_transformed)

print("Miara accuracy:",metrics.accuracy_score(label_test, label_pred))
#y_score = clf.decision_function(X_test)
