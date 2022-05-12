from pyrsistent import b
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
        c_allchannels.append(cA1)
        c_allchannels.append(cD1)
    return c_allchannels



DATA_PATH = "data/mentalload"

path = os.path.join(DATA_PATH)
train = Brainset(path, True).brain_set
test = Brainset(path, False).brain_set

# brainset_dict = {}
# brainset_dict.update('data_train': brainset_train.brain_set[])

data_test = [list[0] for list in test]
data_train = [list[0] for list in train]

label_test = [list[0] for list in test]
label_train = [list[0] for list in train]

data_transformed = np.zeros((np.shape(data_train)[0],21*2))
print(data_transformed[1])
# i = 0
# for sample in data_train:
#     data_transformed[i, :] = EEG_to_dwt(sample)
#     i = i + 1

print(data_transformed)
coeffs = pywt.dwt(data_train[0][1], 'db1')
cA2, cD2 = coeffs
print(np.size(cA2))
print(np.size(cD2))

#lda = LinearDiscriminantAnalysis(n_components=21)
#lda.fit(data_train, label_train)

# clf = svm.SVC(kernel= 'rbf')
#clf.fit(data_train, label_train)
#label_pred = clf.predict(data_test)

#print("Miara accuracy:",metrics.accuracy_score(label_test, label_pred))
#y_score = clf.decision_function(X_test)
