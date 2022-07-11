from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from brainset import *
from sklearn import metrics
import pickle
from datetime import datetime

INPUT_PATH_TRAIN = "data/mentalload/svm/svm_train.npy"
INPUT_PATH_TRAIN_LABELS = "data/mentalload/svm/svm_train_labels.pkl"
INPUT_PATH_TEST = "data/mentalload/svm/svm_test.npy"
INPUT_PATH_TEST_LABELS = "data/mentalload/svm/svm_test_labels.pkl"

if __name__ == '__main__':
    data_train = np.load(INPUT_PATH_TRAIN, mmap_mode=None)
    data_test = np.load(INPUT_PATH_TEST, mmap_mode=None)

    np.random.shuffle(data_train)
    np.random.shuffle(data_test)

    with open(INPUT_PATH_TRAIN_LABELS, "rb") as train_file_labels:
        label_train = pickle.load(train_file_labels)
        label_train = np.array(label_train)
        label_train[label_train == 0.0] = -1.0

    with open(INPUT_PATH_TEST_LABELS, "rb") as test_file_labels:
        label_test = pickle.load(test_file_labels)
        label_test = np.array(label_test)
        label_test[label_test == 0.0] = -1.0

    lda = LinearDiscriminantAnalysis()
    lda.fit(data_train, label_train)
    #with open("./models/Model_LDA_05.07.2022_07:37.pkl", "rb") as lda_file:
    #    lda = pickle.load(lda_file)
    data_train_reduced = lda.transform(data_train)
    data_test_reduced = lda.transform(data_test)

    clf = svm.SVC(kernel='rbf', C=0.1)
    clf.fit(data_train_reduced, label_train)
    #with open("./models/Model_SVM_05.07.2022_07:37.pkl", "rb") as svm_file:
    #    clf = pickle.load(svm_file)
    label_pred = clf.predict(data_test_reduced)

    print("Miara accuracy:", metrics.accuracy_score(label_test, label_pred))
    # y_score = clf.decision_function(X_test)

    models_postfix = datetime.now().strftime("%d.%m.%Y_%H:%M") + ".pkl"

    # Save SVM model
    save_path = "models/Model_SVM_" + models_postfix
    with open(save_path, 'wb') as save_file:
        pickle.dump(clf, save_file)

    # Save LDA model
    save_path = "models/Model_LDA_" + models_postfix
    with open(save_path, 'wb') as save_file:
        pickle.dump(lda, save_file)
