import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SVM:

    def __init__(self, components, kernel='rbf', C=1.0):
        self.components = components
        self.model = svm.SVC(kernel=kernel, C=C, decision_function_shape='ovo')
        self.accuracy = 0.
        self.precision = 0.
        self.recall = 0.
        self.f1 = 0.

    def __call__(self, X_train, X_test, y_train, y_test):
        self.pca = PCA(n_components=self.components).fit(X_train)
        X_train_pca = self.pca.fit_transform(X_train)
        self.model.fit(X_train_pca, y_train)
        print('\nTrain fitting Completed')

        X_test_pca = self.pca.fit_transform(X_test)
        y_pred = self.model.predict(X_test_pca)
        print('Test fitting Completed')

        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, average='macro')
        self.recall = recall_score(y_test, y_pred, average='macro')
        self.f1 = f1_score(y_test, y_pred, average='macro')

    def evaluate(self):
        print('Accuracy: ', self.accuracy)
        print('Precision: ', self.precision)
        print('Recall: ', self.recall)
        print('F1: ', self.f1)


def findBestComponents(X_train, X_test, y_train, y_test, num=20, least=0.2, most=0.4, kernel='rbf', C=1.0):
    n_components = []
    accuracies = []
    for components in np.linspace(least, most, num=num, endpoint=False):
        model = SVM(components, kernel=kernel, C=C)
        start = time.time()
        model(X_train, X_test, y_train, y_test)
        end = time.time()
        print(f'Number Components: {components:.2f}\t Time: {end - start:.2f}s')
        print(f'Accuracy: {model.accuracy * 100:.2f}%\t Precision: {model.precision * 100: .2f}%\t Recall: {model.recall * 100: .2f}%\t F1: {model.f1 * 100: .2f} ')
    ans = n_components[accuracies.index(max(accuracies))]
    print(f'Best Number for Components: {ans}')

    plt.plot(n_components, accuracies, '-o')
    plt.xlabel('n_components')
    plt.ylabel('accuracy')
    plt.show()

