import csv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class DicisionTree:

    def __init__(self, max_depth):
        self.clf = DecisionTreeClassifier(max_depth=max_depth)
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def __call__(self, X_train, X_test, y_train, y_test):
        self.clf.fit(X_train, y_train)
        print("Fit Completed")
        y_pred = self.clf.predict(X_test)
        print("Predict Completed")
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, average='macro')
        self.recall = recall_score(y_test, y_pred, average='macro')
        self.f1 = f1_score(y_test, y_pred, average='macro')

        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def getBestDepth(X_train, X_test, y_train, y_test, least=50, most=100, stride=1):
    d = []
    a = []
    p = []
    r = []
    f1 = []
    l = [i for i in range(least, most, stride)]
    for depth in l:
        model = DicisionTree(max_depth=depth)
        model(X_train, X_test, y_train, y_test)
        print(f'\nDepth: {depth}')
        print(f'Accuracy: {model.accuracy * 100:.2f}%\t Precision: {model.precision * 100: .2f}%\t Recall: '
              f'{model.recall * 100: .2f}%\t F1: {model.f1 * 100: .2f} ')
        d.append(depth)
        a.append(model.accuracy)
        p.append(model.precision)
        r.append(model.recall)
        f1.append(model.f1)

    plt.figure(figsize=(10, 6))

    plt.plot(a, d, label='Accuracy')
    plt.plot(p, d, label='Precision')
    plt.plot(f1, d, label='Recall')
    plt.plot(r, d, label='F1 Score')
    plt.legend()

    plt.title('Decision Tree with Different Depth')
    plt.xlabel('Max Depth')
    plt.show()

