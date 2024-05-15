import csv

import numpy as np
import matplotlib.pyplot as plt


class Evaluate:

    def __init__(self, pred, labels, classes, weight=None):
        self.pred = pred
        self.labels = labels
        self.len = len(labels)
        if weight is None:
            self.weight = [1 for i in range(self.len)]

        # save all scores
        self.res = {}
        self.precision = 0.
        self.recall = 0.
        self.classes = classes
        for item in classes:
            self.res[item] = {
                'precision': 0.,
                'recall': 0.
            }

        self.a = self.accuracy()
        self.eval()
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

    def accuracy(self):
        right = 0
        for i in range(self.len):
            if self.labels[i] == self.pred[i]:
                right += 1
        return right / self.len

    def eval(self):
        for item in self.classes:
            pred_pos = 0
            precision_pos = 0
            recall_pos = 0
            for i in range(self.len):
                if self.labels[i] == item and self.pred[i] == item:
                    pred_pos += 1
                if self.pred[i] == item:
                    precision_pos += 1
                if self.labels[i] == item:
                    recall_pos += 1

            if precision_pos == 0:
                self.res[item]['precision'] = 0
            else:
                self.res[item]['precision'] = pred_pos / precision_pos

            if recall_pos == 0:
                self.res[item]['recall'] = 0
            else:
                self.res[item]['recall'] = recall_pos / recall_pos

        for idx, item in enumerate(self.classes):
            self.precision += self.res[item]['precision'] * self.weight[idx]
            self.recall += self.res[item]['recall'] * self.weight[idx]
        self.precision /= len(self.classes)
        self.recall /= len(self.classes)

    def get_recall(self):
        return self.recall

    def get_precision(self):
        return self.precision

    def get_f1(self):
        return self.f1

    def get_accuracy(self):
        return self.a

    def get_map(self):
        ans = np.zeros((len(self.classes), len(self.classes)))
        for i in range(self.len):
            ans[self.pred[i], self.labels[i]] += 1

        plt.figure(figsize=(8, 6))
        plt.imshow(ans, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.title(f'Confusion Matrix', fontsize=22)
        plt.show()


