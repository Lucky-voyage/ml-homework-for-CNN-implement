from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

import csv
import torch
from PIL import Image
import numpy as np

'''
    about data.csv
    
    Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. 
    
    This pixel-value is an integer between 0 and 255, inclusive.
    
    The training data set, (train.csv), has 785 columns. 
    The first column, called "label", is the digit that was drawn by the user. 
    
    Each pixel column in the training set has a name like pixel_x, where x is an integer between 0 and 783, inclusive. 
    To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, 
    where i and j are integers between 0 and 27, inclusive. 
    Then pixel_x is located on row i and column j of a 28 x 28 matrix, (indexing by zero).
'''


class DataLoader:

    def __init__(self, path, save_path="/data", train=True):
        self.path = path
        self.save_path = Path(save_path)
        self.data = []
        self.read_csv(path, train)
        self.iter = enumerate(self.data)  # construct a iter for data(list)

    def read_csv(self, path, train: bool):
        with open(path, "r") as file:
            reader = csv.reader(file)
            next(reader)

            for idx, row in enumerate(reader):
                if train:       # for train dataset
                    label = int(row[0])
                    row = [int(i) for i in row[1:]]
                    img = np.array([row]).reshape(28, 28)
                    self.data.append((label, img))
                else:           # for test dataset, which has no field 'label'
                    row = [int(i) for i in row]
                    img = np.array([row]).reshape(28, 28)
                    self.data.append(img)

    def get_data(self, batch_size=1):
        ret = []
        for i in range(batch_size):
            try:
                ret.append(self.iter.__next__()[1])
            except StopIteration:
                self.iter = enumerate(self.data)  # loop
                ret.append(self.iter.__next__()[1])
                return ret
        return ret

    def __len__(self):
        return self.data.__len__()
