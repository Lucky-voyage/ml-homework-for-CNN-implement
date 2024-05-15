import json
import torch
from tqdm import tqdm
from utils.model import Model
from utils.data_loader import DataLoader
from utils.get_checkpoint import get_checkpoint

import yaml

with open("config.yml", "r") as f:
    data = yaml.safe_load(f)
TRAIN_DATA_DIR = data["TRAIN_DATA_DIR"]
TEST_DATA_DIR = data["TEST_DATA_DIR"]
LOAD_DIR = data["LOAD_DIR"]
RESULT_DIR = data["RESULT_DIR"]


def main(train=True):
    my_model = Model(10)
    checkpoint = get_checkpoint(LOAD_DIR).get()
    print(f"Load parameter state: {my_model.load(checkpoint)}")

    loader = DataLoader(path=TRAIN_DATA_DIR, train=train)

    right = 0
    wrong = 0
    for i in tqdm(range(loader.__len__())):
        data = loader.get_data()[0]
        label = data[0]
        inputs = torch.tensor(data[1]).unsqueeze(0).to(torch.float)

        output = my_model(inputs)
        _, class_idx = torch.max(output, 1)

        if label == int(class_idx):
            right += 1
        else:
            wrong += 1

    print(f'Right: {right}\nWrong: {wrong}\nAccuracy: {right / (wrong + right) * 100:.2f}%')


if __name__ == '__main__':
    main(train=True)
