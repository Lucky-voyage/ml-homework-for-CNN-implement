import json
import torch
from pathlib import Path
from tqdm import tqdm
from utils.model import Model
from utils.data_loader import DataLoader
from utils.loss_function import CrossEntropyLoss
from utils.get_checkpoint import get_checkpoint

import yaml

with open("config.yml", "r") as f:
    data = yaml.safe_load(f)
TRAIN_DATA_DIR = data["TRAIN_DATA_DIR"]
TEST_DATA_DIR = data["TEST_DATA_DIR"]
LOAD_DIR = data["LOAD_DIR"]
RESULT_DIR = data["RESULT_DIR"]


def main(epoch=200, load=True, save=True, load_path=LOAD_DIR, save_path=LOAD_DIR):
    model = Model(10)
    loader = DataLoader(path=TRAIN_DATA_DIR, train=True)
    func = CrossEntropyLoss()

    if load:
        checkpoint = get_checkpoint(LOAD_DIR).get()
        print(f"Loading checkpoint: {model.load(checkpoint)}")

    for epoch in range(epoch):
        print(f"Epoch {epoch}:")
        for _ in tqdm(range(loader.__len__())):
            data = loader.get_data()[0]
            inputs = torch.tensor(data[1]).unsqueeze(0).to(torch.float32)
            label = torch.zeros(1, 10, 1)
            label[0, data[0], 0] = 1

            model.zero_grad()
            outputs = model(inputs)

            loss = func(outputs, label)
            model.backward(func.backward())

        if save:
            if model.save(save_path):
                print("Model saved successfully\n")


if __name__ == '__main__':
    main(load=False, save=False)
