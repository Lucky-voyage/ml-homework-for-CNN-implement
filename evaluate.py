import torch
import csv
from tqdm import tqdm
from utils.get_checkpoint import get_checkpoint
from utils.model import Model
from utils.data_loader import DataLoader
from utils.evaluate import Evaluate

import yaml

with open("config.yml", "r") as f:
    data = yaml.safe_load(f)
TRAIN_DATA_DIR = data["TRAIN_DATA_DIR"]
TEST_DATA_DIR = data["TEST_DATA_DIR"]
LOAD_DIR = data["LOAD_DIR"]
RESULT_DIR = data["RESULT_DIR"]


def main():
    model = Model(10)
    checkpoint = get_checkpoint(LOAD_DIR).get()
    print(f"Loading checkpoint: {model.load(checkpoint)}")
    loader = DataLoader(TRAIN_DATA_DIR)

    # get answer
    pred = []
    labels = []
    for i in tqdm(range(loader.__len__())):
        data = loader.get_data()[0]
        labels.append(data[0])
        inputs = torch.tensor(data[1]).unsqueeze(0).to(torch.float)

        output = model(inputs)
        _, class_idx = torch.max(output, 1)
        pred.append(class_idx)

    train_res = [['true label', 'predicted label']]
    for i in range(len(pred)):
        train_res.append([labels[i], int(pred[i])])

    with open(RESULT_DIR, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(train_res)

    eval = Evaluate(pred, labels, [i + 1 for i in range(10)])
    print(f"Accuracy: {eval.get_accuracy()}")
    print(f"Precision: {eval.get_precision()}")
    print(f"Recall: {eval.get_recall()}")
    print(f"F1 score: {eval.get_f1()}")
    eval.get_map()


if __name__ == '__main__':
    main()
