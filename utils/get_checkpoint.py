import json
import torch


class get_checkpoint:

    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            self.checkpoint = json.load(f)[0]
            self.convert(self.checkpoint)

    @staticmethod
    def convert(data):
        for key, value in data.items():
            if isinstance(value, dict):
                get_checkpoint.convert(value)
            else:
                data[key] = torch.tensor([float(value[i]) for i in range(len(value))])

    def get(self):
        return self.checkpoint
