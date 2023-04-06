import time
from pathlib import Path

import torch


class Logger:
    def __init__(self, file_name, mode="w", buffer=100):
        Path(file_name).parent.mkdir(exist_ok=True, parents=True)
        self.file_name = file_name
        self.fp = open(file_name, mode)
        self.cnt = 0
        self.stamp = time.time()
        self.buffer = buffer

    def log(self, *args, end="\n"):
        for x in args:
            if isinstance(x, dict):
                for y in x:
                    self.fp.write(str(y) + ":" + str(x[y]) + " ")
            else:
                self.fp.write(str(x) + " ")
        self.fp.write(end)
        self.cnt += 1
        if self.cnt >= self.buffer or time.time() - self.stamp > 5:
            self.cnt = 0
            self.stamp = time.time()
            self.fp.close()
            self.fp = open(self.file_name, "a")

    def close(self):
        self.fp.close()


class Checkpoint:
    def __init__(self, model):
        self.model = model

    def load(self, model_path):
        memory = torch.load(model_path)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(memory)
        else:
            self.model.load_state_dict(memory)

    def save(self, save_path):
        if isinstance(self.model, torch.nn.DataParallel):
            ret = self.model.module.state_dict()
        else:
            ret = self.model.state_dict()
        torch.save(ret, save_path)