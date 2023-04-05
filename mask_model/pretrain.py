import time
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from dataset import PretrainDataset
from model import Transformer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


class Config(dict):
    def __init__(self, version):
        super().__init__()
        self['version'] = version
        self['device'] = 'cuda:0'
        self['multi_train'] = False
        if self['multi_train']:
            self['device_ids'] = [0, 1, 2, 3]
            self['batch_size'] = 64 * len(self['device_ids'])
        else:
            self['batch_size'] = 32
        self['prefix_len'] = 10
        self['input_data'] = "data/pretrain/data_in.npy"
        self['test_input_data'] = "data/pretrain/data_in_test.npy"
        self['output_data'] = "data/pretrain/data_out.npy"
        self['input_len'] = 160
        self['output_len'] = 150
        self['n_token'] = 2000
        self['n_layer'] = 6
        self['sos_id'] = 1
        self['eos_id'] = 2
        self['pad_id'] = 0
        self['lr'] = 3e-5
        self['pretrain_model'] = "model_27_1.420446366071701.pt"
        self['start_epoch'] = 28
        self['n_epoch'] = self['start_epoch'] + 50
        self['model_dir'] = 'checkpoint/pretrain/%d' % self['version']


def array2str(arr, sos_id, eos_id, pad_id):
    out = ""
    for i in range(len(arr)):
        if arr[i] == pad_id or arr[i] == eos_id:
            break
        if arr[i] == sos_id:
            continue
        out += str(int(arr[i])) + " "
    if len(out.strip()) == 0:
        out = "0"
    return out.strip()


def train(config):
    train_data = PretrainDataset(config['input_data'], config['test_input_data'], config['output_data'])
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, drop_last=False)

    pretrain = Transformer(
        input_l=config['input_len'],
        output_l=config['output_len'],
        n_token=config['n_token'],
        encoder_layer=config['n_layer'],
        decoder_layer=config['n_layer'],
        sos_id=config['sos_id'],
        pad_id=config['pad_id']
    ).to(config['device'])
    if config['pretrain_model'] is not None:
        pretrain.load_state_dict(torch.load(config['pretrain_model']), strict=False)
        print("load pretrain model successfully!")
    if config['multi_train']:
        pretrain = torch.nn.DataParallel(pretrain, device_ids=config["device_ids"])

    optim = Adam(pretrain.parameters(), lr=3e-5)
    loss_func = CrossEntropyLoss(ignore_index=0).to(config['device'])

    Path(config["model_dir"]).mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(config["model_dir"])
    logger = Logger(config["model_dir"] + "/log.txt", "w")
    logger.log(config)

    best_loss = np.inf
    for epoch in range(config["start_epoch"], config["n_epoch"]):  # train
        losses = []
        process_bar = tqdm(train_loader)
        for idx, (source, target, position) in enumerate(process_bar):
            source, target, position = source.to(config['device']), target.to(config['device']), \
                position.to(config['device'])
            pred = pretrain(source, target)*position.unsqueeze(dim=-1)
            pred = pred.reshape(-1, pred.shape[-1])
            target = target.reshape(-1)
            loss = loss_func(pred, target.long())
            optim.zero_grad()  # 清空梯度
            loss.backward()
            optim.step()  # 优化一次
            losses.append(loss.item())
            process_bar.set_postfix(
                epoch=epoch, idx=idx, loss="%.3f" % float(np.mean(losses))
            )
        mean_loss = np.mean(losses)
        logger.log("epoch: %d:" % epoch, "mean loss: %f of " % mean_loss, losses)
        writer.add_scalar("batch loss", mean_loss, epoch)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                pretrain.state_dict(),
                config["model_dir"] + "/model_{}_{}.pt".format(epoch, best_loss),
            )
    writer.close()
    logger.close()


if __name__ == '__main__':
    train(Config(version=1))
