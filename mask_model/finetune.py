import time
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from dataset import FinetuneDataset
from evaluate import CiderD
from model import Transformer

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
        self['trans_data'] = "data/finetune/data_trans.npy"
        self['train_batch'] = 32
        self['valid_batch'] = 32
        self['input_len'] = 160
        self['output_len'] = 150
        self['n_token'] = 2000
        self['n_layer'] = 6
        self['sos_id'] = 1
        self['eos_id'] = 2
        self['pad_id'] = 0
        self['lr'] = 3e-5
        self['start_epoch'] = 0
        self['n_epoch'] = self['start_epoch'] + 50
        self['pretrain_model'] = None
        self['model_dir'] = 'checkpoint/finetune/%d' % self['version']


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
            self.fp = open(self.file_name, "w")

    def close(self):
        self.fp.close()


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
    # print(config)
    train_data = FinetuneDataset("train", config["trans_data"])
    valid_data = FinetuneDataset("valid", config["trans_data"])
    train_loader = DataLoader(train_data, batch_size=config["train_batch"] * len(config["device_ids"]),
                              shuffle=True, drop_last=False, num_workers=16)
    valid_loader = DataLoader(valid_data, batch_size=config["valid_batch"] * len(config["device_ids"]),
                              shuffle=True, drop_last=False, num_workers=16)
    # print(len(train_loader), len(valid_loader), sep='\n')

    finetune = Transformer(
        input_l=config['input_len'],
        output_l=config['output_len'],
        n_token=config['n_token'],
        encoder_layer=config['n_layer'],
        decoder_layer=config['n_layer'],
        sos_id=config['sos_id'],
        pad_id=config['pad_id']
    ).to(config['device'])
    # for _, parameter in finetune.encoder.named_parameters():
    #     parameter.requires_grad = False
    if config['multi_train']:
        finetune = torch.nn.DataParallel(finetune, device_ids=config["device_ids"])

    # optim = Adam(filter(lambda p: p.requires_grad, finetune.parameters()), lr=config["lr"]) # 只更新decoder部分参数
    optim = Adam(finetune.parameters(), lr=config["lr"])
    loss_func = CrossEntropyLoss()
    Path(config["model_dir"]).mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(config["model_dir"])
    logger = Logger(config["model_dir"] + "/log.txt", "w")
    logger.log(config)

    for epoch in range(config["start_epoch"], config["n_epoch"]):  # train
        losses = []
        process_bar = tqdm(train_loader)
        for idx, (source, target) in enumerate(process_bar):
            source, target = source.to(config["device"]), target.to(config["device"])
            # print(source.shape, target.shape, sep='\n')
            pred = finetune(source, target)
            pred = pred[:, :-1]
            target = target[:, 1:]
            loss = loss_func(
                pred.reshape(-1, pred.shape[-1]), target.reshape(-1).long()
            )
            optim.zero_grad()  # 清空梯度
            loss.backward()
            optim.step()  # 优化一次
            losses.append(loss.item())
            process_bar.set_postfix(
                epoch=epoch, idx=idx, loss="%.3f" % float(np.mean(losses))
            )
        logger.log("epoch: %d:" % epoch, "mean loss: %f of " % np.mean(losses), losses)
        writer.add_scalar("batch loss", np.mean(losses), epoch)
        if epoch % 5 == 0:  # valid
            finetune.eval()
            res, gts = [], {}
            tot = 0
            CiderD_scorer = CiderD(df="corpus", sigma=15)
            process_bar = tqdm(valid_loader)
            for idx, (source, target) in enumerate(process_bar):
                source = source.to(config["device"])
                pred = finetune(source)
                pred = pred.cpu().numpy()
                for i in range(pred.shape[0]):
                    res.append(
                        {
                            "image_id": tot,
                            "caption": [
                                array2str(
                                    pred[i],
                                    config["sos_id"],
                                    config["eos_id"],
                                    config["pad_id"],
                                )
                            ],
                        }
                    )
                    gts[tot] = [
                        array2str(
                            target[i],
                            config["sos_id"],
                            config["eos_id"],
                            config["pad_id"],
                        )
                    ]
                    tot += 1
                cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
                process_bar.set_postfix(epoch=epoch, idx=idx, CIDEr=cider_score)
            cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
            torch.save(
                finetune.state_dict(),
                config["model_dir"] + "/model_{}_{}.pt".format(epoch, cider_score),
            )
            finetune.train()
            logger.log("valid: cider score = %f of " % cider_score, list(cider_scores))
            writer.add_scalar("cider score", cider_score, epoch)
    writer.close()
    logger.close()


if __name__ == '__main__':
    train(Config(1))
