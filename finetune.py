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
from utils import Logger, Checkpoint
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Config(dict):
    def __init__(self, version):
        super().__init__()
        self['version'] = version
        self['device'] = 'cuda:0'
        self['multi_train'] = False
        if self['multi_train']:
            self['device_ids'] = [0, 1]
            self['train_batch'] = 64 * len(self['device_ids'])
        else:
            self['train_batch'] = 32
        self['valid_batch'] = 32
        self['train_data'] = "data/finetune/train.npy"
        self['valid_data'] = "data/finetune/valid.npy"
        self['input_len'] = 160
        self['output_len'] = 150
        self['n_token'] = 2000
        self['n_layer'] = 6
        self['sos_id'] = 1
        self['eos_id'] = 2
        self['pad_id'] = 0
        self['lr'] = 3e-5
        self['pretrain_model'] = "model_2_2.370066770303609.pt"
        self['start_epoch'] = 0
        self['n_epoch'] = self['start_epoch'] + 50
        self['model_dir'] = 'checkpoint/finetune/%d' % self['version']


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
    train_data = FinetuneDataset(config["train_data"])
    valid_data = FinetuneDataset(config["valid_data"])
    train_loader = DataLoader(train_data, batch_size=config["train_batch"],
                              shuffle=True, drop_last=False, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=config["valid_batch"],
                              shuffle=True, drop_last=False, num_workers=0)
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
    checkpoint = Checkpoint(finetune)
    # for _, parameter in finetune.encoder.named_parameters():
    #     parameter.requires_grad = False
    if config['multi_train']:
        finetune = torch.nn.DataParallel(finetune, device_ids=config["device_ids"])
    if config['pretrain_model'] is not None:
        checkpoint.load(config['pretrain_model'])
        print("load pretrain model successfully!")

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
            pred, target = pred[:, :-1], target[:, 1:]
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
        if epoch % 1 == 0:  # valid
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
            checkpoint.save(config["model_dir"] + "/model_{}_{}.pt".format(epoch, cider_score))
            finetune.train()
            logger.log("valid: cider score = %f of " % cider_score, list(cider_scores))
            writer.add_scalar("cider score", cider_score, epoch)
    writer.close()
    logger.close()


if __name__ == '__main__':
    train(Config(1))
