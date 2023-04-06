import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from model import Transformer

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class InferDataset(Dataset):
    def __init__(self, data_file: str, trans_prefix=None, input_len: int = 160, pad_id: int = 0):
        super().__init__()
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            self.samples = [row for row in reader]
        fp.close()
        if trans_prefix is None:
            trans_prefix = [3] * 10
        self.input_len = input_len
        self.pad_id = pad_id
        self.trans_prefix = trans_prefix

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        source = [int(x) for x in self.samples[idx][1].split()]
        source = self.trans_prefix + source
        if len(source) < self.input_len:
            source.extend([self.pad_id] * (self.input_len - len(source)))
        return np.array(source)


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


def inference(model_file, data_file, input_len=160, output_len=150,
              sos_id=1, eos_id=2, pad_id=0,
              n_token=2000, encoder_layer=6, decoder_layer=6):
    test_data = InferDataset(data_file)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, drop_last=False)
    print("Get loader successfully!")
    trans_model = Transformer(input_len, output_len, n_token, encoder_layer, decoder_layer).cuda()
    trans_model.load_state_dict(torch.load(model_file))
    trans_model.eval()
    fp = open(Path(model_file).parent / "pred.csv", "w", newline="")
    writer = csv.writer(fp)
    tot = 0
    for source in tqdm(test_loader):
        source = source.to("cuda")
        pred = trans_model(source)
        pred = pred.cpu().numpy()
        for i in range(pred.shape[0]):
            writer.writerow(
                [
                    tot,
                    array2str(
                        pred[i],
                        sos_id,
                        eos_id,
                        pad_id,
                    ),
                ]
            )
            tot += 1
    fp.close()


if __name__ == "__main__":
    inference(model_file="model_2_2.370066770303609.pt",
              data_file="data/test.csv")
