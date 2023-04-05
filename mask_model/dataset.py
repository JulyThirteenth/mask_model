import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PretrainDataset(Dataset):
    def __init__(self, input_path: str, test_input_path, output_path: str, shuffle=False):
        super().__init__()
        self.data = np.vstack((np.load(input_path, allow_pickle=True),
                               np.load(test_input_path, allow_pickle=True),
                               np.load(output_path, allow_pickle=True)))
        if shuffle:
            np.random.shuffle(self.data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return np.array(self.data[item, 0]), np.array(self.data[item, 1]), np.array(self.data[item, 2])


class FinetuneDataset(Dataset):
    def __init__(self, data_file: str, shuffle: bool = False):
        super().__init__()
        self.data = np.load(data_file, allow_pickle=True)
        if shuffle:
            np.random.shuffle(self.data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return np.array(self.data[item, 0]), np.array(self.data[item, 1])


if __name__ == "__main__":
    pretrain = PretrainDataset("data/pretrain/data_in.npy", "data/pretrain/data_in_test.npy",
                               "data/pretrain/data_out.npy")
    pretrain_loader = DataLoader(pretrain, batch_size=32, shuffle=True, drop_last=False)
    (source, target, position) = next(iter(pretrain_loader))
    print("source shape: {}".format(source.shape), "target shape: {}".format(target.shape),
          "position shape: {}".format(position.shape), sep='\n')
    finetune = FinetuneDataset("data/finetune/data_trans.npy")
    finetune_loader = DataLoader(finetune, batch_size=32, shuffle=True, drop_last=False)
    (source, target) = next(iter(finetune_loader))
    print("source shape: {}".format(source.shape), "target shape: {}".format(target.shape), sep='\n')
