import csv
import random
import numpy as np


class Preprocess:
    def __init__(self, mask_id: int = 9, mask_prob: float = 0.15,
                 input_prefix=None, output_prefix=None, trans_prefix=None,
                 input_len: int = 160, output_len: int = 150,
                 pad_id: int = 0, sos_id: int = 1, eos_id: int = 2):
        if input_prefix is None:
            input_prefix = [1] * 10  # 预测输入mask任务的prefix
        if output_prefix is None:
            output_prefix = [2] * 10  # 预测输出mask任务的prefix
        if trans_prefix is None:
            trans_prefix = [3] * 10  # 翻译任务的prefix
        self.__mask_id = mask_id
        self.__mask_prob = mask_prob
        self.__input_prefix = input_prefix
        self.__output_prefix = output_prefix
        self.__trans_prefix = trans_prefix
        self.__input_len = input_len
        self.__output_len = output_len
        self.__pad_id = pad_id
        self.__sos_id = sos_id
        self.__eos_id = eos_id

    def for_finetune(self, data_file: str, train_perc: float = 0.9, shuffle=True):
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            samples = [row for row in reader]
        fp.close()
        if shuffle:  # 打乱原始数据
            random.shuffle(samples)
        '''
        构造finetune训练数据集
        '''
        train_data = []
        for sample in samples[:int(train_perc * len(samples))]:
            source = [int(x) for x in sample[1].split()]
            target = [self.__sos_id] + [int(x) for x in sample[2].split()] + [self.__eos_id]
            if len(target) < self.__output_len:
                target.extend([self.__pad_id] * (self.__output_len - len(target)))
            for _ in range(4):
                mask_src = self.__random_mask(source)[0]
                mask_src = self.__trans_prefix + mask_src
                if len(mask_src) < self.__input_len:
                    mask_src.extend([self.__pad_id] * (self.__input_len - len(mask_src)))
                train_data.append([mask_src, target])
            source = self.__trans_prefix + source
            if len(source) < self.__input_len:
                source.extend([self.__pad_id] * (self.__input_len - len(source)))
            train_data.append([source, target])
        '''
        构造finetune测试数据集
        '''
        valid_data = []
        for sample in samples[int(train_perc * len(samples)):]:
            source = [int(x) for x in sample[1].split()]
            if len(source) < self.__input_len:
                source.extend([self.__pad_id] * (self.__input_len - len(source)))
            target = [self.__sos_id] + [int(x) for x in sample[2].split()] + [self.__eos_id]
            if len(target) < self.__output_len:
                target.extend([self.__pad_id] * (self.__output_len - len(target)))
            valid_data.append([source, target])
        # 打乱训练测试数据集
        random.shuffle(valid_data)
        random.shuffle(train_data)
        return np.array(train_data, dtype=object), np.array(valid_data, dtype=object)

    def for_pretrain(self, data_file: str, shuffle=False):
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            samples = [row for row in reader]
        fp.close()
        data_input, data_output = [], []
        for sample in samples:
            for _ in range(5):  # random mask five times
                input_words = [int(x) for x in sample[1].split()]
                data_input.append(self.__random_mask(input_words))
                if len(sample) == 3:  # if there are output pretrain, random mask five times
                    output_words = [int(x) for x in sample[2].split()]
                    data_output.append(self.__random_mask(output_words))
        if shuffle:
            random.shuffle(data_input)
            random.shuffle(data_output)
        for ele in data_input:  # handle input pretrain
            ele[0] = self.__input_prefix + ele[0]  # add input task prefix
            if len(ele[0]) < self.__input_len:  # add pad_id
                ele[0].extend([self.__pad_id] * (self.__input_len - len(ele[0])))
            if len(ele[1]) < self.__output_len:  # add pad_id
                ele[1].extend([self.__pad_id] * (self.__output_len - len(ele[1])))
            if len(ele[2]) < self.__output_len:  # add pad_id
                ele[2].extend([self.__pad_id] * (self.__output_len - len(ele[2])))
        for ele in data_output:  # handle output pretrain
            ele[1] = self.__output_prefix + ele[1]  # add output task prefix
            if len(ele[0]) < self.__input_len:  # add pad_id
                ele[0].extend([self.__pad_id] * (self.__input_len - len(ele[0])))
            # ele[1] = [self.sos_id] + ele[1] + [self.eos_id]  # add sos_id, eos_id
            if len(ele[1]) < self.__output_len:  # add pad_id
                ele[1].extend([self.__pad_id] * (self.__output_len - len(ele[1])))
            if len(ele[2]) < self.__output_len:  # add pad_id
                ele[2].extend([self.__pad_id] * (self.__output_len - len(ele[2])))
        if len(data_output) == 0:
            return np.array(data_input, dtype=object)
        return np.array(data_input, dtype=object), np.array(data_output, dtype=object)

    def __random_mask(self, words: list):
        # obviously, if there's only one word, don't mask it
        if len(words) == 1:
            return words
        # randomly mask words with probability p
        mask_words = []  # record words after mask
        mask_label = []  # record ground truth
        mask_pos = []  # record mask position
        mask_num = 0
        for word in words:
            r = random.uniform(0, 1)
            if r < self.__mask_prob:
                mask_words.append(self.__mask_id)
                mask_label.append(word)
                mask_pos.append(1)
                mask_num += 1
            else:
                mask_words.append(word)
                mask_label.append(0)
                mask_pos.append(0)
        if mask_num == 0:  # mask one word at least!!!
            index = random.randint(0, len(mask_words) - 1)
            mask_label[index] = mask_words[index]
            mask_words[index] = self.__mask_id
            mask_pos[index] = 1
        return [mask_words, mask_label, mask_pos]


if __name__ == "__main__":
    import os
    from pathlib import Path

    current_work_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(current_work_dir)  # change work dir to current work dir
    # 预训练数据预处理
    Path('data/pretrain').mkdir(exist_ok=True, parents=True)
    pre = Preprocess()
    (data_in, data_out) = pre.for_pretrain("data/train.csv")
    # for idx in range(5):
    #     print("mask input with mask token, mask position: ",
    #           data_in[idx][0], data_in[idx][1], data_in[idx][2], sep='\n')
    #     print("mask output with mask token, mask position: ",
    #           data_out[idx][0], data_out[idx][1], data_in[idx][2], sep='\n')
    print("input pretrain shape: ", data_in.shape, "output pretrain shape: ", data_out.shape)
    np.save('data/pretrain/data_in', data_in)
    np.save('data/pretrain/data_out', data_out)
    data_in = pre.for_pretrain('data/test.csv')
    # for idx in range(5):
    #     print("mask with unmask: ", data_in[idx][0], data_in[idx][1], sep='\n')
    print("test input pretrain shape: ", data_in.shape)
    np.save('data/pretrain/data_in_test', data_in)
    # finetune数据预处理
    Path('data/finetune').mkdir(exist_ok=True, parents=True)
    train, valid = pre.for_finetune("data/train.csv", shuffle=True)
    # for idx in range(5):
    #     print("trans: ", train[idx][0], train[idx][1], sep='\n')
    #     print("trans mask:", valid[idx][0], valid[idx][1], sep='\n')
    print("train finetune shape: ", train.shape, "valid finetune shape: ", valid.shape)
    np.save('data/finetune/train', train)
    np.save('data/finetune/valid', valid)
