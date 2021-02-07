"""
Signal Dataset Classes

"""


from data.config import HOME
import os.path as osp
import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from matplotlib import pyplot as plt



class Signal_Data(data.Dataset):
    """Signal Data Dataset Object
    Arguments:
        root (string): 文件路径
        transform (callable, optional): 图片是否进行转换
        target_transform (callable, optional): annotation是否进行转换
            (eg: take in caption string, return tensor of word indices)

    """
    def __init__(self, root,
                transform = None,target_transform = None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_root = osp.join(root, 'train_data')
        self.label_root = osp.join(root, 'train_label')
        self.train_filename = os.listdir(self.train_root)
        self.label_filename = os.listdir(self.label_root)


    def __getitem__(self, index):
        data, target = self.pull_item(index)

        return data, target

    def __len__(self):
        return len(self.train_filename)
    def pull_item(self, index):
        """
        输入：下标
        输出：index下标所对应的数据及标签
        """

        read_data = np.load(osp.join(self.train_root, self.train_filename[index - 1]))
        target = np.load(osp.join(self.label_root, self.label_filename[index - 1]), allow_pickle=True)
        
        fft_10 = read_data[read_data.files[0]]
        fft_80 = read_data[read_data.files[1]]
        RC_20 = read_data[read_data.files[2]]
        RC_40 = read_data[read_data.files[3]]

        data = [fft_10, fft_80, RC_20, RC_40]
        
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            target = np.array(target)
            data, lines, labels = self.transform(data, target[:, :2], target[:, 2])
        return [torch.from_numpy(d) for d in data], target


if __name__ == '__main__':
    dataset = Signal_Data(root = 'D:\signal_data\G35_recorder_npydata')
    print(len(dataset))
    data, label = next(iter(dataset))
    print('hihihi')


