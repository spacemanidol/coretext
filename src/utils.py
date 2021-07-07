import os
import random
import numpy as np
import h5py
import json
import torch
from torch.utils.data import Dataset
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
class CustomDataset(Dataset):
    def __init__(self, data_folder, data_name):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        """
        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']
        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, captions_prefix + 'captions_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)
        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, captions_prefix + 'captions_length_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
        # Total number of datapoints
        self.dataset_size = self.h['images'].shape[0]
        self.transform = None #transform

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        data = (img, caption, caplen)
        return data
    
    def __len__(self):
        return self.dataset_size
