import glob
import os
import random

import numpy as np
from aiapy.calibrate.util import get_correction_table
from dateutil.parser import parse
from torch.utils.data import Dataset

from chronnos.data.convert import getMapData


class FITSDataset(Dataset):
    def __init__(self, files, calibrate=True):
        """
        Load fits files for model training and evaluation
        :param files: list of fits files. channel order for CHRONNOS: [94, 131, 171, 193, 211, 304, 335, LOS magnetogram]
        :param calibrate: adjust device degradation and exposure time. For pre-processed files set calibration=False.
        """
        self.files = files
        self.correction_table = get_correction_table()
        self.calibrate = calibrate
        super().__init__()

    def __getitem__(self, index):
        file_cube = self.files[index]
        x = np.array([getMapData(file, 512, correction_table=self.correction_table, calibrate=self.calibrate) for file in file_cube])
        x = x * 2 - 1  # scale to [-1, 1]
        x = np.transpose(x, axes=[2, 0, 1])
        return np.array(x.data.tolist(), dtype=np.float32)

    def __len__(self):
        return len(self.files)


class MapDataset(Dataset):
    def __init__(self, files, channel=None):
        """
        Load saved maps for model training.
        :param files: list of npy files
        :param channel (optional): select subset of channels (idx)
        """
        self.files = files
        self.channel = None if channel is None else channel if isinstance(channel, list) else [channel]
        super().__init__()

    def __getitem__(self, index):
        file = self.files[index]
        x = np.load(file)
        x = x * 2 - 1  # scale to [-1, 1]
        x = np.transpose(x, axes=[2, 0, 1])
        if self.channel is not None:
            x = x[self.channel]
        return np.array(x.data.tolist(), dtype=np.float32)

    def __len__(self):
        return len(self.files)


class MaskDataset(Dataset):
    def __init__(self, files):
        """
        Load saved masks for model training.
        :param files: list of npy files
        """
        self.files = files
        super().__init__()

    def __getitem__(self, index):
        file = self.files[index]
        y = np.load(file)
        y = (y >= 0.1).astype(np.float32)  # make hard labels
        y = np.transpose(y, axes=[2, 0, 1])
        return np.array(y.data.tolist(), dtype=np.float32)

    def __len__(self):
        return len(self.files)


class CombinedCHDataset(Dataset):
    def __init__(self, map_files, mask_files, flip_prob=0.5, channel=None):
        """
        Load paired maps and masks for model training.
        :param map_files: list of npy files
        :param mask_files: list of npy files
        :param flip_prob: probability of applying a random flip
        :param channel (optional): select subset of channels (idx)
        """
        assert len(map_files) == len(mask_files), 'Number of files does not match!'
        self.flip_prob = flip_prob
        self.map_ds = MapDataset(map_files, channel=channel)
        self.mask_ds = MaskDataset(mask_files)
        super().__init__()

    def __getitem__(self, index):
        x, y = self.map_ds[index], self.mask_ds[index]
        if random.random() < self.flip_prob:
            x = np.flip(x, axis=1)
            y = np.flip(y, axis=1)
        if random.random() < self.flip_prob:
            x = np.flip(x, axis=2)
            y = np.flip(y, axis=2)
        return np.array(x.data.tolist(), dtype=np.float32), np.array(y.data.tolist(), dtype=np.float32)

    def __len__(self):
        return len(self.map_ds)


def getDataSet(ds_path, resolution, train_months=None):
    """
    Group files for model training
    :param ds_path: base path to the converted files
    :param resolution: target resolution
    :param train_months: filter by month
    :return:
    """
    train_months = list(range(1, 11)) if train_months is None else train_months
    mask_files = sorted(glob.glob(os.path.join(os.path.join(ds_path, 'mask', '%d' % resolution), '*.npy')))
    map_files = sorted(glob.glob(os.path.join(os.path.join(ds_path, 'map', '%d' % resolution), '*.npy')))

    basename_mask = [os.path.basename(f) for f in mask_files]
    basename_map = [os.path.basename(f) for f in map_files]
    assert np.all(np.array(basename_mask) == np.array(basename_map)), 'Filenames of masks and maps need to match!'

    dates = [parse(f.split('.')[0]) for f in basename_map]
    train_condition = np.array([d.month in train_months for d in dates])
    valid_condition = np.array([d.month not in train_months for d in dates])

    map_train = np.array(list(map_files))[train_condition]
    mask_train = np.array(list(mask_files))[train_condition]

    map_valid = np.array(list(map_files))[valid_condition]
    mask_valid = np.array(list(mask_files))[valid_condition]

    return map_train, mask_train, map_valid, mask_valid
