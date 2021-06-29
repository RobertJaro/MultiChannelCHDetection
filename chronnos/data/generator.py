import numpy as np
from aiapy.calibrate.util import get_correction_table
from torch.utils.data import Dataset

from chronnos.data.converter import getMapData


class MapDataset(Dataset):
    def __init__(self, files):
        """
        Load fits files for model training and evaluation
        :param files: list of fits files. channel order: [94, 131, 171, 193, 304, 211, 335, LOS magnetogram]
        """
        self.files = files
        self.correction_table = get_correction_table()
        super().__init__()

    def __getitem__(self, index):
        file_cube = self.files[index]
        x = np.array([getMapData(file, 512, correction_table=self.correction_table) for file in file_cube])
        x = x * 2 - 1  # scale to [-1, 1]
        x = np.flip(x, 1)
        return np.array(x.data.tolist(), dtype=np.float32)

    def __len__(self):
        return len(self.files)
