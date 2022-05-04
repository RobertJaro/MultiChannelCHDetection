import argparse
import logging
import os

from chronnos.data.convert import get_intersecting_files, convertMaps, convertMasks, sdo_norms_dict
from chronnos.train.model import Trainer

parser = argparse.ArgumentParser(description='Predict CHRONNOS masks from SDO fits files')
parser.add_argument('--base_path', type=str,
                    help='the path to store the training results.')
parser.add_argument('--data_path', type=str, required=False, default=None,
                    help='the path to the data directory. Files need to be separated by channel (94, 131, 171, 193, 211, 304, 335, 6173). Only required if convert==True.')
parser.add_argument('--converted_path', type=str,
                    help='the path to store the converted data. This path will be used for loading the training samples.')
parser.add_argument('--convert', type=str, required=False, default='true',
                    help='perform conversion of FTIS files to training data.')
parser.add_argument('-channels', '--channels', nargs='+', help='set to use a subset of wavelenghts.', required=False, default=None)

args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ])

image_channels = 8 if args.channels is None else len(args.channels)
start_resolution = 8
n_dims = [1024, 512, 256, 128, 64, 32, 16]
n_convs = [3, 3, 3, 2, 2, 2, 1]

resolutions = [start_resolution * 2 ** i for i in range(len(n_dims))]

if args.convert.lower() == 'true':
    assert args.data_path is not None, '--data_path required for option --convert True'
    logging.info('====================== CONVERTING DATA ======================')
    grouped_files = get_intersecting_files(args.data_path,
                                           dirs=['94', '131', '171', '193', '211', '304', '335', '6173', 'prep_masks'],
                                           extensions=['.fits'] * 8 + ['.fits.gz'])
    convertMaps(grouped_files[:-1], os.path.join(args.converted_path, 'map'), resolutions)
    convertMasks(grouped_files[-1], os.path.join(args.converted_path, 'mask'), resolutions)

channels = None if args.channels is None else [list(sdo_norms_dict.keys()).index(int(wl)) for wl in args.channels]
logging.info('====================== INIT TRAINER ======================')
trainer = Trainer(args.base_path, args.converted_path, image_channels=image_channels, start_resolution=start_resolution,
                  n_dims=n_dims, n_convs=n_convs, channels=channels)
trainer.train()
