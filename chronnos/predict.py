import argparse
import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sunpy.map import all_coordinates_from_map, Map
from torch.utils.data import DataLoader
from tqdm import tqdm

from chronnos.data.converter import buildFITS
from chronnos.data.generator import MapDataset

parser = argparse.ArgumentParser(description='Predict CHRONNOS masks from SDO fits files')
parser.add_argument('data_path', type=str,
                    help='the path to the data directory. Files need to be separated by channel (94, 131, 171, 193, 304, 211, 335, 6173)')
parser.add_argument('model_path', type=str, help='path to the pretrained CHRONNOS model')
parser.add_argument('eval_path', type=str, help='path to the output files')
parser.add_argument('-plot_samples', type=bool, help='visualize results', default=True)

args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
evaluation_path = args.eval_path
plot_samples = args.plot_samples
os.makedirs(evaluation_path, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)
model.eval()

dirs = ['94', '131', '171', '193', '304', '211', '335', '6173']
map_paths = [sorted(glob.glob(os.path.join(data_path, dir, '*.fits'))) for dir in dirs]

map_ds = MapDataset(list(zip(*map_paths)))
map_loader = DataLoader(map_ds, batch_size=2, shuffle=False)

predictions = []
map_data = []
with torch.no_grad():
    for batch in tqdm(map_loader, desc='model prediction'):
        batch = torch.flip(batch, (2,))  # flip for chronnos
        batch = batch.to(device)
        pred = model(batch).detach().cpu()
        predictions.append(pred)
        map_data.append(batch.detach().cpu())

predictions = torch.cat(predictions, 0).numpy()
map_data = torch.cat(map_data, 0).numpy()

for prediction, ref in tqdm(zip(predictions, map_paths[3]), desc='writing files', total=len(predictions)):
    s_map = buildFITS(prediction[0], ref)
    hpc_coords = all_coordinates_from_map(s_map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
    s_map.data[r > 1] = 0
    #
    map_path = os.path.join(evaluation_path,
                            'prob_map_%s.fits' % s_map.date.datetime.isoformat('T', timespec='seconds'))
    if os.path.exists(map_path):
        os.remove(map_path)
    s_map.save(map_path)
    #
    if plot_samples:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        s_map = Map(map_path)
        s_map.plot(axes=axs[0])
        s_map.draw_limb(axes=axs[0])
        ref_map = Map(ref)
        ref_map.plot(axes=axs[1])
        ref_map.draw_limb(axes=axs[1])
        axs[2].imshow(ref_map.data, **ref_map.plot_settings)
        axs[2].contour(s_map.data, levels=[0.5], colors='red')
        fig.tight_layout()
        fig.savefig(os.path.join(evaluation_path, '%s.jpg' % s_map.date.datetime.isoformat('T', timespec='seconds')),
                    dpi=100)
        plt.close()
