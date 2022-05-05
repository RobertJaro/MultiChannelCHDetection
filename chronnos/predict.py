import argparse
import gzip
import os

from matplotlib import pyplot as plt
from sunpy.map import Map
from tqdm import tqdm

from chronnos.data.convert import get_intersecting_files, sdo_norms
from chronnos.evaluate.detect import CHRONNOSDetector

parser = argparse.ArgumentParser(description='Predict CHRONNOS masks from SDO FITS files')
parser.add_argument('--data_path', type=str,
                    help='the path to the data directory. Files need to be separated by channel.')
parser.add_argument('--model_path', type=str, help='path to the pretrained CHRONNOS model')
parser.add_argument('--evaluation_path', type=str, help='path to the output files')
parser.add_argument('--no_reproject', action='store_false', help='reproject coronal hole maps to the original AIA map.')
parser.add_argument('--plot_samples', action='store_true', help='visualize results')
parser.add_argument('--channels', '-channels', nargs='+', required=False,
                    default=['94', '131', '171', '193', '211', '304', '335', '6173'],
                    help='subset of channels to load. The order must match the input channels of the model.')
parser.add_argument('--n_workers', type=int, help='number of parallel threads for converting data', default=8)

args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
evaluation_path = args.evaluation_path
plot_samples = args.plot_samples
reproject = not args.no_reproject
os.makedirs(evaluation_path, exist_ok=True)

map_paths = get_intersecting_files(data_path, dirs=args.channels)

chronnos_detector = CHRONNOSDetector(model_path=model_path)

ch_generator = chronnos_detector.ipredict(map_paths, reproject=reproject, num_workers=args.n_workers)
for ch_map, aia_map_path in tqdm(zip(ch_generator, map_paths[3]), total=len(map_paths[3])):
    # save fits
    map_path = os.path.join(evaluation_path, os.path.basename(aia_map_path))
    if os.path.exists(map_path):
        os.remove(map_path)
    Map(ch_map.data.astype('int16'), ch_map.meta).save(map_path)
    with open(map_path, 'rb') as f_in, gzip.open(map_path + '.gz', 'wb') as f_out:
        f_out.writelines(f_in)
    os.remove(map_path)
    # plot overlay
    if plot_samples:
        plt.figure(figsize=(4, 4))
        aia_map = Map(aia_map_path)
        aia_map.plot(norm=sdo_norms[3])
        ch_map.draw_contours(levels=[0.5], colors='red', origin='lower')
        plt.savefig(os.path.join(evaluation_path, os.path.basename(aia_map_path).replace('.fits', '.jpg')), dpi=100)
        plt.close()
