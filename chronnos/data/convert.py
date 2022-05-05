import glob
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import astropy.io.ascii
import numpy as np
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.util import get_correction_table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import block_reduce
from astropy.visualization import AsinhStretch, ImageNormalize, LinearStretch
from dateutil.parser import parse
from reproject import reproject_interp
from sunpy.map import Map, all_coordinates_from_map
from sunpy.map.sources import HMIMap
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

sdo_norms = [ImageNormalize(vmin=0, vmax=445.5, stretch=AsinhStretch(0.005), clip=True),  # 94
             ImageNormalize(vmin=0, vmax=981.3, stretch=AsinhStretch(0.005), clip=True),  # 131
             ImageNormalize(vmin=0, vmax=6457.5, stretch=AsinhStretch(0.005), clip=True),  # 171
             ImageNormalize(vmin=0, vmax=7757.31, stretch=AsinhStretch(0.005), clip=True),  # 193
             ImageNormalize(vmin=0, vmax=6539.8, stretch=AsinhStretch(0.005), clip=True),  # 211
             ImageNormalize(vmin=0, vmax=3756, stretch=AsinhStretch(0.005), clip=True),  # 304
             ImageNormalize(vmin=0, vmax=915, stretch=AsinhStretch(0.005), clip=True),  # 335
             ImageNormalize(vmin=-100, vmax=100, stretch=LinearStretch(), clip=True),  # mag
             ]

sdo_norms_dict = {k: v for k, v in zip([94, 131, 171, 193, 211, 304, 335, 6173], sdo_norms)}

sdo_cmaps = [cm.sdoaia94, cm.sdoaia131, cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia304, cm.sdoaia335, "gray"]


def getMapData(file, resolution, correction_table=None, remove_off_disk=False, calibrate=True):
    """Returns the prepared data of the FITS file.

    :param file: the FITS map file
    :param resolution: pixels along x- and y-axis
    :param correction_table: the AIA correction table (aiapy.calibrate.util.get_correction_table)
    :param remove_off_disk: True to set all off-limb pixels to min. HMI images are automatically truncated.
    :return: 2D numpy array
    """
    s_map = Map(file)
    s_map = prepMap(s_map, resolution)
    #
    if not isinstance(s_map, HMIMap) and calibrate:
        s_map = correct_degradation(s_map, correction_table=correction_table)
        data = np.nan_to_num(s_map.data)
        data = data / s_map.meta["exptime"]
    else:  # truncate boundary for HMI images
        data = np.nan_to_num(s_map.data)
        hpc_coords = all_coordinates_from_map(s_map)
        r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
        data[r > 1] = 0
    if remove_off_disk:
        hpc_coords = all_coordinates_from_map(s_map)
        r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
        data[r > 1] = np.min(s_map.data)
    data = sdo_norms_dict[int(s_map.wavelength.value)](data).data
    data = data.astype(np.float32)
    return data


def buildFITS(prediction: np.array, reference: str, reproject=True):
    """Returns a '~sunpy.map.Map' from the mask based on the WCS information of the reference file.

    :param prediction: 2D numpy array
    :param reference: reference map file
    :return: sunpy map of the prediction
    """
    ref_map = Map(reference)
    prep_map = prepMap(ref_map, prediction.shape[0])
    prep_map = Map(prediction, prep_map.meta)
    if reproject:
        output, footprint = reproject_interp(prep_map, ref_map.wcs, ref_map.data.shape)
        pred_map = Map((output >= 0.5).astype(np.bool), ref_map.wcs)
    else:
        pred_map = Map((prediction >= 0.5).astype(np.bool), prep_map.wcs)
    pred_map.meta['rsun_obs'] = ref_map.meta['rsun_obs']
    pred_map.meta['instrume'] = 'CHRONNOS'
    pred_map.meta['detector'] = 'CHRONNOS'
    pred_map.meta['CDELT1'] *= 60 * 60
    pred_map.meta['CDELT2'] *= 60 * 60
    pred_map.meta['CUNIT1'] = 'arcsec'
    pred_map.meta['CUNIT2'] = 'arcsec'
    return pred_map


def prepMap(s_map: Map, resolution: int, padding_factor: float = 0.1):
    """Returns the adjusted map. The solar disk is centered to the image,
    the north pole axis aligned with the y-axis and the scale adjusted to (1 + padding) * R_Sun[arcsec] / resolution

    :param s_map: '~sunpy.map.Map' object
    :param resolution: pixels along x- and y-axis
    :param padding_factor: distance between solar limb and image border given in R_Sun
    :return: adjusted sunpy map
    """
    warnings.simplefilter("ignore")  # ignore warnings
    r_obs_pix = s_map.rsun_obs / s_map.scale[0]  # normalize solar radius
    r_obs_pix = (1 + padding_factor) * r_obs_pix
    scale_factor = resolution / (2 * r_obs_pix.value)
    s_map = Map(np.nan_to_num(s_map.data).astype(np.float32), s_map.meta)
    s_map = s_map.rotate(recenter=True, scale=scale_factor, missing=0, order=3)
    arcs_frame = (resolution / 2) * s_map.scale[0].value
    s_map = s_map.submap(SkyCoord(-arcs_frame * u.arcsec, -arcs_frame * u.arcsec, frame=s_map.coordinate_frame),
                         top_right=SkyCoord(arcs_frame * u.arcsec, arcs_frame * u.arcsec, frame=s_map.coordinate_frame))
    # remove overlap after submap
    pad_x = s_map.data.shape[0] - resolution
    pad_y = s_map.data.shape[1] - resolution
    s_map = s_map.submap(bottom_left=[pad_x // 2, pad_y // 2] * u.pix,
                         top_right=[pad_x // 2 + resolution - 1, pad_y // 2 + resolution - 1] * u.pix)
    #
    s_map.meta['r_sun'] = s_map.rsun_obs.value / s_map.meta['cdelt1']
    return s_map


def convertMaps(grouped_files, converted_path, resolutions, n_workers=8, replace=False):
    """Convert FITS files for model training.

    :param grouped_files: list of FITS files in the format (channel, file)
    :param converted_path: path where the converted data is stored
    :param resolutions: set of resolutions used for training
    :param n_workers: number of parallel worker threads
    :param replace: replace existing files
    :return: None
    """
    [os.makedirs(os.path.join(converted_path, '%d' % res), exist_ok=True) for res in resolutions]
    correction_table = get_local_correction_table()

    converter = _MapConverter(converted_path, resolutions, correction_table, replace)
    # async conversion
    with Pool(n_workers) as p:
        [None for _ in tqdm(p.imap_unordered(converter.convert, np.array(grouped_files).transpose()), total=len(grouped_files[0]))]
    # _convert(np.array(grouped_files).transpose()[0])

class _MapConverter:

    def __init__(self, converted_path, resolutions, correction_table, replace):
        self.converted_path = converted_path
        self.resolutions = resolutions
        self.replace = replace
        self.max_res = max(resolutions)
        self.correction_table = correction_table

    def convert(self, c_files):
        # check if file already exists
        if all([os.path.exists(
                os.path.join(self.converted_path, '%d' % res, os.path.basename(c_files[0]).replace('.fits', '.npy')))
            for
            res in self.resolutions]) and self.replace == False:
            return
        maps_data = [getMapData(c_file, self.max_res, self.correction_table) for c_file in c_files]
        maps_data = np.stack(maps_data, -1)
        for resolution in self.resolutions:
            path = os.path.join(self.converted_path, '%d' % resolution,
                                os.path.basename(c_files[0]).replace('.fits', '.npy'))
            block = (maps_data.shape[0] // resolution, maps_data.shape[1] // resolution, 1)
            map_data_reduced = block_reduce(maps_data, block, np.mean)
            np.save(path, map_data_reduced.astype(np.float32))

def convertMasks(files, converted_path, resolutions, n_workers=8, replace=False):
    """Convert label masks for model training.

    :param files: list of FITS files
    :param converted_path: path where the converted data is stored
    :param resolutions: set of resolutions used for training
    :param n_workers: number of parallel worker threads
    :param replace: replace existing files
    :return: None
    """
    [os.makedirs(os.path.join(converted_path, '%d' % res), exist_ok=True) for res in resolutions]
    converter = _MaskConverter(converted_path, resolutions, replace)
    # async conversion
    with Pool(n_workers) as p:
        [None for _ in tqdm(p.imap_unordered(converter.convert, files), total=len(files))]


class _MaskConverter:
    def __init__(self, converted_path, resolutions, replace):
        self.converted_path = converted_path
        self.resolutions = resolutions
        self.replace = replace
        self.max_res = max(resolutions)

    # conversion function
    def convert(self, file):
        # check if file already exists
        if all([os.path.exists(os.path.join(self.converted_path, '%d' % res,
                                            os.path.basename(file).replace('.fits.gz', '.npy')))
                for res in self.resolutions]) and self.replace == False:
            return
        s_map = Map(file)
        s_map = prepMap(s_map, self.max_res)
        data = s_map.data > 0.3  # back to binary
        data = data[..., None] # expand last dimension
        for resolution in self.resolutions:
            path = os.path.join(self.converted_path, '%d' % resolution,
                                os.path.basename(file).replace('.fits.gz', '.npy'))
            block = (data.shape[0] // resolution, data.shape[1] // resolution, 1)
            map_data_reduced = block_reduce(data, block, np.mean)
            np.save(path, map_data_reduced.astype(np.float32))

def get_intersecting_files(path, dirs, months=None, years=None, extensions=None):
    """Find intersecting files in directory.

    :param path: base directory
    :param dirs: directories to scan
    :param months: filter for months
    :param years: filter for years
    :param extensions: file extension for each directory
    :return: list of grouped files
    """
    extensions = extensions if extensions is not None else ['.fits'] * len(dirs)
    basenames = [
        [os.path.basename(path).replace(ext, '') for path in glob.glob(os.path.join(path, str(d), '**', '*%s' % ext), recursive=True)]
        for d, ext in zip(dirs, extensions)]
    basenames = list(set(basenames[0]).intersection(*basenames))
    if months:  # assuming filename is parsable datetime
        basenames = [bn for bn in basenames if parse(bn.split('.')[0]).month in months]
    if years:  # assuming filename is parsable datetime
        basenames = [bn for bn in basenames if parse(bn.split('.')[0]).year in years]
    basenames = sorted(list(basenames))
    return [[os.path.join(path, str(dir), b + ext) for b in basenames] for dir, ext in zip(dirs, extensions)]

def get_local_correction_table():
    """Load AIA correction table from home directory.

    Downloads a new table if no file exists.

    :return: the correction table
    """
    path = os.path.join(Path.home(), 'aiapy', 'correction_table.dat')
    if os.path.exists(path):
        return get_correction_table(path)
    os.makedirs(os.path.join(Path.home(), 'aiapy'), exist_ok=True)
    correction_table = get_correction_table()
    astropy.io.ascii.write(correction_table, path)
    return correction_table