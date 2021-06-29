import warnings

import numpy as np
from aiapy.calibrate import correct_degradation
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import AsinhStretch, ImageNormalize, LinearStretch
from reproject import reproject_interp
from sunpy.map import Map, all_coordinates_from_map
from sunpy.map.sources import HMIMap
from sunpy.visualization.colormaps import cm

sdo_norms = [ImageNormalize(vmin=0, vmax=445.5, stretch=AsinhStretch(0.005), clip=True),  # 94
             ImageNormalize(vmin=0, vmax=981.3, stretch=AsinhStretch(0.005), clip=True),  # 131
             ImageNormalize(vmin=0, vmax=6457.5, stretch=AsinhStretch(0.005), clip=True),  # 171
             ImageNormalize(vmin=0, vmax=7757.31, stretch=AsinhStretch(0.005), clip=True),  # 193
             ImageNormalize(vmin=0, vmax=3756, stretch=AsinhStretch(0.005), clip=True),  # 304
             ImageNormalize(vmin=0, vmax=6539.8, stretch=AsinhStretch(0.005), clip=True),  # 211
             ImageNormalize(vmin=0, vmax=915, stretch=AsinhStretch(0.005), clip=True),  # 335
             ImageNormalize(vmin=-100, vmax=100, stretch=LinearStretch(), clip=True),  # mag
             ]

sdo_norms_dict = {k: v for k, v in zip([94, 131, 171, 193, 304, 211, 335, 6173], sdo_norms)}

sdo_cmaps = [cm.sdoaia94, cm.sdoaia131, cm.sdoaia171, cm.sdoaia193, cm.sdoaia304, cm.sdoaia211, cm.sdoaia335, "gray"]


def getMapData(file, resolution, correction_table=None, remove_off_disk=False):
    """Returns the prepared data of the FITS file.

    :param file: the FITS map file
    :param resolution: pixels along x- and y-axis
    :param correction_table: the AIA correction table (aiapy.calibrate.util.get_correction_table)
    :param remove_off_disk: True to set all off-limb pixels to min
    :return: 2D numpy array
    """
    s_map = Map(file)
    s_map = prepMap(s_map, resolution)
    #
    if not isinstance(s_map, HMIMap):
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


def buildFITS(prediction: np.array, reference: str):
    """Returns a '~sunpy.map.Map' from the mask based on the WCS information of the reference file.

    :param prediction: 2D numpy array
    :param reference: reference map file
    :return: sunpy map of the prediction
    """
    ref_map = Map(reference)
    prep_map = prepMap(ref_map, prediction.shape[0])
    prep_map = Map(prediction, prep_map.meta)
    output, footprint = reproject_interp(prep_map, ref_map.wcs, ref_map.data.shape)
    pred_map = Map(output.astype('float32'), ref_map.wcs)
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
