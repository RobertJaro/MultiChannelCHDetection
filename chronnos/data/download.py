import argparse
import logging
import multiprocessing
import os
import threading
import traceback
from datetime import timedelta
from multiprocessing.queues import JoinableQueue
from pathlib import Path
from urllib.request import urlopen

import drms
import numpy as np
import pandas as pd
from astropy.io.fits import HDUList
from dateutil.parser import parse
from sunpy.map import Map


class DataSetFetcher:
    """Download tool for JSOC.

    Attributes:
        ds_path: the path to the download directory
        num_worker_threads: number of parallel threads being used
        hmi_series: the JSOOC series used for HMI magnetograms
    """

    def __init__(self, ds_path, num_worker_threads=8, hmi_series='hmi.M_720s'):
        """Init the DataSetFetcher."""
        self.ds_path = ds_path
        self.dirs = ['94', '131', '171', '193', '211', '304', '335', '6173']
        os.makedirs(ds_path, exist_ok=True)
        [os.makedirs(os.path.join(ds_path, dir), exist_ok=True) for dir in self.dirs]
        self.hmi_series = hmi_series

        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler("{0}/{1}.log".format(ds_path, "info_log")),
                logging.StreamHandler()
            ])

        self.drms_client = drms.Client(verbose=False)
        self.download_queue = JoinableQueue(ctx=multiprocessing.get_context())
        for i in range(num_worker_threads):
            t = threading.Thread(target=self.download_worker)
            t.start()

    def download_worker(self):
        """Worker thread for data download"""
        while True:
            header, segment, t = self.download_queue.get()
            logging.info('Start download: %s / %s' % (t.isoformat(' '), header['WAVELNTH']))
            dir = os.path.join(self.ds_path, '%d' % header['WAVELNTH'])
            map_path = os.path.join(dir, '%s.fits' % t.isoformat('T', timespec='seconds').replace(':', ''))
            if os.path.exists(map_path):
                self.download_queue.task_done()
                continue
            try:
                # load map
                url = 'http://jsoc.stanford.edu' + segment
                url_request = urlopen(url)
                fits_data = url_request.read()
                hdul = HDUList.fromstring(fits_data)
                hdul.verify('silentfix')
                data = hdul[1].data
                header = {k: v for k, v in header.items() if not pd.isna(v)}
                header['DATE_OBS'] = header['DATE__OBS']
            except Exception as ex:
                logging.info('Download failed: %s (requeue)' % header['DATE__OBS'])
                logging.info(ex)
                self.download_queue.put((header, segment, t))
                self.download_queue.task_done()
                continue
            s_map = Map(data, header)
            if os.path.exists(map_path):
                os.remove(map_path)
            s_map.save(map_path)
            self.download_queue.task_done()
            logging.info('Finished download: %s / %s' % (t.isoformat(' '), header['WAVELNTH']))

    def fetchDates(self, dates):
        """Download the closest observations to the specified dates.

        :param dates: list of datetime dates to download
        """
        for date in dates:
            if all([os.path.exists(os.path.join(self.ds_path, dir,
                                                date.isoformat('T', timespec='seconds').replace(':', '') + '.fits'))
                    for dir in self.dirs]):
                continue
            try:
                self.fetchData(date)
            except Exception as ex:
                logging.error(traceback.format_exc())
                logging.error('Unable to download: %s' % date.isoformat())
        self.download_queue.join()

    def fetchData(self, time):
        """Adds a single date to the download queue.

        :param time: datetime to download
        """
        # query Magnetogram
        time_param = '%sZ' % time.isoformat('_', timespec='seconds')
        ds_hmi = '%s[%s]{magnetogram}' % (self.hmi_series, time_param)
        keys_hmi = self.drms_client.keys(ds_hmi)
        header_hmi, segment_hmi = self.drms_client.query(ds_hmi, key=','.join(keys_hmi), seg='magnetogram')
        if len(header_hmi) != 1 or np.any(header_hmi.QUALITY != 0):
            self.fetchDataFallback(time)
            return

        # query EUV
        time_param = '%sZ' % time.isoformat('_', timespec='seconds')
        ds_euv = 'aia.lev1_euv_12s[%s]{image}' % time_param
        keys_euv = self.drms_client.keys(ds_euv)
        header_euv, segment_euv = self.drms_client.query(ds_euv, key=','.join(keys_euv), seg='image')
        if len(header_euv) != 7 or np.any(header_euv.QUALITY != 0):
            self.fetchDataFallback(time)
            return

        for (idx, h), s in zip(header_hmi.iterrows(), segment_hmi.magnetogram):
            self.download_queue.put((h.to_dict(), s, time))
        for (idx, h), s in zip(header_euv.iterrows(), segment_euv.image):
            self.download_queue.put((h.to_dict(), s, time))

    def fetchDataFallback(self, time):
        """Alternative download method in case of errors.

        :param time: datetime to download
        """
        id = time.isoformat()

        logging.info('Fallback download: %s' % id)
        # query Magnetogram
        t = time - timedelta(hours=6)
        ds_hmi = 'hmi.M_720s[%sZ/12h@720s]{magnetogram}' % t.replace(tzinfo=None).isoformat('_', timespec='seconds')
        keys_hmi = self.drms_client.keys(ds_hmi)
        header_tmp, segment_tmp = self.drms_client.query(ds_hmi, key=','.join(keys_hmi), seg='magnetogram')
        assert len(header_tmp) != 0, 'No data found!'
        date_str = header_tmp['DATE__OBS'].replace('MISSING', '').str.replace('60', '59')  # fix date format
        date_diff = np.abs(pd.to_datetime(date_str).dt.tz_localize(None) - time)
        # sort and filter
        header_tmp['date_diff'] = date_diff
        header_tmp.sort_values('date_diff')
        segment_tmp['date_diff'] = date_diff
        segment_tmp.sort_values('date_diff')
        cond_tmp = header_tmp.QUALITY == 0
        header_tmp = header_tmp[cond_tmp]
        segment_tmp = segment_tmp[cond_tmp]
        assert len(header_tmp) > 0, 'No valid quality flag found'
        # replace invalid
        header_hmi = header_tmp.iloc[0].drop('date_diff')
        segment_hmi = segment_tmp.iloc[0].drop('date_diff')
        ############################################################
        # query EUV
        header_euv, segment_euv = [], []
        t = time - timedelta(hours=6)
        for wl in [94, 131, 171, 193, 211, 304, 335]:
            euv_ds = 'aia.lev1_euv_12s[%sZ/12h@12s][%d]{image}' % (
                t.replace(tzinfo=None).isoformat('_', timespec='seconds'), wl)
            keys_euv = self.drms_client.keys(euv_ds)
            header_tmp, segment_tmp = self.drms_client.query(euv_ds, key=','.join(keys_euv), seg='image')
            assert len(header_tmp) != 0, 'No data found!'
            date_str = header_tmp['DATE__OBS'].replace('MISSING', '').str.replace('60', '59')  # fix date format
            date_diff = (pd.to_datetime(date_str).dt.tz_localize(None) - time).abs()
            # sort and filter
            header_tmp['date_diff'] = date_diff
            header_tmp.sort_values('date_diff')
            segment_tmp['date_diff'] = date_diff
            segment_tmp.sort_values('date_diff')
            cond_tmp = header_tmp.QUALITY == 0
            header_tmp = header_tmp[cond_tmp]
            segment_tmp = segment_tmp[cond_tmp]
            assert len(header_tmp) > 0, 'No valid quality flag found'
            # replace invalid
            header_euv.append(header_tmp.iloc[0].drop('date_diff'))
            segment_euv.append(segment_tmp.iloc[0].drop('date_diff'))

        self.download_queue.put((header_hmi.to_dict(), segment_hmi.magnetogram, time))
        for h, s in zip(header_euv, segment_euv):
            self.download_queue.put((h.to_dict(), s.image, time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download AIA and HMI files for CHRONNNOS image segmenation.')
    parser.add_argument('--path', type=str, help='the path to the storage directory',
                        default=os.path.join(Path.home(), 'chronnos'))
    parser.add_argument('--dates', nargs='*', type=lambda s: parse(s), help='dates in dateutil parseable format')
    parser.add_argument('--hmi_series', type=str, help='jsoc hmi series (hmi.M_720s or hmi.M_45s)',
                        default='hmi.M_720s')
    parser.add_argument('--n_workers', type=int, help='number of parallel threads used for download',
                        default=8)

    args = parser.parse_args()

    fetcher = DataSetFetcher(ds_path=args.path, hmi_series=args.hmi_series, num_worker_threads=args.n_workers)
    fetcher.fetchDates(args.dates)
