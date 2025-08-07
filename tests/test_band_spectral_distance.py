from pathlib import Path
from unittest import TestCase

import remotior_sensus


class TestBandSpectralDistance(TestCase):

    def test_band_spectral_distance(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        catalog = rs.bandset_catalog()
        data_path = Path(__file__).parent / 'data'
        file_list_1 = ['S2_2020-01-03/S2_B02.tif', 'S2_2020-01-03/S2_B03.tif',
                       'S2_2020-01-03/S2_B04.tif']
        file_list_2 = ['S2_2020-01-04/S2_B02.tif', 'S2_2020-01-04/S2_B03.tif',
                       'S2_2020-01-04/S2_B04.tif']
        catalog.create_bandset(
            file_list_1, wavelengths=['Sentinel-2'], bandset_number=1,
            root_directory=str(data_path)
        )
        catalog.create_bandset(
            file_list_2, wavelengths=['Sentinel-2'], bandset_number=2,
            root_directory=str(data_path)
        )
        bandset_list = [catalog.get_bandset(1), catalog.get_bandset(2)]
        cfg.logger.log.debug('>>> test spectral distance')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        distance = rs.band_spectral_distance(
            input_bandsets=bandset_list, output_path=temp
        )
        self.assertTrue(rs.files_directories.is_file(distance.path))
        self.assertTrue(distance.check)
        cfg.logger.log.debug('>>> test spectral distance with catalog')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        distance = rs.band_spectral_distance(
            input_bandsets=[1, 2], output_path=temp, bandset_catalog=catalog
        )
        self.assertTrue(rs.files_directories.is_file(distance.path))
        self.assertTrue(distance.check)
        cfg.logger.log.debug('>>> test spectral distance using spectral angle')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        distance = rs.band_spectral_distance(
            input_bandsets=bandset_list, output_path=temp,
            algorithm_name=cfg.spectral_angle_mapping_a
        )
        self.assertTrue(rs.files_directories.is_file(distance.path))
        self.assertTrue(distance.check)
        cfg.logger.log.debug('>>> test spectral distance with threshold')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        distance = rs.band_spectral_distance(
            input_bandsets=bandset_list, output_path=temp, threshold=1000
        )
        self.assertTrue(rs.files_directories.is_file(distance.path))
        self.assertTrue(distance.check)

        # clear temporary directory
        rs.close()
