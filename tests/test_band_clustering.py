from pathlib import Path
from unittest import TestCase

import remotior_sensus


class TestBandClassification(TestCase):

    def test_band_classification(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        cfg.logger.log.debug('>>> test semiautomatic classification')
        # create BandSet
        catalog = rs.bandset_catalog()
        file_list = ['L8_2020-01-01/L8_B2.tif', 'L8_2020-01-01/L8_B3.tif',
                     'L8_2020-01-01/L8_B4.tif', 'L8_2020-01-01/L8_B5.tif',
                     'L8_2020-01-01/L8_B6.tif', 'L8_2020-01-01/L8_B7.tif']
        data_path = Path(__file__).parent / 'data'
        catalog.create_bandset(file_list, root_directory=str(data_path))
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
        )
        rs.band_clustering(
            input_bands=catalog.get(1), output_raster_path=temp,
            algorithm_name=cfg.minimum_distance, class_number=3,
            max_iter=2, seed_signatures=cfg.random_pixel
        )
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
        )
        clustering = rs.band_clustering(
            input_bands=catalog.get(1), output_raster_path=temp,
            algorithm_name=cfg.minimum_distance, class_number=3,
            max_iter=2, seed_signatures=cfg.band_mean
        )
        self.assertTrue(rs.files_directories.is_file(clustering.path))
        self.assertTrue(
            rs.files_directories.is_file(clustering.extra['signature_path'])
        )

        # clear temporary directory
        rs.close()
