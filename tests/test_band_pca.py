from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestBandPCA(TestCase):

    def test_band_pca(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        catalog = rs.bandset_catalog()
        file_list = ['S2_2020-01-01/S2_B02.tif', 'S2_2020-01-01/S2_B03.tif',
                     'S2_2020-01-01/S2_B04.tif']
        root_directory = './data'
        catalog.create_bandset(
            file_list, wavelengths=['Sentinel-2'], bandset_number=1,
            root_directory=root_directory
            )
        cfg.logger.log.debug('>>> test band PCA input BandSet')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        output = rs.band_pca(input_bands=catalog.get_bandset(1),
                             output_path=temp)
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))
        cfg.logger.log.debug('>>> test band PCA input BandSet with coordinate')
        coordinate_list = [230250, 4674550, 230320, 4674440]
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        output = rs.band_pca(input_bands=catalog.get_bandset(1),
                             output_path=temp,
                             extent_list=coordinate_list)
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))
        cfg.logger.log.debug('>>> test band PCA input BandSet with coordinate')
        bs = catalog.get_bandset(1)
        bs.box_coordinate_list = [230250, 4674550, 230320, 4674440]
        output = rs.band_pca(input_bands=catalog.get_bandset(1),
                             extent_list=coordinate_list)
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))

        # clear temporary directory
        rs.close()
