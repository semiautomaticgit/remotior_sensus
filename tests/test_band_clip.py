from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestBandClip(TestCase):

    def test_band_clip(self):
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
        cfg.logger.log.debug('>>> test band clip input BandSet')
        # box coordinate list
        extent_list = [230250, 4674510, 230320, 4674440]
        output = rs.band_clip(input_bands=catalog.get_bandset(1),
                              output_path=cfg.temp.dir, prefix='clip_',
                              extent_list=extent_list)
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))

        v = './data/files/roi.gpkg'
        output = rs.band_clip(input_bands=catalog.get_bandset(1),
                              output_path=cfg.temp.dir, prefix='clip_',
                              vector_path=v)
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))
        v = './data/files/roi.gpkg'
        output = rs.band_clip(input_bands=catalog.get_bandset(1),
                              output_path=cfg.temp.dir, prefix='clip_',
                              vector_path=v, vector_field='class')
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))

        # clear temporary directory
        rs.close()
