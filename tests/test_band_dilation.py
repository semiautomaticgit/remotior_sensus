from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestBandDilation(TestCase):

    def test_band_dilation(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        file_list = ['./data/S2_2020-01-01/S2_B02.tif',
                     './data/S2_2020-01-01/S2_B03.tif']
        cfg.logger.log.debug('>>> test band_dilation')
        dilation = rs.band_dilation(
            input_bands=file_list, output_path=cfg.temp.dir,
            value_list=[1, 425], size=3, circular_structure=True,
            prefix='dilation_'
            )
        self.assertTrue(files_directories.is_file(dilation.paths[0]))
        cfg.logger.log.debug('>>> test band_dilation without output')
        dilation = rs.band_dilation(
            input_bands=file_list, value_list=[1, 425], size=3,
            circular_structure=True, prefix='dilation_'
            )
        self.assertTrue(files_directories.is_file(dilation.paths[0]))
        cfg.logger.log.debug('>>> test band_dilation with coordinate list')
        coordinate_list = [230250, 4674550, 230320, 4674440]
        dilation = rs.band_dilation(
            input_bands=file_list, value_list=[1, 425], size=3,
            circular_structure=True, prefix='dilation_',
            extent_list=coordinate_list
            )
        self.assertTrue(files_directories.is_file(dilation.paths[0]))

        # clear temporary directory
        rs.close()
