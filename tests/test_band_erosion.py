from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestBandErosion(TestCase):

    def test_band_erosion(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        file_list = ['./data/S2_2020-01-01/S2_B02.tif',
                     './data/S2_2020-01-01/S2_B03.tif']
        cfg.logger.log.debug('>>> test band_erosion')
        erosion = rs.band_erosion(
            input_bands=file_list, output_path=cfg.temp.dir,
            value_list=[1, 425], size=1, circular_structure=True,
            prefix='erosion_'
            )
        self.assertTrue(files_directories.is_file(erosion.paths[0]))

        # clear temporary directory
        rs.close()
