from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestNeighborPixels(TestCase):

    def test_neighbor_pixels(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        file_list = ['./data/S2_2020-01-01/S2_B02.tif',
                     './data/S2_2020-01-01/S2_B03.tif']
        cfg.logger.log.debug('>>> test neighbor_pixels')
        neighbor = rs.band_neighbor_pixels(
            input_bands=file_list, output_path=cfg.temp.dir, size=1,
            circular_structure=True, stat_name='Mean', prefix='neighbor_'
            )
        self.assertTrue(files_directories.is_file(neighbor.paths[0]))
        cfg.logger.log.debug('>>> test neighbor_pixels with coordinates')
        coordinate_list = [230250, 4674550, 230320, 4674440]
        neighbor = rs.band_neighbor_pixels(
            input_bands=file_list, output_path=cfg.temp.dir, size=1,
            circular_structure=True, stat_name='Mean', prefix='neighbor_',
            extent_list=coordinate_list
            )
        self.assertTrue(files_directories.is_file(neighbor.paths[0]))

        # clear temporary directory
        rs.close()
