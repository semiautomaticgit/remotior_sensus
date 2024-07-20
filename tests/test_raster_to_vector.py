from pathlib import Path
from unittest import TestCase

import remotior_sensus


class TestRasterToVector(TestCase):

    def test_raster_to_vector(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        data_path = Path(__file__).parent / 'data'
        p = str(data_path / 'S2_2020-01-01' / 'S2_B02.tif')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.gpkg_suffix)
        cfg.logger.log.debug('>>> test raster_to_vector')
        vector = rs.raster_to_vector(p, temp)
        self.assertTrue(rs.files_directories.is_file(vector.path))
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.gpkg_suffix)
        cfg.logger.log.debug('>>> test raster_to_vector dissolve')
        vector = rs.raster_to_vector(p, temp, dissolve=True)
        self.assertTrue(rs.files_directories.is_file(vector.path))

        # clear temporary directory
        rs.close()
