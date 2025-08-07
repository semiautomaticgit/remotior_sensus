from pathlib import Path
from unittest import TestCase

import remotior_sensus


class TestVectorToRaster(TestCase):

    def test_vector_to_raster(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        cfg.logger.log.debug('>>> test vector to raster')
        data_path = Path(__file__).parent / 'data'
        v = str(data_path / 'files' / 'roi.gpkg')
        r = str(data_path / 'S2_2020-01-01' / 'S2_B02.tif')
        temp = cfg.temp.temporary_file_path(
            name='raster', name_suffix=cfg.tif_suffix
        )
        raster = rs.vector_to_raster(vector_path=v, align_raster=r,
                                     constant=1, output_path=temp)
        self.assertTrue(rs.files_directories.is_file(raster.path))
        raster = rs.vector_to_raster(vector_path=v, align_raster=r,
                                     method='area_based', vector_field='class')
        self.assertTrue(rs.files_directories.is_file(raster.path))

        # clear temporary directory
        rs.close()
