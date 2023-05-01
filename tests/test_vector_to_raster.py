from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestVectorToRaster(TestCase):

    def test_vector_to_raster(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        v = './data/files/roi.gpkg'
        r = './data/S2_2020-01-01/S2_B02.tif'
        temp = cfg.temp.temporary_file_path(
            name='raster', name_suffix=cfg.tif_suffix
        )
        raster = rs.vector_to_raster(vector_path=v, align_raster=r,
                                     constant=1, output_path=temp)
        self.assertTrue(files_directories.is_file(raster.path))
        raster = rs.vector_to_raster(vector_path=v, align_raster=r,
                                     vector_field='class')
        self.assertTrue(files_directories.is_file(raster.path))
        temp = cfg.temp.temporary_file_path(
            name='raster', name_suffix=cfg.tif_suffix
        )
        raster = rs.vector_to_raster(vector_path=v, align_raster=r,
                                     vector_field='class', method='area_based',
                                     area_precision=2,
                                     minimum_extent=False, output_path=temp)
        self.assertTrue(files_directories.is_file(raster.path))

        # clear temporary directory
        rs.close()
