from unittest import TestCase

import remotior_sensus
from remotior_sensus.tools.raster_reclassification import (
    unique_values_table, _import_reclassification_table
)
from remotior_sensus.util import files_directories


class TestRasterReclassification(TestCase):

    def test_reclassification(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        p = './data/S2_2020-01-01/S2_B02.tif'
        cfg.logger.log.debug('>>> test unique values list')
        unique_list = unique_values_table(raster_path=p)
        self.assertGreater(len(unique_list), 0)
        unique_list = unique_values_table(raster_path=p, incremental=True)
        self.assertGreater(len(unique_list), 0)
        reclass_file = './data/files/reclass.csv'
        unique_list = _import_reclassification_table(csv_path=reclass_file)
        self.assertGreater(len(unique_list.extra['table']), 0)
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        reclassification_2 = rs.raster_reclassification(
            raster_path=p, output_path=temp,
            reclassification_table=unique_list.extra['table']
            )
        self.assertTrue(files_directories.is_file(reclassification_2.path))
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        reclassification = rs.raster_reclassification(
            raster_path=reclassification_2.path, output_path=temp,
            reclassification_table=[[1, -10], ['nan', 6000]]
            )
        self.assertTrue(files_directories.is_file(reclassification.path))
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        coordinate_list = [230250, 4674550, 230320, 4674440]
        reclassification = rs.raster_reclassification(
            raster_path=p, output_path=temp, extent_list=coordinate_list,
            reclassification_table=[['raster <= 3000', 1], ['raster > 500', 2]]
            )
        self.assertTrue(files_directories.is_file(reclassification.path))

        # clear temporary directory
        rs.close()
