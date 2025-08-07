from pathlib import Path
from unittest import TestCase

import remotior_sensus


class TestRasterEdit(TestCase):

    def test_raster_edit(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        data_path = Path(__file__).parent / 'data'
        p = str(data_path / 'S2_2020-01-01' / 'S2_B02.tif')
        v = str(data_path / 'files' / 'roi.gpkg')
        cfg.logger.log.debug('>>> test raster edit')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        rs.files_directories.copy_file(in_path=p, out_path=temp)
        edit = rs.raster_edit(
            raster_path=temp, vector_path=v, constant_value=10
        )
        self.assertTrue(edit.extra['column_start'] > 0)
        cfg.logger.log.debug('>>> test undo raster edit')
        edit_2 = rs.raster_edit(
            raster_path=temp, column_start=edit.extra['column_start'],
            row_start=edit.extra['row_start'],
            old_array=edit.extra['old_array']
        )
        self.assertTrue(edit_2.extra['column_start'] is None)
        cfg.logger.log.debug('>>> test raster edit with expression')
        temp_3 = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        rs.files_directories.copy_file(in_path=p, out_path=temp_3)
        expression = 'where(%s%s%s > 500, 10, %s%s%s)' % (
            cfg.variable_band_quotes, cfg.variable_raster_name,
            cfg.variable_band_quotes, cfg.variable_band_quotes,
            cfg.variable_raster_name, cfg.variable_band_quotes
        )
        edit_3 = rs.raster_edit(
            raster_path=temp_3, vector_path=v, expression=expression
        )
        self.assertTrue(edit_3.extra['column_start'] > 0)

        # clear temporary directory
        rs.close()
