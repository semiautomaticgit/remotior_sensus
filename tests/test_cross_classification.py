from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import read_write_files, files_directories


class TestCrossClassification(TestCase):

    def test_cross_classification(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        p1 = './data/S2_2020-01-01/S2_B04.tif'
        p2 = './data/S2_2020-01-01/S2_B02.tif'
        v = './data/files/roi.gpkg'
        cfg.logger.log.debug('>>> test cross_classification')
        coordinate_list = [230250, 4674550, 230320, 4674440]
        cross = rs.cross_classification(
            classification_path=p1, reference_path=v, vector_field='class',
            extent_list=coordinate_list
        )
        self.assertTrue(files_directories.is_file(cross.paths[0]))
        cross = rs.cross_classification(
            classification_path=p1, reference_path=p2, cross_matrix=True
        )
        raster, text = cross.paths
        table = read_write_files.open_text_file(text)
        table_f = read_write_files.format_csv_new_delimiter(
            table, cfg.tab_delimiter
        )
        self.assertGreater(len(table_f), 0)
        table_split = table.split(cfg.new_line)
        self.assertGreater(int(table_split[1][0]), 0)
        cfg.logger.log.debug('>>> test cross_classification accuracy matrix')
        cross = rs.cross_classification(
            classification_path=p1, reference_path=p2, error_matrix=True
        )
        raster, text = cross.paths
        table = read_write_files.open_text_file(text)
        table_f = read_write_files.format_csv_new_delimiter(
            table, cfg.tab_delimiter
        )
        self.assertGreater(len(table_f), 0)
        table_split = table.split(cfg.new_line)
        self.assertGreater(int(table_split[1][0]), 0)
        cfg.logger.log.debug('>>> test cross_classification regression')
        temp2 = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        cross = rs.cross_classification(
            classification_path=p1, reference_path=p2, output_path=temp2,
            regression_raster=True
        )
        raster, text = cross.paths
        table = read_write_files.open_text_file(text)
        table_f = read_write_files.format_csv_new_delimiter(
            table, cfg.tab_delimiter
        )
        self.assertGreater(len(table_f), 0)
        table_split = table.split(cfg.new_line)
        self.assertGreater(int(table_split[1][0]), 0)

        # clear temporary directory
        rs.close()
