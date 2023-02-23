from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import read_write_files


class TestRasterReport(TestCase):

    def test_report(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        p = './data/S2_2020-01-01/S2_B02.tif'
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.csv_suffix)
        cfg.logger.log.debug('>>> test raster_report')
        report = rs.raster_report(p, temp)
        table = read_write_files.open_text_file(report.path)
        table_f = read_write_files.format_csv_new_delimiter(
            table, cfg.tab_delimiter
            )
        self.assertGreater(len(table_f), 0)
        table_html = read_write_files.format_csv_text_html(table)
        self.assertGreater(len(table_html), 0)
        table_split = table.split(cfg.new_line)
        self.assertGreater(int(table_split[1][0]), 0)

        # clear temporary directory
        rs.close()
