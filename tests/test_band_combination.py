from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import read_write_files


class TestBandCombination(TestCase):

    def test_band_combination(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        catalog = rs.bandset_catalog()
        file_list = ['S2_2020-01-01/S2_B02.tif', 'S2_2020-01-01/S2_B03.tif',
                     'S2_2020-01-01/S2_B04.tif']
        date = '2021-01-01'
        root_directory = './data'
        catalog.create_bandset(
            file_list, wavelengths=['Sentinel-2'], date=date, bandset_number=1,
            root_directory=root_directory
            )
        cfg.logger.log.debug('>>> test band combination input BandSet')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        combination = rs.band_combination(
            input_bands=catalog.get_bandset(1), output_path=temp
            )
        raster, text = combination.paths
        table = read_write_files.open_text_file(text)
        table_f = read_write_files.format_csv_new_delimiter(
            table, cfg.tab_delimiter
            )
        self.assertGreater(len(table_f), 0)
        cfg.logger.log.debug('>>> test band combination input BandSet number')
        bs = catalog.get_bandset(1)
        bs.box_coordinate_list = [230250, 4674550, 230320, 4674440]
        combination = rs.band_combination(
            input_bands=1, bandset_catalog=catalog
            )
        raster, text = combination.paths
        table = read_write_files.open_text_file(text)
        table_f = read_write_files.format_csv_new_delimiter(
            table, cfg.tab_delimiter
            )
        self.assertGreater(len(table_f), 0)
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        file_list = ['./data/S2_2020-01-01/S2_B02.tif',
                     './data/S2_2020-01-01/S2_B03.tif',
                     './data/S2_2020-01-01/S2_B04.tif']
        cfg.logger.log.debug('>>> test band combination input file list')
        combination = rs.band_combination(input_bands=file_list,
                                          output_path=temp)
        raster, text = combination.paths
        table = read_write_files.open_text_file(text)
        table_f = read_write_files.format_csv_new_delimiter(
            table, cfg.tab_delimiter
            )
        self.assertGreater(len(table_f), 0)
        table_split = table.split(cfg.new_line)
        self.assertGreater(int(table_split[1][0]), 0)
        cfg.logger.log.debug('>>> test band combination without output table')
        combination = rs.band_combination(
            input_bands=file_list, output_table=False
            )
        combinations_array = combination.extra['combinations']
        self.assertGreater(combinations_array.shape[0], 1)

        # clear temporary directory
        rs.close()
