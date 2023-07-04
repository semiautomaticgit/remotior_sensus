from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestBandStackSplit(TestCase):

    def test_band_stack_split(self):
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
        cfg.logger.log.debug('>>> test band stack')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        stack = rs.band_stack(input_bands=1, output_path=temp,
                              bandset_catalog=catalog)
        cfg.logger.log.debug('>>> test raster split')
        self.assertTrue(files_directories.is_file(stack.path))
        split = rs.raster_split(raster_path=stack.path,
                                output_path=cfg.temp.dir)
        self.assertTrue(files_directories.is_file(split.paths[0]))

        # clear temporary directory
        rs.close()
