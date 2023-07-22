from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestBandMask(TestCase):

    def test_band_mask(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        catalog = rs.bandset_catalog()
        file_list = ['S2_2020-01-01/S2_B02.tif', 'S2_2020-01-01/S2_B03.tif',
                     'S2_2020-01-01/S2_B04.tif']
        root_directory = './data'
        catalog.create_bandset(
            file_list, wavelengths=['Sentinel-2'], bandset_number=1,
            root_directory=root_directory
            )
        cfg.logger.log.debug('>>> test band mask')
        v = './data/files/roi.gpkg'
        output = rs.band_mask(input_bands=1, input_mask=v,
                              bandset_catalog=catalog,
                              output_path=cfg.temp.dir)
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))

        files = ['./data/S2_2020-01-01/S2_B02.tif',
                 './data/S2_2020-01-01/S2_B03.tif',
                 './data/S2_2020-01-01/S2_B04.tif']
        output = rs.band_mask(input_bands=files,
                              input_mask='./data/S2_2020-01-01/S2_B02.tif',
                              prefix="mask_", buffer=1, mask_values=[1, 425],
                              nodata_value=-32768, virtual_output=True,
                              output_path=cfg.temp.dir)
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))

        # clear temporary directory
        rs.close()
