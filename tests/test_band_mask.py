from pathlib import Path
from unittest import TestCase

import remotior_sensus


class TestBandMask(TestCase):

    def test_band_mask(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        cfg.logger.log.debug('>>> test band mask')
        catalog = rs.bandset_catalog()
        file_list = ['S2_2020-01-01/S2_B02.tif', 'S2_2020-01-01/S2_B03.tif',
                     'S2_2020-01-01/S2_B04.tif']
        data_path = Path(__file__).parent / 'data'
        catalog.create_bandset(
            file_list, wavelengths=['Sentinel-2'], bandset_number=1,
            root_directory=str(data_path)
        )
        cfg.logger.log.debug('>>> test band mask')
        v = str(data_path / 'files' / 'roi.gpkg')
        output = rs.band_mask(input_bands=1, input_mask=v,
                              bandset_catalog=catalog,
                              output_path=cfg.temp.dir)
        self.assertTrue(output.check)
        self.assertTrue(rs.files_directories.is_file(output.paths[0]))
        files = [
            str(data_path / 'S2_2020-01-01' / 'S2_B02.tif'),
            str(data_path / 'S2_2020-01-01' / 'S2_B03.tif'),
            str(data_path / 'S2_2020-01-01' / 'S2_B04.tif')
        ]
        mask_path = str(data_path / 'S2_2020-01-01' / 'S2_B02.tif')
        output = rs.band_mask(input_bands=files,
                              input_mask=mask_path,
                              prefix='mask_', buffer=1, mask_values=[1, 425],
                              nodata_value=-32768, virtual_output=True,
                              output_path=cfg.temp.dir)
        self.assertTrue(output.check)
        self.assertTrue(rs.files_directories.is_file(output.paths[0]))

        # clear temporary directory
        rs.close()
