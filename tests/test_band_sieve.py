from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestBandSieve(TestCase):

    def test_band_sieve(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        file_list = ['./data/S2_2020-01-01/S2_B02.tif',
                     './data/S2_2020-01-01/S2_B03.tif']
        cfg.logger.log.debug('>>> test band_sieve')
        sieve = rs.band_sieve(
            input_bands=file_list, output_path=cfg.temp.dir, size=2,
            connected=False, prefix='sieve_'
            )
        self.assertTrue(files_directories.is_file(sieve.paths[0]))

        coordinate_list = [230250, 4674550, 230320, 4674440]
        sieve = rs.band_sieve(
            input_bands=file_list, output_path=cfg.temp.dir, size=2,
            connected=False, prefix='sieve_', extent_list=coordinate_list
            )
        self.assertTrue(files_directories.is_file(sieve.paths[0]))

        # clear temporary directory
        rs.close()
