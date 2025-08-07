from pathlib import Path
from unittest import TestCase

import remotior_sensus


class TestBandErosion(TestCase):

    def test_band_erosion(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        data_path = Path(__file__).parent / 'data'
        file_list = [
            str(data_path / 'S2_2020-01-01' / 'S2_B02.tif'),
            str(data_path / 'S2_2020-01-01' / 'S2_B03.tif')
        ]
        cfg.logger.log.debug('>>> test band_erosion')
        erosion = rs.band_erosion(
            input_bands=file_list, output_path=cfg.temp.dir,
            value_list=[1, 425], size=1, circular_structure=True,
            prefix='erosion_'
        )
        self.assertTrue(rs.files_directories.is_file(erosion.paths[0]))

        # clear temporary directory
        rs.close()
