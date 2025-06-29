from pathlib import Path
from unittest import TestCase

import remotior_sensus


class TestRasterLabel(TestCase):

    def test_raster_label(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        data_path = Path(__file__).parent / 'data'
        p = str(data_path / 'S2_2020-01-02' / 'S2_B02.tif')
        cfg.logger.log.debug('>>> test raster label')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.vrt_suffix)
        label = rs.raster_label(raster_path=p, output_path=temp,
                                virtual_output=True)
        self.assertTrue(rs.files_directories.is_file(label.path))

        # clear temporary directory
        rs.close()
