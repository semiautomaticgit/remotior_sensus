from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import download_tools, files_directories


class TestDownloadTools(TestCase):

    def test_download_tools(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        url = 'https://www.python.org'
        temp = cfg.temp.temporary_file_path(name_suffix='.html')
        download_tools.download_file(url=url, output_path=temp)
        self.assertTrue(files_directories.is_file(temp))
        url_2 = ''.join(
            ['https://storage.googleapis.com/gcp-public-data-sentinel-2/',
             'L2/tiles/33/S/VB/S2A_MSIL2A_20210104T094411_N0214_R036_T33S',
             'VB_20210104T122314.SAFE/GRANULE/L2A_T33SVB_A028919_20210104',
             'T094407/IMG_DATA/R60m/T33SVB_20210104T094411_B01_60m.jp2']
        )
        temp_2 = cfg.temp.temporary_file_path(name_suffix='.jp2')
        download_tools.download_file(url=url_2, output_path=temp_2)
        self.assertTrue(files_directories.is_file(temp_2))

        # clear temporary directory
        rs.close()
