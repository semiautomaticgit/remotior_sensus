from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestBandResample(TestCase):

    def test_band_resample(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        file_list = ['./data/S2_2020-01-01/S2_B02.tif',
                     './data/S2_2020-01-01/S2_B03.tif']
        cfg.logger.log.debug('>>> test band_sieve')
        resample = rs.band_resample(
            input_bands=file_list, output_path=cfg.temp.dir, resampling='mode',
            resample_pixel_factor=2, prefix='resample_'
            )
        self.assertTrue(files_directories.is_file(resample.paths[0]))

        # reproject band set (input files from bandset)
        reproject = rs.band_resample(
            input_bands=file_list, output_path=cfg.temp.dir,
            prefix='reproj_', epsg_code='32632', align_raster=None,
            resampling='nearest_neighbour', nodata_value=None,
            x_y_resolution=[20.0, 20.0], resample_pixel_factor=None,
            output_data_type=None, same_extent=False, virtual_output=False,
            compress=True, compress_format='LZW'
            )
        self.assertTrue(files_directories.is_file(reproject.paths[0]))

        # clear temporary directory
        rs.close()
