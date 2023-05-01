from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import raster_vector, files_directories


class TestRasterVector(TestCase):

    def test_raster_vector(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        p = './data/S2_2020-01-01/S2_B02.tif'
        crs = raster_vector.get_crs(p)
        self.assertGreater(len(crs), 0)
        raster_list = ['./data/S2_2020-01-01/S2_B02.tif',
                       './data/S2_2020-01-03/S2_B02.tif']
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.vrt_suffix)
        cfg.logger.log.debug('>>> test create_virtual_raster')
        raster_vector.create_virtual_raster(
            input_raster_list=raster_list, output=temp, relative_to_vrt=False
            )
        self.assertTrue(files_directories.is_file(temp))
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.vrt_suffix)
        cfg.logger.log.debug('>>> test create_virtual_raster_2_mosaic')
        raster_vector.create_virtual_raster_2_mosaic(
            input_raster_list=raster_list, output=temp
            )
        self.assertTrue(files_directories.is_file(temp))
        p = './data/S2_2020-01-01/S2_B02.tif'
        cfg.logger.log.debug('>>> test read raster')
        n = raster_vector.get_number_bands(p)
        self.assertEqual(n, 1)
        # virtual raster of BandSet
        cfg.logger.log.debug('>>> test create_virtual_raster BandSet')
        # box coordinate list
        catalog = rs.bandset_catalog()
        file_list = ['S2_2020-01-01/S2_B02.tif', 'S2_2020-01-01/S2_B03.tif',
                     'S2_2020-01-01/S2_B04.tif']
        root_directory = './data'
        coordinate_list = [230250, 4674550, 230320, 4674440]
        catalog.create_bandset(
            file_list, wavelengths=['Sentinel-2'],
            root_directory=root_directory, box_coordinate_list=coordinate_list
            )
        temp_2 = cfg.temp.temporary_file_path(name_suffix=cfg.vrt_suffix)
        raster_vector.create_virtual_raster(
            bandset=catalog.get(1), output=temp_2
            )
        self.assertTrue(files_directories.is_file(temp_2))

        # clear temporary directory
        rs.close()
