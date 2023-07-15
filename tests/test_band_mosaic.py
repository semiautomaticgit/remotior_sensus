from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestMosaicBands(TestCase):

    def test_mosaic_bands(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        catalog = rs.bandset_catalog()
        file_list = ['./data/S2_2020-01-01/S2_B02.tif',
                     './data/S2_2020-01-02/S2_B02.tif']
        temp = cfg.temp.dir
        cfg.logger.log.debug('>>> test mosaic')
        mosaic = rs.mosaic(file_list, temp)
        self.assertTrue(files_directories.is_file(mosaic.paths[0]))
        self.assertTrue(mosaic.check)
        file_list_1 = ['S2_2020-01-01/S2_B02.tif', 'S2_2020-01-01/S2_B03.tif',
                       'S2_2020-01-01/S2_B04.tif']
        file_list_2 = ['S2_2020-01-02/S2_B02.tif', 'S2_2020-01-02/S2_B03.tif',
                       'S2_2020-01-02/S2_B04.tif']
        file_list_3 = ['S2_2020-01-03/S2_B02.tif', 'S2_2020-01-03/S2_B03.tif',
                       'S2_2020-01-03/S2_B04.tif']
        root_directory = './data'
        catalog.create_bandset(
            file_list_1, wavelengths=['Sentinel-2'], bandset_number=1,
            root_directory=root_directory
            )
        catalog.create_bandset(
            file_list_2, wavelengths=['Sentinel-2'], bandset_number=2,
            root_directory=root_directory
            )
        catalog.create_bandset(
            file_list_3, wavelengths=['Sentinel-2'], bandset_number=3,
            root_directory=root_directory
            )
        bandset_list = [catalog.get_bandset(1), catalog.get_bandset(2),
                        catalog.get_bandset(3)]
        cfg.logger.log.debug('>>> test mosaic BandSet')
        mosaic = rs.mosaic(bandset_list, temp)
        self.assertTrue(files_directories.is_file(mosaic.paths[0]))
        self.assertTrue(mosaic.check)
        bandset_list = [1, 2]
        mosaic = rs.mosaic(
            bandset_list, output_path=temp, bandset_catalog=catalog
            )
        self.assertTrue(files_directories.is_file(mosaic.paths[0]))
        self.assertTrue(mosaic.check)

        band_list_1 = ['./data/S2_2020-01-01/S2_B02.tif',
                       './data/S2_2020-01-02/S2_B02.tif',
                       './data/S2_2020-01-02/S2_B02.tif']
        band_list_2 = ['./data/S2_2020-01-01/S2_B03.tif',
                       './data/S2_2020-01-02/S2_B03.tif',
                       './data/S2_2020-01-02/S2_B03.tif'
                       ]
        band_list_3 = ['./data/S2_2020-01-01/S2_B04.tif',
                       './data/S2_2020-01-02/S2_B04.tif',
                       './data/S2_2020-01-02/S2_B04.tif']
        band_list = [band_list_1, band_list_2, band_list_3]
        mosaic = rs.mosaic(
            band_list, output_path=temp, bandset_catalog=catalog,
            prefix='prefix', output_name='output_name'
            )
        self.assertTrue(files_directories.is_file(mosaic.paths[0]))
        self.assertTrue(mosaic.check)

        # clear temporary directory
        rs.close()
