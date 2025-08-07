from pathlib import Path
from unittest import TestCase

import remotior_sensus


class TestMosaicBands(TestCase):

    def test_mosaic_bands(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        catalog = rs.bandset_catalog()
        data_path = Path(__file__).parent / 'data'
        file_list = [
            str(data_path / 'S2_2020-01-01' / 'S2_B02.tif'),
            str(data_path / 'S2_2020-01-02' / 'S2_B02.tif')
        ]
        temp = cfg.temp.dir
        cfg.logger.log.debug('>>> test mosaic')
        mosaic = rs.mosaic(file_list, temp)
        self.assertTrue(rs.files_directories.is_file(mosaic.paths[0]))
        self.assertTrue(mosaic.check)
        file_list_1 = ['S2_2020-01-01/S2_B02.tif', 'S2_2020-01-01/S2_B03.tif',
                       'S2_2020-01-01/S2_B04.tif']
        file_list_2 = ['S2_2020-01-02/S2_B02.tif', 'S2_2020-01-02/S2_B03.tif',
                       'S2_2020-01-02/S2_B04.tif']
        file_list_3 = ['S2_2020-01-03/S2_B02.tif', 'S2_2020-01-03/S2_B03.tif',
                       'S2_2020-01-03/S2_B04.tif']
        catalog.create_bandset(
            file_list_1, wavelengths=['Sentinel-2'], bandset_number=1,
            root_directory=str(data_path)
        )
        catalog.create_bandset(
            file_list_2, wavelengths=['Sentinel-2'], bandset_number=2,
            root_directory=str(data_path)
        )
        catalog.create_bandset(
            file_list_3, wavelengths=['Sentinel-2'], bandset_number=3,
            root_directory=str(data_path)
        )
        bandset_list = [catalog.get_bandset(1), catalog.get_bandset(2),
                        catalog.get_bandset(3)]
        cfg.logger.log.debug('>>> test mosaic BandSet')
        mosaic = rs.mosaic(bandset_list, temp)
        self.assertTrue(rs.files_directories.is_file(mosaic.paths[0]))
        self.assertTrue(mosaic.check)
        bandset_list = [1, 2]
        mosaic = rs.mosaic(
            bandset_list, output_path=temp, bandset_catalog=catalog
        )
        self.assertTrue(rs.files_directories.is_file(mosaic.paths[0]))
        self.assertTrue(mosaic.check)

        band_list_1 = [
            str(data_path / 'S2_2020-01-01' / 'S2_B02.tif'),
            str(data_path / 'S2_2020-01-02' / 'S2_B02.tif'),
            str(data_path / 'S2_2020-01-02' / 'S2_B02.tif')
        ]
        band_list_2 = [
            str(data_path / 'S2_2020-01-01' / 'S2_B03.tif'),
            str(data_path / 'S2_2020-01-02' / 'S2_B03.tif'),
            str(data_path / 'S2_2020-01-02' / 'S2_B03.tif')
        ]
        band_list_3 = [
            str(data_path / 'S2_2020-01-01' / 'S2_B04.tif'),
            str(data_path / 'S2_2020-01-02' / 'S2_B04.tif'),
            str(data_path / 'S2_2020-01-02' / 'S2_B04.tif')
        ]
        band_list = [band_list_1, band_list_2, band_list_3]
        mosaic = rs.mosaic(
            band_list, output_path=temp, bandset_catalog=catalog,
            prefix='prefix', output_name='output_name'
        )
        self.assertTrue(rs.files_directories.is_file(mosaic.paths[0]))
        self.assertTrue(mosaic.check)

        # clear temporary directory
        rs.close()
