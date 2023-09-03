import datetime
from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestBandSet(TestCase):

    def test_create(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        # create BandSet Catalog
        catalog = rs.bandset_catalog()
        cfg.logger.log.debug('test')
        data_directory = './data/S2_2020-01-01'
        file_list = files_directories.files_in_directory(
            data_directory, sort_files=True, path_filter='S',
            suffix_filter=cfg.tif_suffix
        )
        file_list.pop(0)
        band_names = []
        wavelengths = []
        multiplicative_factors = []
        additive_factors = []
        for f in range(len(file_list)):
            raster_name = files_directories.file_name(file_list[f], False)
            band_names.append(raster_name)
            wavelengths.append(f + 1)
            multiplicative_factors.append(f + 1)
            additive_factors.append(f)
        date = '2020-01-01'
        cfg.logger.log.debug('>>> test bandset create')
        bandset = rs.bandset.create(
            file_list, band_names=band_names, wavelengths=wavelengths,
            multiplicative_factors=multiplicative_factors,
            additive_factors=additive_factors, dates=date, catalog=catalog
        )
        # find bandset values in list
        bandset_values = bandset.find_values_in_list(
            attribute='wavelength', value_list=wavelengths[2:5],
            output_attribute='path'
        )
        self.assertEqual(bandset_values, file_list[2:5])
        self.assertGreater(len(str(bandset.crs)), 0)
        # get BandSet attributes function
        bs = bandset.get_band_attributes
        # BandSet band count
        bs_count = len(bandset.bands)
        self.assertEqual(bs_count, 11)
        # BandSet absolute paths
        bs_apaths = bandset.get_absolute_paths()
        self.assertEqual(bs_apaths, bs('path'))
        # BandSet relative paths
        bs_paths = bandset.get_paths()
        self.assertEqual(bs_paths, file_list)
        self.assertEqual(bs('path'), file_list)
        # BandSet date
        bs_date = bandset.date
        self.assertEqual(str(bs_date), date)
        self.assertEqual(
            bs('date')[0], datetime.datetime.strptime(date, '%Y-%m-%d').date()
        )
        # multiplicative and additive factors
        bs_multi = bs('multiplicative_factor')
        self.assertEqual(bs_multi, multiplicative_factors)
        bs_add = bs('additive_factor')
        self.assertEqual(bs_add, additive_factors)
        cfg.logger.log.debug('>>> test bandset satellite')
        bandset = rs.bandset.create(
            [data_directory, 'tif'], wavelengths=['Sentinel-2'],
            dates=cfg.date_auto, catalog=catalog
        )
        # BandSet date
        self.assertEqual(str(bandset.date), date)
        # unit
        self.assertEqual(bandset.get_wavelength_units()[0], cfg.wl_micro)
        # spectral range bands
        (blue_band, green_band, red_band, nir_band, swir_1_band,
         swir_2_band) = bandset.spectral_range_bands(output_as_number=False)
        self.assertAlmostEqual(blue_band['wavelength'], cfg.blue_center, 1)
        self.assertAlmostEqual(red_band['wavelength'], cfg.red_center, 1)
        self.assertAlmostEqual(nir_band['wavelength'], cfg.nir_center, 1)
        self.assertEqual(
            blue_band['wavelength_unit'], bandset.get_wavelength_units()[0]
        )
        self.assertGreater(len(red_band['crs']), 0)
        self.assertGreater(len(nir_band['path']), 0)
        cfg.logger.log.debug('>>> test methods')
        # get band by nearest wavelength
        band_x = bandset.get_band_by_wavelength(
            cfg.green_center, threshold=0.1
        )
        self.assertEqual(band_x['wavelength'], cfg.green_center)
        # get bands by attribute
        band_b = bandset.get_bands_by_attributes(
            'wavelength', band_x['wavelength']
        )
        self.assertEqual(band_b['wavelength'], band_x['wavelength'])
        file_list = ['S2_2020-01-01/S2_B01.tif', 'S2_2020-01-01/S2_B02.tif',
                     'S2_2020-01-01/S2_B03.tif']
        cfg.logger.log.debug('>>> test bandset dates')
        bandset = rs.bandset.create(
            file_list, wavelengths=['Sentinel-2'], dates=cfg.date_auto,
            root_directory='./data', catalog=catalog
        )
        # get BandSet attributes function
        bs = bandset.get_band_attributes
        # BandSet names
        bs_names = bs('name')
        self.assertEqual(
            bs_names[0], files_directories.file_name(file_list[0], False)
        )
        # BandSet relative paths
        bs_paths = bandset.get_paths()
        self.assertEqual(bs_paths[0], file_list[0])
        # bandset wavelengths
        bs_wls = bandset.get_wavelengths()
        self.assertEqual(bs_wls, bs('wavelength'))
        self.assertEqual(bandset.get_wavelengths(), bs('wavelength'))
        self.assertEqual(
            bandset.get_band_attributes('wavelength')[0],
            cfg.satellites[cfg.satSentinel2][0][0]
        )
        # get BandSet band function
        band = bandset.get_band
        # band wavelength
        b_wl = band(1)['wavelength']
        self.assertEqual(b_wl, cfg.satellites[cfg.satSentinel2][0][0])
        # band absolute path
        b_apath = bandset.get_absolute_path(band_number=1)
        self.assertEqual(
            files_directories.file_name(b_apath),
            files_directories.file_name(file_list[0], False)
        )
        # band x size
        b_x_size = band(1).x_size
        self.assertGreater(b_x_size, 0)
        # band nodata
        b_nodata = band(1).nodata
        self.assertGreater(b_nodata, 0)
        # reset
        bandset.reset()
        self.assertEqual(bandset.bands, None)
        # sort bands by wavelength
        wavelengths = list(range(1, len(file_list) + 1))
        wavelengths.pop(-1)
        wavelengths.pop(-1)
        wavelengths.append(3)
        wavelengths.append(2)
        bandset_w = rs.bandset.create(
            file_list, wavelengths=wavelengths, root_directory='./data',
            catalog=catalog
        )
        bandset_w.sort_bands_by_wavelength()
        self.assertEqual(bandset_w.get_band_attributes('wavelength')[-1], 3)
        # box coordinate list
        coordinate_list = [230250, 4674550, 230320, 4674440]
        bandset = rs.bandset.create(
            file_list, wavelengths=['Sentinel-2'], dates=cfg.date_auto,
            root_directory='./data', box_coordinate_list=coordinate_list,
            catalog=catalog
        )
        self.assertEqual(bandset.box_coordinate_list, coordinate_list)
        # export as xml
        temp = cfg.temp.temporary_file_path(name_suffix='.xml')
        bandset.export_as_xml(temp)
        bandset2 = rs.bandset.create(file_list, root_directory='./data',
                                     catalog=catalog)
        bandset2.import_as_xml(temp)
        for i in range(len(bandset.bands[0])):
            if str(bandset.bands[0][i]) != 'NaT':
                self.assertEqual(bandset.bands[0][i], bandset2.bands[0][i])

        """ commented because time consuming
        cfg.logger.log.debug('>>> test tools')
        expression = (
                cfg.variable_band_quotes + cfg.variable_ndvi_name
                + cfg.variable_band_quotes
        )
        calc = bandset.calc(expression)
        self.assertTrue(calc.check)
        self.assertTrue(files_directories.is_file(calc.paths[0]))
        combination = bandset.combination()
        self.assertTrue(combination.check)
        self.assertTrue(files_directories.is_file(combination.paths[0]))
        dilation = bandset.dilation(value_list=[1000], size=2)
        self.assertTrue(dilation.check)
        self.assertTrue(files_directories.is_file(dilation.paths[0]))
        erosion = bandset.erosion(value_list=[1000], size=1)
        self.assertTrue(erosion.check)
        self.assertTrue(files_directories.is_file(erosion.paths[0]))
        pca = bandset.pca()
        self.assertTrue(pca.check)
        self.assertTrue(files_directories.is_file(pca.paths[0]))
        sieve = bandset.sieve(size=3, connected=True)
        self.assertTrue(sieve.check)
        self.assertTrue(files_directories.is_file(sieve.paths[0]))
        neighbor_pixels = bandset.neighbor_pixels(
            size=10, circular_structure=True, stat_name='Sum'
        )
        self.assertTrue(neighbor_pixels.check)
        self.assertTrue(files_directories.is_file(neighbor_pixels.paths[0]))
        """

        # clear temporary directory
        rs.close()
