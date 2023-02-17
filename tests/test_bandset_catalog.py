from unittest import TestCase

import numpy as np

import remotior_sensus


class TestBandSetCatalog(TestCase):

    def test_bandset_catalog(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        data_directory = './data/S2_2020-01-01'
        cfg.logger.log.debug('>>> test bandset catalog create')
        # create BandSet Catalog
        catalog = rs.bandset_catalog()
        # BandSet count
        self.assertEqual(catalog.get_bandset_count(), 1)
        # empty BandSet
        catalog.create_bandset(insert=True, bandset_name='BandSet X')
        # BandSet count
        self.assertEqual(catalog.get_bandset_count(), 2)
        # BandSet from directory
        catalog.create_bandset(
            paths=[data_directory, 'tif'], wavelengths=['Sentinel-2'],
            date=cfg.date_auto, bandset_number=1, bandset_name='bandset_1'
        )
        file_list = ['S2_2020-01-01/S2_B02.tif', 'S2_2020-01-01/S2_B03.tif',
                     'S2_2020-01-01/S2_B04.tif']
        date = '2021-01-01'
        root_directory = './data'
        catalog.create_bandset(
            file_list, wavelengths=['Sentinel-2'], date=date, bandset_number=2,
            root_directory=root_directory
        )
        # find BandSet from name list
        name_list = ['ba', 'S2_B']
        bandset_list = catalog.find_bandset_names_in_list(names=name_list)
        self.assertEqual(len(bandset_list), 2)
        bandset_list = catalog.find_bandset_names_in_list(
            names=name_list, exact_match=True
        )
        self.assertEqual(len(bandset_list), 1)
        # box coordinate list
        coordinate_list = [230250, 4674550, 230320, 4674440]
        catalog.create_bandset(
            file_list, wavelengths=['Sentinel-2'], date=date, bandset_number=3,
            root_directory=root_directory, box_coordinate_list=coordinate_list
        )
        self.assertEqual(
            catalog.get(3).box_coordinate_list,
            catalog.get_box_coordinate_list(3)
        )
        self.assertEqual(coordinate_list, catalog.get_box_coordinate_list(3))
        # set current BandSet
        catalog.current_bandset = 10
        # current BandSet
        self.assertEqual(catalog.current_bandset, catalog.get_bandset_count())
        # create second catalog
        catalog_2 = rs.bandset_catalog()
        self.assertEqual(str(catalog_2.bandsets_table['date'][0]), 'NaT')
        # BandSet count
        self.assertEqual(catalog.get_bandset_count(), 3)
        self.assertEqual(catalog_2.get_bandset_count(), 1)
        # get BandSet function
        bs = catalog.get_bandset
        # get BandSet
        self.assertEqual(bs(1), catalog.get_bandset(1))
        self.assertEqual(type(catalog.get_bandset(2)), rs.bandset)
        self.assertGreater(len(str(catalog.get_bandset(1).crs)), 0)
        # get bands
        self.assertEqual(
            bs(2).bands['name'].tolist(),
            catalog.get_bandset(2).bands['name'].tolist()
        )
        self.assertEqual(type(catalog.get_bandset(1).bands), np.recarray)
        # BandSet absolute paths
        self.assertEqual(
            bs(2).get_absolute_paths(), catalog.get_bandset(2, 'absolute_path')
        )
        # BandSet relative paths
        self.assertEqual(bs(2).get_paths(), bs(2, 'path'))
        self.assertEqual(
            catalog.get_bandset(2).get_paths(), catalog.get_bandset(2, 'path')
        )
        self.assertEqual(
            catalog.get_bandset(2).get_band_attributes('path'), file_list
        )
        # BandSet date
        self.assertEqual(str(bs(2, 'date')[0]), date)
        self.assertEqual(bs(2).date, catalog.get_bandset(2).date)
        self.assertEqual(
            bs(2).date, str(catalog.get_bandset_catalog_attributes(2, 'date'))
        )
        self.assertEqual(str(catalog.get_bandset(2, 'date')[0]), date)
        date_current = catalog.get_bandset_catalog_attributes(
            catalog.current_bandset, 'date'
        )
        self.assertEqual(
            date_current, catalog.get_bandset(
                catalog.current_bandset
            ).get_band_attributes('date')[0]
        )
        # BandSet wavelengths
        self.assertEqual(
            bs(2).get_wavelengths(), catalog.get_bandset(2).get_wavelengths()
        )
        names = catalog.iterate_bandset_bands('name')
        self.assertGreater(len(names), 0)
        # move band in BandSet
        band_2 = catalog.get_bandset(1).get_band(2).name
        catalog.move_band_in_bandset(
            bandset_number=1, band_number_input=2, band_number_output=5
        )
        self.assertEqual(band_2, catalog.get_bandset(1).get_band(5).name[0])
        # add band to BandSet
        bandset_1_count = catalog.get_bandset(1).get_band_count()
        catalog.add_band_to_bandset(
            path='./data/S2_2020-01-01/S2_B02.tif', bandset_number=1,
            band_number=1,
            raster_band=1
        )
        self.assertGreater(
            catalog.get_bandset(1).get_band_count(), bandset_1_count
        )
        band_1_name = catalog.get(1).get(1).name
        # sort bands by wavelength
        catalog.sort_bands_by_wavelength(bandset_number=1)
        # get band by nearest wavelength
        band_1 = catalog.get_bandset(1).get_band_by_wavelength(wavelength=1)
        self.assertEqual(band_1.name, band_1_name)
        # get BandSet band from attribute
        band_w = catalog.get_bandset_bands_by_attribute(
            bandset_number=1, attribute='wavelength',
            attribute_value=catalog.get(
                1
                ).get(1).wavelength
            )
        self.assertEqual(
            band_w[0].wavelength, catalog.get_bandset(1).get_band(1).wavelength
        )
        # remove band in BandSet
        band_number = catalog.get_bandset(1).get_band_by_wavelength(
            1, output_as_number=True
        )
        catalog.remove_band_in_bandset(
            bandset_number=1, band_number=band_number
        )
        self.assertEqual(
            catalog.get_bandset(1).get_band_count(), bandset_1_count
        )
        # get BandSet by number
        bandset_by_number = catalog.get_bandset_by_number(1)
        self.assertEqual(bandset_by_number.get_band_count(), bandset_1_count)
        # get BandSet by name
        bandset_by_name = catalog.get_bandset_by_name('bandset_1')
        self.assertEqual(bandset_by_name.get_band_count(), bandset_1_count)
        # get band list
        band_list = rs.bandset_catalog.get_band_list(1, catalog)
        self.assertGreater(len(band_list), 1)
        # load BandSet catalog
        empty_catalog = rs.bandset_catalog()
        self.assertEqual(empty_catalog.get_bandset_count(), 1)
        empty_catalog._load(bandset_catalog=catalog, current_bandset=1)
        self.assertEqual(
            empty_catalog.get_bandset_count(), catalog.get_bandset_count()
        )
        empty_catalog.add_bandset(
            catalog.get_bandset(1), bandset_number=1, insert=True
        )
        self.assertEqual(
            empty_catalog.get_bandset_count(), catalog.get_bandset_count() + 1
        )
        empty_catalog.move_bandset(
            bandset_number_input=1, bandset_number_output=3
        )
        self.assertEqual(
            empty_catalog.get_bandset(3).get_band_attributes('name'),
            catalog.get_bandset(1).get_band_attributes('name')
        )
        # set BandSet root directory
        empty_catalog.set_root_directory(
            bandset_number=1, root_directory=data_directory
        )
        self.assertEqual(
            empty_catalog.get(1).root_directory,
            empty_catalog.get_root_directory(1)
        )
        # set BandSet date
        empty_catalog.set_date(bandset_number=1, date=date)
        self.assertEqual(empty_catalog.get(1).date, empty_catalog.get_date(1))
        # BandSet band count
        self.assertEqual(
            empty_catalog.get_band_count(bandset_number=1),
            catalog.get_band_count(bandset_number=1)
        )
        # set BandSet name
        empty_catalog.set_name(bandset_number=1, name='new_name')
        self.assertEqual(empty_catalog.get(1).name, empty_catalog.get_name(1))
        # set BandSet box coordinate list
        empty_catalog.set_box_coordinate_list(
            bandset_number=1, box_coordinate_list=coordinate_list
        )
        self.assertEqual(
            empty_catalog.get_box_coordinate_list(1), coordinate_list
        )
        band_string = empty_catalog.create_band_string_list(bandset_number=1)
        self.assertGreater(len(band_string), 0)
        # remove BandSets
        empty_catalog.remove_bandset(1)
        empty_catalog.remove_bandset(1)
        empty_catalog.remove_bandset(1)
        empty_catalog.remove_bandset(1)
        self.assertEqual(empty_catalog.get_bandset(1).get_band_count(), 0)

        # clear temporary directory
        rs.close()
