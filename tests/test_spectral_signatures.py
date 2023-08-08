from unittest import TestCase

import remotior_sensus


class TestSpectralSignatures(TestCase):

    def test_spectral_signatures(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        # create Spectral Signature Catalog
        cfg.logger.log.debug('>>> test create Spectral Signature Catalog')
        signature_catalog = rs.spectral_signatures_catalog()
        value_list = [1.1, 1.2, 1.3, 1.4]
        wavelength_list = [1, 2, 3, 4]
        signature_catalog.add_spectral_signature(
            value_list=value_list, wavelength_list=wavelength_list
        )
        self.assertEqual(len(signature_catalog.signatures), 1)
        signature_catalog.add_spectral_signature(
            value_list=value_list, wavelength_list=wavelength_list,
            macroclass_id=2
        )
        self.assertEqual(signature_catalog.table.shape[0], 2)
        cfg.logger.log.debug('>>> test save Spectral Signature Catalog')
        temp = cfg.temp.temporary_file_path(name_suffix='.sscx')
        signature_catalog.save(output_path=temp)
        self.assertTrue(rs.files_directories.is_file(temp))
        signature_catalog_x = rs.spectral_signatures_catalog()
        signature_catalog_x.load(file_path=temp)
        self.assertEqual(signature_catalog.table.shape[0],
                         signature_catalog_x.table.shape[0])
        self.assertEqual(signature_catalog.table[0],
                         signature_catalog_x.table[0])
        self.assertEqual(len(signature_catalog.signatures),
                         len(signature_catalog_x.signatures))
        # create BandSet
        cfg.logger.log.debug('>>> test create BandSet')
        catalog = rs.bandset_catalog()
        file_list = ['S2_2020-01-01/S2_B02.tif', 'S2_2020-01-01/S2_B03.tif',
                     'S2_2020-01-01/S2_B04.tif', 'S2_2020-01-01/S2_B05.tif']
        root_directory = 'data'
        catalog.create_bandset(
            file_list, wavelengths=['Sentinel-2'],
            root_directory=root_directory
            )
        # set BandSet in SpectralCatalog
        signature_catalog_2 = rs.spectral_signatures_catalog(
            bandset=catalog.get(1)
        )
        # add spectral signature getting wavelength from BandSet
        signature_catalog_2.add_spectral_signature(value_list=value_list)
        self.assertEqual(
            signature_catalog_2.signatures[
                list(signature_catalog_2.signatures.keys())[0]][
                'wavelength'].tolist(), catalog.get(1).get_wavelengths()
        )
        # add spectral signature with macroclass_id = 2
        signature_catalog_2.add_spectral_signature(
            value_list=value_list, macroclass_id=2
        )
        self.assertEqual(
            signature_catalog_2.table[
                signature_catalog_2.table[
                    'macroclass_id'] == 2].macroclass_id, 2
        )
        # import spectral signature csv
        cfg.logger.log.debug('>>> test import spectral signature csv')
        # signature with standard deviation
        signature_catalog_2.import_spectral_signature_csv(
            csv_path='data/files/spectral_signature_1.csv', macroclass_id=4,
            class_id=4, macroclass_name='imported_1', class_name='imported_1'
        )
        self.assertEqual(
            signature_catalog_2.table[
                signature_catalog_2.table['macroclass_id'] == 4].macroclass_id,
            4
        )
        # signature without standard deviation
        signature_catalog_2.import_spectral_signature_csv(
            csv_path='data/files/spectral_signature_2.csv', macroclass_id=5,
            class_id=5, macroclass_name='imported_2', class_name='imported_2'
        )
        self.assertEqual(
            signature_catalog_2.table[
                signature_catalog_2.table[
                    'macroclass_id'] == 5].macroclass_id,
            5
            )
        # import vector
        cfg.logger.log.debug('>>> test import vector')
        signature_catalog_2.import_vector(
            file_path='data/files/roi.gpkg', macroclass_value=7,
            class_field='class', macroclass_name_field='macroclass',
            class_name_field='class', calculate_signature=True
        )
        self.assertEqual(
            signature_catalog_2.table[
                signature_catalog_2.table['macroclass_id'] == 4].macroclass_id,
            4
        )
        # import vector
        cfg.logger.log.debug('>>> test remove signature')
        temp = cfg.temp.temporary_file_path(name_suffix='.sscx')
        signature_catalog_2.save(output_path=temp)
        self.assertTrue(rs.files_directories.is_file(temp))
        sig_id = signature_catalog_2.table[
            signature_catalog_2.table['macroclass_id'] == 7].signature_id[0]
        signature_catalog_2.remove_signature_by_id(signature_id=sig_id)
        temp = cfg.temp.temporary_file_path(name_suffix='.sscx')
        signature_catalog_2.save(output_path=temp)
        self.assertTrue(rs.files_directories.is_file(temp))

        # region growing
        cfg.logger.log.debug('>>> test region growing')
        catalog_3 = rs.bandset_catalog()
        file_list = ['S2_2020-01-04/S2_B02.tif', 'S2_2020-01-04/S2_B03.tif',
                     'S2_2020-01-04/S2_B04.tif']
        root_directory = 'data'
        catalog_3.create_bandset(
            file_list, wavelengths=['Sentinel-2'],
            root_directory=root_directory
            )
        region_vector = rs.shared_tools.region_growing_polygon(
            coordinate_x=230303, coordinate_y=4674704,
            input_bands=1, max_width=4,
            max_spectral_distance=0.1, minimum_size=1,
            bandset_catalog=catalog_3
        )
        self.assertTrue(rs.files_directories.is_file(region_vector))
        # set BandSet in SpectralCatalog
        signature_catalog_3 = rs.spectral_signatures_catalog(
            bandset=catalog_3.get(1)
        )
        signature_catalog_3.import_vector(
            file_path=region_vector, macroclass_value=5,
            class_value=11, macroclass_name='macroclass',
            class_name='class', calculate_signature=True
        )
        self.assertEqual(
            signature_catalog_3.table[
                signature_catalog_3.table['macroclass_id'] == 5].macroclass_id,
            5
        )
        region_vector = rs.shared_tools.region_growing_polygon(
            coordinate_x=230316, coordinate_y=4674708,
            input_bands=1, max_width=4,
            max_spectral_distance=0.1, minimum_size=1,
            bandset_catalog=catalog_3
        )
        signature_catalog_3.import_vector(
            file_path=region_vector, macroclass_value=7,
            class_value=21, macroclass_name='macroclass_2',
            class_name='class_2', calculate_signature=True
        )
        signature_count = signature_catalog_3.table.shape[0]
        ids = signature_catalog_3.table.signature_id.tolist()
        signature_catalog_3.merge_signatures_by_id(
            signature_id_list=ids, calculate_signature=False)
        self.assertEqual(
            signature_catalog_3.table.shape[0], signature_count + 1
        )
        signature = signature_catalog_3.table[
            signature_catalog_3.table['signature'] == 0].signature_id[0]
        ids.append(signature)
        signature_catalog_3.merge_signatures_by_id(
            signature_id_list=ids, calculate_signature=True,
            class_name='merged3'
        )
        self.assertEqual(
            signature_catalog_3.table[
                signature_catalog_3.table['class_name'] == 'merged3'
            ].signature, 1
        )

        # clear temporary directory
        rs.close()
