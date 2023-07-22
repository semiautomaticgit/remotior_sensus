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
            value_list=value_list, wavelength_list=wavelength_list
        )
        self.assertEqual(signature_catalog.table.shape[0], 2)
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
            file_path='data/files/roi.gpkg', macroclass_field='macroclass',
            class_field='class', macroclass_name_field='macroclass',
            class_name_field='class', calculate_signature=True
        )
        self.assertEqual(
            signature_catalog_2.table[
                signature_catalog_2.table['macroclass_id'] == 3].macroclass_id,
            3
        )

        # clear temporary directory
        rs.close()
