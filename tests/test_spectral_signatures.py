from unittest import TestCase

import remotior_sensus


class TestSpectralSignatures(TestCase):

    def test_spectral_signatures(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
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
            value_list=value_list, macroclass_id=2,
            wavelength_list=wavelength_list
            )
        self.assertEqual(signature_catalog.table.shape[0], 2)
        cfg.logger.log.debug('>>> test save Spectral Signature Catalog')
        temp = cfg.temp.temporary_file_path(name_suffix='.scpx')
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
        temp = cfg.temp.temporary_file_path(name_suffix='.scpx')
        signature_catalog_2.save(output_path=temp)
        self.assertTrue(rs.files_directories.is_file(temp))
        sig_id = signature_catalog_2.table[
            signature_catalog_2.table['macroclass_id'] == 7].signature_id[0]
        signature_catalog_2.remove_signature_by_id(signature_id=sig_id)
        temp = cfg.temp.temporary_file_path(name_suffix='.scpx')
        signature_catalog_2.save(output_path=temp)
        self.assertTrue(rs.files_directories.is_file(temp))
        shape = signature_catalog_2.table.shape[0]
        signature_catalog_2.import_file(file_path=temp)
        self.assertEqual(
            signature_catalog_2.table.shape[0],
            len(signature_catalog_2.signatures)
        )
        self.assertEqual(
            signature_catalog_2.table.shape[0], shape * 2
        )
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
        signature_7_pixel_count = signature_catalog_3.table[
            signature_catalog_3.table['macroclass_id'] == 7].pixel_count[0]
        signature_7_unit = signature_catalog_3.table[
            signature_catalog_3.table['macroclass_id'] == 7].unit[0]
        self.assertTrue(signature_7_pixel_count > 0)
        self.assertEqual(signature_7_unit, cfg.wl_micro)
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
        # signature distance
        cfg.logger.log.debug('>>> test signature distance')
        # calculate Bray-Curtis similarity
        bray_curtis = signature_catalog_3.calculate_bray_curtis_similarity(
            signature_id_x=ids[0], signature_id_y=ids[1]
        )
        self.assertTrue(bray_curtis > 0)
        # calculate Euclidean Distance
        euclidean_distance = signature_catalog_3.calculate_euclidean_distance(
            signature_id_x=ids[0], signature_id_y=ids[1]
        )
        self.assertTrue(euclidean_distance > 0)
        # calculate spectral angle
        spectral_angle = signature_catalog_3.calculate_spectral_angle(
            signature_id_x=ids[0], signature_id_y=ids[1]
        )
        self.assertTrue(spectral_angle > 0)
        # signature to plot
        cfg.logger.log.debug('>>> test signature to plot')
        plot_catalog = rs.spectral_signatures_plot_catalog()
        signature_id = signature_catalog_3.table.signature_id[0]
        signature_catalog_3.export_signature_values_for_plot(
            signature_id=signature_id, plot_catalog=plot_catalog
            )
        self.assertEqual(plot_catalog.get_signature_count(), 1)
        for plot in plot_catalog.catalog:
            signature = plot_catalog.get_signature(signature_id=plot)

            self.assertEqual(signature.signature_id, signature_id)
        signature_ids = plot_catalog.get_signature_ids()
        self.assertEqual(len(signature_ids), 1)
        """
        # interactive plot
        signature_catalog_3.add_signatures_to_plot_by_id(signature_ids)
        """
        plot_catalog.remove_signature(signature_id)
        self.assertEqual(plot_catalog.get_signature_count(), 0)
        """
        # interactive plot
        histogram = signature_catalog_3.calculate_scatter_plot_by_id(
            signature_id=signature_id, band_x=1, band_y=2, decimal_round=1,
            plot=True
        )
        """
        histogram = signature_catalog_3.calculate_scatter_plot_by_id(
            signature_id=signature_id, band_x=1, band_y=2, decimal_round=1
        )
        self.assertTrue(histogram is not None)

        # export vector
        temp_gpkg = cfg.temp.temporary_file_path(name_suffix='.gpkg')
        signature_catalog_3.export_vector(
            signature_id_list=[signature_id], output_path=temp_gpkg,
            vector_format='GPKG'
        )
        self.assertTrue(rs.files_directories.is_file(temp_gpkg))
        temp_shp = cfg.temp.temporary_file_path(name_suffix='.shp')
        signature_catalog_3.export_vector(
            signature_id_list=[signature_id], output_path=temp_shp,
            vector_format='ESRI Shapefile'
        )
        self.assertTrue(rs.files_directories.is_file(temp_shp))
        # export as csv
        csv = signature_catalog_3.export_signatures_as_csv(
            signature_id_list=[signature_id], output_directory=cfg.temp.dir
        )
        self.assertTrue(rs.files_directories.is_file(csv[0]))

        # clear temporary directory
        rs.close()
