from pathlib import Path
from unittest import TestCase

import remotior_sensus
# noinspection PyProtectedMember
from remotior_sensus.tools.band_classification import _get_x_y_arrays_from_rois


class TestBandClassification(TestCase):

    def test_band_classification(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('>>> test semiautomatic classification')
        data_path = Path(__file__).parent / 'data'
        # create BandSet
        catalog = rs.bandset_catalog()
        file_list = ['L8_2020-01-01/L8_B2.tif', 'L8_2020-01-01/L8_B3.tif',
                     'L8_2020-01-01/L8_B4.tif', 'L8_2020-01-01/L8_B5.tif',
                     'L8_2020-01-01/L8_B6.tif', 'L8_2020-01-01/L8_B7.tif']
        catalog.create_bandset(
            file_list, wavelengths=['Landsat 8'], root_directory=str(data_path)
            )
        # set BandSet in SpectralCatalog
        signature_catalog_1 = rs.spectral_signatures_catalog(
            bandset=catalog.get(1)
        )
        # import vector
        signature_catalog_1.import_vector(
            file_path=str(data_path / 'files' / 'roi.gpkg'),
            macroclass_field='macroclass', class_field='class',
            macroclass_name_field='macroclass', class_name_field='class',
            calculate_signature=True
        )
        input_raster_list = [
            str(data_path / 'L8_2020-01-01/L8_B2.tif'),
            str(data_path / 'L8_2020-01-01/L8_B3.tif'),
            str(data_path / 'L8_2020-01-01/L8_B4.tif'),
        ]
        x_y_arrays = _get_x_y_arrays_from_rois(
            raster_paths=input_raster_list,
            roi_path=signature_catalog_1.geometry_file,
            spectral_signatures=signature_catalog_1,
            same_geotransformation=True
        )
        self.assertTrue(x_y_arrays.extra['x'].shape[0] > 0)
        cfg.logger.log.debug('>>> test input multiband')
        catalog.create_bandset(
            [str(data_path / 'S2_2020-01-05' / 'S2_2020-01-05.tif')],
            bandset_number=2
        )
        # set BandSet in SpectralCatalog
        signature_catalog_2 = rs.spectral_signatures_catalog(
            bandset=catalog.get(2)
            )
        # import vector
        signature_catalog_2.import_vector(
            file_path=str(data_path / 'files' / 'roi.gpkg'),
            macroclass_field='macroclass', class_field='class',
            macroclass_name_field='macroclass', class_name_field='class',
            calculate_signature=True
        )
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_1, signature_raster=False
            )
        temp_sig = cfg.temp.temporary_file_path(name_suffix=cfg.scpx_suffix)
        signature_catalog_1.save(temp_sig)
        temp_class_sig = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp_class_sig,
            spectral_signatures=temp_sig
            )
        self.assertTrue(rs.files_directories.is_file(temp))
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.vrt_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(2), output_path=temp,
            spectral_signatures=signature_catalog_2,
            algorithm_name=cfg.spectral_angle_mapping_a,
            signature_raster=False
            )
        self.assertTrue(rs.files_directories.is_file(temp))
        cfg.logger.log.debug(
            '>>> test semiautomatic classification with signature rasters'
            )
        # semiautomatic classification with signature rasters
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.vrt_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_1,
            signature_raster=True, n_processes=2
            )
        cfg.logger.log.debug('>>> test maximum likelihood')
        self.assertTrue(rs.files_directories.is_file(temp))
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_1,
            algorithm_name=cfg.maximum_likelihood_a, signature_raster=False
            )
        self.assertTrue(rs.files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test minimum distance')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_1,
            algorithm_name=cfg.minimum_distance_a, signature_raster=False
            )
        self.assertTrue(rs.files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test spectral angle mapping')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_1,
            algorithm_name=cfg.spectral_angle_mapping_a,
            signature_raster=True
            )
        self.assertTrue(rs.files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test random forest')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_1,
            algorithm_name=cfg.random_forest_a, signature_raster=False,
            rf_max_features=5, rf_number_trees=20, rf_min_samples_split=2
            )
        self.assertTrue(rs.files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test random forest ovr')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_1,
            algorithm_name=cfg.random_forest_ovr_a, signature_raster=False
            )
        self.assertTrue(rs.files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test support vector machine')
        classifier = rs.band_classification(
            input_bands=catalog.get(1),
            spectral_signatures=signature_catalog_1,
            algorithm_name=cfg.support_vector_machine, only_fit=True, svm_c=1,
            svm_gamma='scale', svm_kernel='rbf'
            )
        self.assertTrue(classifier.extra['classifier'] is not None)
        self.assertTrue(classifier.check)
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_1,
            algorithm_name=cfg.support_vector_machine_a, macroclass=True,
            signature_raster=False
            )
        self.assertTrue(rs.files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test multi layer perceptron')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_1,
            algorithm_name=cfg.multi_layer_perceptron_a,
            classification_confidence=True,
            signature_raster=False, cross_validation=True, mlp_max_iter=5
            )
        self.assertTrue(rs.files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test multi layer perceptron')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_1,
            algorithm_name=cfg.multi_layer_perceptron, signature_raster=False,
            find_best_estimator=True, pytorch_loss_function=None,
            mlp_hidden_layer_sizes=[100, 50], mlp_alpha=0.0001,
            mlp_learning_rate_init=0.001, mlp_max_iter=5, mlp_batch_size=10,
            mlp_activation='relu'
            )
        self.assertTrue(rs.files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test pytorch multi layer perceptron')
        temp = cfg.temp.temporary_file_path(name='class')
        classification = rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_1,
            algorithm_name=cfg.pytorch_multi_layer_perceptron_a, only_fit=True,
            save_classifier=True, class_weight='balanced',
            mlp_hidden_layer_sizes=[100, 50], mlp_max_iter=5,
            pytorch_device='cpu'
            )
        self.assertTrue(
            rs.files_directories.is_file(classification.extra['model_path'])
            )
        cfg.logger.log.debug('>>> test load classification')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        classification2 = rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            load_classifier=classification.extra['model_path']
            )
        self.assertTrue(rs.files_directories.is_file(classification2.path))

        # clear temporary directory
        rs.close()
