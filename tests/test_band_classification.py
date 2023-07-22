from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestBandClassification(TestCase):

    def test_band_classification(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        cfg.logger.log.debug('>>> test semiautomatic classification')
        # create BandSet
        catalog = rs.bandset_catalog()
        file_list = ['L8_2020-01-01/L8_B2.tif', 'L8_2020-01-01/L8_B3.tif',
                     'L8_2020-01-01/L8_B4.tif', 'L8_2020-01-01/L8_B5.tif',
                     'L8_2020-01-01/L8_B6.tif', 'L8_2020-01-01/L8_B7.tif']
        root_directory = 'data/'
        catalog.create_bandset(
            file_list, wavelengths=['Landsat 8'], root_directory=root_directory
            )
        # set BandSet in SpectralCatalog
        signature_catalog_2 = rs.spectral_signatures_catalog(
            bandset=catalog.get(1)
            )
        # import vector
        signature_catalog_2.import_vector(
            file_path='data/files/roi.gpkg',
            macroclass_field='macroclass', class_field='class',
            macroclass_name_field='macroclass', class_name_field='class',
            calculate_signature=True
        )
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_2,
            signature_raster=False
            )
        self.assertTrue(files_directories.is_file(temp))
        cfg.logger.log.debug(
            '>>> test semiautomatic classification with signature rasters'
            )
        # semiautomatic classification with signature rasters
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.vrt_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_2,
            signature_raster=True, n_processes=2
            )
        cfg.logger.log.debug('>>> test maximum likelihood')
        self.assertTrue(files_directories.is_file(temp))
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_2,
            algorithm_name=cfg.maximum_likelihood, signature_raster=False
            )
        #self.assertTrue(files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test minimum distance')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_2,
            algorithm_name=cfg.minimum_distance, signature_raster=False
            )
        self.assertTrue(files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test spectral angle mapping')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_2,
            algorithm_name=cfg.spectral_angle_mapping,
            signature_raster=True
            )
        self.assertTrue(files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test random forest')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_2,
            algorithm_name=cfg.random_forest, signature_raster=False,
            rf_max_features=5,
            rf_number_trees=20, rf_min_samples_split=2
            )
        self.assertTrue(files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test random forest ovr')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_2,
            algorithm_name=cfg.random_forest_ovr, signature_raster=False
            )
        self.assertTrue(files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test support vector machine')
        classifier = rs.band_classification(
            input_bands=catalog.get(1),
            spectral_signatures=signature_catalog_2,
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
            spectral_signatures=signature_catalog_2,
            algorithm_name=cfg.support_vector_machine, macroclass=True,
            signature_raster=False
            )
        self.assertTrue(files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test multi layer perceptron')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_2,
            algorithm_name=cfg.multi_layer_perceptron,
            classification_confidence=True,
            signature_raster=False, cross_validation=True, mlp_max_iter=5
            )
        self.assertTrue(files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test multi layer perceptron')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_2,
            algorithm_name=cfg.multi_layer_perceptron, signature_raster=False,
            find_best_estimator=True,
            pytorch_loss_function=None, mlp_hidden_layer_sizes=[100, 50],
            mlp_alpha=0.0001,
            mlp_learning_rate_init=0.001, mlp_max_iter=5, mlp_batch_size=10,
            mlp_activation='relu'
            )
        self.assertTrue(files_directories.is_file(temp))
        cfg.logger.log.debug('>>> test pytorch multi layer perceptron')
        temp = cfg.temp.temporary_file_path(name='class')
        classification = rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_2,
            algorithm_name=cfg.pytorch_multi_layer_perceptron, only_fit=True,
            save_classifier=True, class_weight='balanced',
            mlp_hidden_layer_sizes=[100, 50], mlp_max_iter=5,
            pytorch_device='cpu'
            )
        self.assertTrue(
            files_directories.is_file(classification.extra['model_path'])
            )
        cfg.logger.log.debug('>>> test load classification')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
            )
        classification2 = rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            load_classifier=classification.extra['model_path']
            )
        self.assertTrue(files_directories.is_file(classification2.path))

        # clear temporary directory
        rs.close()
