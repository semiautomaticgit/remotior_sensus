from pathlib import Path
from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import read_write_files
from remotior_sensus.core.processor_functions import spectral_signature


class TestMultiprocess(TestCase):

    def test_multiprocess(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10, mpi_module=True
        )
        cfg = rs.configurations
        data_path = Path(__file__).parent / 'data'
        cfg.logger.log.debug('>>> test query database Sentinel-2')
        rs.download_products.product_names()
        output_manager = rs.download_products.search(
            product=cfg.sentinel2_mpc, date_from='2025-01-01',
            date_to='2025-01-30', max_cloud_cover=100,
            result_number=5, name_filter='T33TTG', mpi_module=True
        )
        product_table = output_manager.extra['product_table']
        # download Sentinel-2 bands
        cfg.logger.log.debug('>>> test download Sentinel-2 bands')
        output_manager = rs.download_products.download(
            product_table=product_table[product_table['cloud_cover'] < 2],
            output_path=cfg.temp.dir + '/test_1', band_list=['01']
        )
        if cfg.mpi_rank == 0:
            self.assertTrue(
                rs.files_directories.is_file(output_manager.paths[0]))

        cfg.logger.log.debug('>>> test raster_report')
        p = str(data_path / 'S2_2020-01-01' / 'S2_B02.tif')
        report = rs.raster_report(p)
        table = read_write_files.open_text_file(report.path)
        table_f = read_write_files.format_csv_new_delimiter(
            table, cfg.tab_delimiter
        )
        if cfg.mpi_rank == 0:
            self.assertGreater(len(table_f), 0)

        cfg.logger.log.debug('>>> test spectral_signature')
        file_list = [
            str(data_path / 'L8_2020-01-01/L8_B2.tif'),
            str(data_path / 'L8_2020-01-01/L8_B3.tif'),
            str(data_path / 'L8_2020-01-01/L8_B4.tif'),
        ]
        roi_paths = [str(data_path / 'files' / 'roi.gpkg')] * len(file_list)
        cfg.multiprocess.run_separated(
            raster_path_list=file_list, function=spectral_signature,
            function_argument=roi_paths, function_variable=file_list,
            n_processes=2, keep_output_argument=True,
            progress_message='calculate signature'
        )
        cfg.multiprocess.multiprocess_spectral_signature()
        (value_list, standard_deviation_list,
         count_list) = cfg.multiprocess.output
        if cfg.mpi_rank == 0:
            self.assertGreater(len(value_list), 0)
        cfg.logger.log.debug('>>> test band_erosion')
        file_list = [
            str(data_path / 'S2_2020-01-01' / 'S2_B02.tif'),
            str(data_path / 'S2_2020-01-01' / 'S2_B03.tif')
        ]
        erosion = rs.band_erosion(
            input_bands=file_list, output_path=cfg.temp.dir,
            value_list=[1, 425], size=1, circular_structure=True,
            prefix='erosion_'
        )
        if cfg.mpi_rank == 0:
            self.assertTrue(rs.files_directories.is_file(erosion.paths[0]))

        cfg.logger.log.debug('>>> test warp ')
        t_pmd = cfg.temp.temporary_raster_path(extension=cfg.vrt_suffix)
        raster = cfg.multiprocess.create_warped_vrt(
            raster_path=str(data_path / 'S2_2020-01-01' / 'S2_B02.tif'),
            output_path=t_pmd)
        if cfg.mpi_rank == 0:
            self.assertTrue(rs.files_directories.is_file(raster))

        cfg.logger.log.debug('>>> test raster_to_vector')
        p = str(data_path / 'S2_2020-01-01' / 'S2_B02.tif')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.gpkg_suffix,
                                            rank=True)
        vector = rs.raster_to_vector(p, temp)
        if cfg.mpi_rank == 0:
            self.assertTrue(rs.files_directories.is_file(vector.path))

        cfg.logger.log.debug('>>> test band_sieve')
        file_list = [
            str(data_path / 'S2_2020-01-01' / 'S2_B02.tif'),
            str(data_path / 'S2_2020-01-01' / 'S2_B03.tif')
        ]
        sieve = rs.band_sieve(
            input_bands=file_list, output_path=cfg.temp.dir, size=2,
            connected=False, prefix='sieve_'
        )
        if cfg.mpi_rank == 0:
            self.assertTrue(rs.files_directories.is_file(sieve.paths[0]))
        cfg.logger.log.debug('>>> test vector to raster')
        v = str(data_path / 'files' / 'roi_2.gpkg')
        r = str(data_path / 'S2_2020-01-01' / 'S2_B02.tif')
        temp = cfg.temp.temporary_file_path(
            name='raster', name_suffix=cfg.tif_suffix
        )
        raster = rs.vector_to_raster(vector_path=v, align_raster=r,
                                     constant=1, output_path=temp)
        if cfg.mpi_rank == 0:
            self.assertTrue(rs.files_directories.is_file(raster.path))
        temp = cfg.temp.temporary_file_path(
            name='raster', name_suffix=cfg.tif_suffix
        )
        raster = rs.vector_to_raster(vector_path=v, align_raster=r,
                                     method='area_based', vector_field='class',
                                     output_path=temp)
        if cfg.mpi_rank == 0:
            self.assertTrue(rs.files_directories.is_file(raster.path))
        cfg.logger.log.debug('>>> test band clip input BandSet')
        catalog = rs.bandset_catalog()
        file_list = ['S2_2020-01-01/S2_B02.tif', 'S2_2020-01-01/S2_B03.tif',
                     'S2_2020-01-01/S2_B04.tif']
        data_path = Path(__file__).parent / 'data'
        catalog.create_bandset(
            file_list, wavelengths=['Sentinel-2'], bandset_number=1,
            root_directory=str(data_path)
        )
        v = str(data_path / 'files' / 'roi.gpkg')
        output = rs.band_clip(input_bands=catalog.get_bandset(1),
                              output_path=cfg.temp.dir, prefix='clip3_',
                              vector_path=v)
        if cfg.mpi_rank == 0:
            self.assertTrue(output.check)
            self.assertTrue(rs.files_directories.is_file(output.paths[0]))

        cfg.logger.log.debug('>>> test multi layer perceptron')
        temp = cfg.temp.temporary_file_path(
            name='class', name_suffix=cfg.tif_suffix
        )
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
            calculate_signature=True, rank=True
        )

        signature_catalog_1 = cfg.mpi_comm.bcast(signature_catalog_1, root=0)
        rs.band_classification(
            input_bands=catalog.get(1), output_path=temp,
            spectral_signatures=signature_catalog_1,
            algorithm_name=cfg.multi_layer_perceptron_a,
            classification_confidence=True,
            signature_raster=False, cross_validation=True, mlp_max_iter=5
        )
        if cfg.mpi_rank == 0:
            self.assertTrue(rs.files_directories.is_file(temp))

        # clear temporary directory
        rs.close()
