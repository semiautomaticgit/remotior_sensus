import time
from unittest import TestCase

import remotior_sensus


class TestDownloadProducts(TestCase):

    def test_download_products(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        cfg.logger.log.debug('>>> test query database Sentinel-2')
        rs.download_products.product_names()
        coordinate_list = [8, 43, 10, 41]
        output_manager = rs.download_products.query_sentinel_2_database(
            date_from='2024-01-01', date_to='2024-01-30', max_cloud_cover=80,
            result_number=5, coordinate_list=coordinate_list, name_filter='L2A'
        )
        product_table = output_manager.extra['product_table']
        temp = cfg.temp.temporary_file_path(name_suffix='.xml')
        rs.download_products.export_product_table_as_xml(
            product_table=product_table, output_path=temp
        )
        self.assertTrue(rs.files_directories.is_file(temp))
        imported_table = rs.download_products.import_as_xml(xml_path=temp)
        product_table_i = imported_table.extra['product_table']
        self.assertTrue(product_table_i[0] == product_table[0])
        self.assertEqual(product_table['product'][0], cfg.sentinel2)
        cfg.logger.log.debug('>>> test search')
        output_manager = rs.download_products.search(
            product=cfg.sentinel2, date_from='2024-01-01',
            date_to='2024-01-30', max_cloud_cover=80,
            result_number=5, coordinate_list=coordinate_list, name_filter='L2A'
        )
        product_table = output_manager.extra['product_table']
        self.assertEqual(product_table['product'][0], cfg.sentinel2)
        output_manager = rs.download_products.query_sentinel_2_database(
            date_from='2024-01-01', date_to='2024-01-10', max_cloud_cover=80,
            result_number=5, name_filter='33SVB'
        )
        product_table = output_manager.extra['product_table']
        self.assertEqual(product_table['product'][0], cfg.sentinel2)
        # export download links Sentinel-2
        cfg.logger.log.debug('>>> test export download links Sentinel-2')
        output_manager = rs.download_products.download(
            product_table=product_table, output_path=cfg.temp.dir,
            exporter=True
        )
        self.assertTrue(rs.files_directories.is_file(output_manager.path))
        time.sleep(1)
        # download Sentinel-2 bands
        cfg.logger.log.debug('>>> test download Sentinel-2 bands')
        output_manager = rs.download_products.download(
            product_table=product_table[product_table['cloud_cover'] < 10],
            output_path=cfg.temp.dir + '/test_1', band_list=['01']
        )
        self.assertTrue(rs.files_directories.is_file(output_manager.paths[0]))
        time.sleep(1)
        # download Sentinel-2 virtual bands
        cfg.logger.log.debug('>>> test download sentinel-2 virtual bands')
        output_manager = rs.download_products.download(
            product_table=product_table[product_table['cloud_cover'] < 10],
            output_path=cfg.temp.dir + '/test_2', band_list=['01'],
            virtual_download=True
        )
        self.assertTrue(rs.files_directories.is_file(output_manager.paths[0]))
        time.sleep(1)
        # download Sentinel-2 virtual bands with subset
        cfg.logger.log.debug(
            '>>> test download sentinel-2 virtual bands with subset'
        )
        output_manager = rs.download_products.download(
            product_table=product_table[product_table['cloud_cover'] < 10],
            output_path=cfg.temp.dir + '/test_3', band_list=['01'],
            virtual_download=True,
            extent_coordinate_list=[494000, 4175000, 501000, 4169000]
        )
        self.assertTrue(rs.files_directories.is_file(output_manager.paths[0]))
        cfg.logger.log.debug('>>> test query Sentinel HLS')
        output_manager = rs.download_products.query_nasa_cmr(
            product=cfg.sentinel2_hls, date_from='2021-01-01',
            date_to='2021-01-30', max_cloud_cover=80,
            result_number=20, coordinate_list=[11, 43, 12, 42]
        )
        product_table_2 = output_manager.extra['product_table']
        self.assertEqual(product_table_2['product'][0], cfg.sentinel2_hls)
        # export download links HLS
        cfg.logger.log.debug('>>> test export download links HLS')
        output_manager = rs.download_products.download(
            product_table=product_table_2, output_path=cfg.temp.dir,
            exporter=True
        )
        self.assertTrue(rs.files_directories.is_file(output_manager.path))
        cfg.logger.log.debug('>>> test query Sentinel-2 MPC')
        output_manager = rs.download_products.search(
            product=cfg.sentinel2_mpc, date_from='2020-01-01',
            date_to='2020-01-30', max_cloud_cover=100,
            result_number=5, name_filter='T33TTG'
        )
        product_table_3 = output_manager.extra['product_table']
        output_manager = rs.download_products.download(
            product_table=product_table_3[product_table_3['cloud_cover'] < 28],
            output_path=cfg.temp.dir, exporter=True
        )
        self.assertTrue(rs.files_directories.is_file(output_manager.path))
        self.assertEqual(product_table_3['product'][0], cfg.sentinel2_mpc)
        output_manager = rs.download_products.search(
            product=cfg.landsat_mpc, date_from='1991-01-27',
            date_to='1991-01-27', max_cloud_cover=9,
            result_number=5, coordinate_list=[8, 43, 10, 41]
        )
        product_table_4 = output_manager.extra['product_table']
        self.assertEqual(product_table_4['product'][0], cfg.landsat_mpc)
        output_manager = rs.download_products.search(
            product=cfg.modis_09q1_mpc, date_from='2023-01-09',
            date_to='2023-01-09', max_cloud_cover=100,
            result_number=5, coordinate_list=[8, 43, 10, 41]
        )
        product_table_5 = output_manager.extra['product_table']
        self.assertEqual(product_table_5['product'][0], cfg.modis_09q1_mpc)
        output_manager = rs.download_products.search(
            product=cfg.modis_11a2_mpc, date_from='2023-01-01',
            date_to='2023-01-30',
            result_number=5, coordinate_list=[8, 43, 10, 41]
        )
        product_table_6 = output_manager.extra['product_table']
        self.assertEqual(product_table_6['product'][0], cfg.modis_11a2_mpc)
        output_manager = rs.download_products.download(
            product_table_6[product_table_6['product_id'] == product_table_6[
                'product_id'][0]],
            output_path=cfg.temp.dir + '/test_1', band_list=['01']
        )
        self.assertTrue(rs.files_directories.is_file(output_manager.paths[0]))
        output_manager = rs.download_products.search(
            product=cfg.aster_l1t_mpc, date_from='2003-01-01',
            date_to='2003-01-30',
            result_number=5, coordinate_list=[8, 43, 10, 41]
        )
        product_table_7 = output_manager.extra['product_table']
        self.assertEqual(product_table_7['product'][0], cfg.aster_l1t_mpc)
        output_manager = rs.download_products.search(
            product=cfg.cop_dem_glo_30_mpc, date_from='2003-01-01',
            date_to='2003-01-30',
            result_number=5, coordinate_list=[8, 43, 10, 41]
        )
        product_table_8 = output_manager.extra['product_table']
        self.assertEqual(product_table_8['product'][0], cfg.cop_dem_glo_30_mpc)

        """# user and password required
        # download HLS bands
        cfg.logger.log.debug('>>> test download HLS bands')
        output_manager = rs.download_products.download(
            product_table=product_table_2[product_table_2['cloud_cover'] < 10],
            output_path=cfg.temp.dir + '/test_4', band_list=['01'],
            nasa_user='', nasa_password='')
        self.assertTrue(rs.files_directories.is_file(output_manager.paths[0]))
        """

        """# user and password required
        # download Copernicus service bands
        cfg.logger.log.debug('>>> test download Copernicus service bands')
        output_manager = rs.download_products.search(
            product='Sentinel-2', date_from='2023-07-01', date_to='2023-07-30',
            max_cloud_cover=80, result_number=1, name_filter='32TNN',
            copernicus_user='', copernicus_password=''
        )
        product_table_3 = output_manager.extra['product_table']
        output_manager = rs.download_products.download(
            product_table=product_table_3,
            output_path=cfg.temp.dir + '/test_5', band_list=['01'],
            copernicus_user='', copernicus_password='')
        self.assertTrue(rs.files_directories.is_file(output_manager.paths[0]))
        """

        # clear temporary directory
        rs.close()
