from unittest import TestCase

import remotior_sensus


class TestPreprocessProducts(TestCase):

    def test_preprocess_products(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        # test Sentinel-2
        cfg.logger.log.debug('>>> test sentinel-2')
        table = rs.preprocess_products.create_product_table(
            input_path='data/S2_2020-01-01',
            metadata_file_path='data/files/sentinel_2_metadata_test_l1c.xml'
        )
        out_1 = rs.preprocess_products.perform_preprocess(
            product_table=table, output_path=cfg.temp.dir + '/test_1',
            dos1_correction=False
            )
        self.assertTrue(out_1.check)
        # test DOS1
        cfg.logger.log.debug('>>> test DOS1')
        out_2 = rs.preprocess_products.perform_preprocess(
            product_table=table, output_path=cfg.temp.dir + '/test_2',
            dos1_correction=True
            )
        self.assertTrue(out_2.check)
        # test Landsat
        cfg.logger.log.debug('>>> test Landsat')
        table2 = rs.preprocess_products.create_product_table(
            input_path='data/L8_2020-01-01',
            metadata_file_path='data/files/landsat_8_metadata_mtl.xml'
        )
        out_3 = rs.preprocess_products.perform_preprocess(
            product_table=table2, output_path=cfg.temp.dir + '/test_3'
            )
        self.assertTrue(out_3.check)
        table3 = rs.preprocess_products.create_product_table(
            input_path='data/L8_2020-01-01',
            metadata_file_path='data/files/landsat_5_metadata_mtl.xml'
        )
        out_4 = rs.preprocess_products.perform_preprocess(
            product_table=table3, output_path=cfg.temp.dir + '/test_4',
            dos1_correction=True
            )
        self.assertTrue(out_4.check)
        out_5 = rs.preprocess_products.preprocess(
            input_path='data/L8_2020-01-01',
            output_path=cfg.temp.dir + '/test_5',
            metadata_file_path='data/files/landsat_5_metadata_mtl.xml',
            dos1_correction=True
            )
        self.assertTrue(out_5.check)

        # clear temporary directory
        rs.close()
