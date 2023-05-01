from unittest import TestCase

import remotior_sensus


class TestOutputManager(TestCase):

    def test_output_manager(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        path = './data/S2_2020-01-01/S2_B02.tif'
        cfg.logger.log.debug('>>> test output manager')
        output = rs.output_manager(path=path)
        self.assertEqual(output.path, path)
        # create BandSet Catalog
        catalog = rs.bandset_catalog()
        cfg.logger.log.debug('>>> test add to bandset')
        # add to BandSet 1
        output.add_to_bandset(
            bandset_catalog=catalog, bandset_number=1, band_number=1
        )
        self.assertEqual(catalog.get_bandset(1).get_band(1).path, path)

        # clear temporary directory
        rs.close()
