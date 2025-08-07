from unittest import TestCase

import remotior_sensus


class TestLog(TestCase):

    def test_create(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        cfg.logger.log.debug('>>> test logger file path')
        self.assertTrue(rs.files_directories.is_file(cfg.logger.file_path))

        # clear temporary directory
        rs.close()
