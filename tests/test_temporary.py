from unittest import TestCase

import remotior_sensus


class TestTemporary(TestCase):

    def test_create_root_temporary_directory(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('>>> test temp directory')
        self.assertTrue(rs.files_directories.is_directory(cfg.temp.dir))

        # clear temporary directory
        rs.close()
