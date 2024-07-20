from unittest import TestCase

import remotior_sensus


class TestFilesDirectories(TestCase):

    def test_files_directories(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('>>> test files directories')
        path = '/home/user/file.tif'
        root = '/home/user/'
        relative_path = rs.files_directories.absolute_to_relative_path(
            path=path, root=root
            )
        absolute_path = rs.files_directories.relative_to_absolute_path(
            path=relative_path, root=root
        )
        self.assertEqual(absolute_path, path)

        # clear temporary directory
        rs.close()
