from unittest import TestCase

import remotior_sensus

from remotior_sensus.util import files_directories


class TestFilesDirectories(TestCase):

    def test_files_directories(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        path = '/home/user/file.tif'
        root = '/home/user/'
        relative_path = files_directories.absolute_to_relative_path(
            path=path,
            root=root
            )
        print(relative_path)
        absolute_path = files_directories.relative_to_absolute_path(
            path=relative_path, root=root
        )
        print(absolute_path)
        self.assertEqual(absolute_path, path)

        # clear temporary directory
        rs.close()
