# Remotior Sensus , software to process remote sensing and GIS data.
# Copyright (C) 2022-2023 Luca Congedo.
# Author: Luca Congedo
# Email: ing.congedoluca@gmail.com
#
# This file is part of Remotior Sensus.
# Remotior Sensus is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# Remotior Sensus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Remotior Sensus. If not, see <https://www.gnu.org/licenses/>.

import random
import tempfile

from remotior_sensus.core import configurations as cfg
from remotior_sensus.util import files_directories, dates_times, shared_tools


class Temporary(object):

    def __init__(self, temp_dir=None):
        self.dir = temp_dir

    # create root temporary directory
    @classmethod
    def create_root_temporary_directory(cls, prefix=None, directory=None):
        times = dates_times.get_date_string()
        t_dir = tempfile.mkdtemp(
            prefix='{}_{}'.format(prefix, times), dir=directory
        )
        return cls(t_dir)

    # clear root temporary directory
    def clear(self):
        remove_root_temporary_directory(self.dir)
        self.dir = None
        return self.dir

    # create temporary directory
    def create_temporary_directory(self):
        times = dates_times.get_time_string()
        directory = shared_tools.join_path(
            self.dir, '{}{}'.format('t', times)
        ).replace('\\', '/')
        files_directories.create_directory(directory)
        return directory

    # create temporary file path
    def temporary_file_path(self, name_suffix=None, name_prefix=None,
                            name=None, directory=None):
        times = dates_times.get_time_string()
        if name is None:
            r = str(random.randint(0, 10000))
            name = 't{}_{}'.format(times, r)
        else:
            directory = shared_tools.join_path(
                self.dir, '{}{}'.format('t', times)
            ).replace('\\', '/')
            files_directories.create_directory(directory)
        if name_suffix is not None:
            name = '%s%s' % (name, name_suffix)
        if name_prefix is not None:
            name = '%s%s' % (name_prefix, name)
        if directory is None:
            directory = self.dir
        path = shared_tools.join_path(directory, name).replace('\\', '/')
        return path

    # create temporary raster file path
    def temporary_raster_path(
            self, name=None, name_suffix=None, name_prefix=None,
            extension='.tif'
    ):
        file_path = self.temporary_file_path(
            name_suffix=name_suffix, name_prefix=name_prefix, name=name
        )
        path = '%s%s' % (file_path, extension)
        return path


# remove root temporary directory
def remove_root_temporary_directory(directory):
    # close log handlers
    try:
        for h in cfg.logger.log.handlers:
            h.close()
    except Exception as err:
        str(err)
    try:
        cfg.logger.log.handlers = []
    except Exception as err:
        str(err)
    files_directories.remove_directory(directory)
    return directory
