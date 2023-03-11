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

"""
Tools to manage operating systems
"""

import platform
import sys

from remotior_sensus.core import configurations as cfg


# get system information
def get_system_info():
    if sys.maxsize > 2 ** 32:
        cfg.sys_64bit = True
    else:
        cfg.sys_64bit = False
    # file system encoding
    cfg.file_sys_encoding = sys.getfilesystemencoding()
    # system information
    cfg.sys_name = platform.system()
    cfg.logger.log.info(
        'system: %s; 64bit: %s; n_processes: %s; ram: %s; temp.dir: %s'
        % (cfg.sys_name, cfg.sys_64bit, cfg.n_processes, cfg.available_ram,
           cfg.temp.dir)
        )
