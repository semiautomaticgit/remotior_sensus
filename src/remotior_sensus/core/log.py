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
"""Logging manger.

Core class that manages logs during processes.

Typical usage example:

    >>> # create a log file in a directory
    >>> Log(directory='directory_path', level=10)
"""

import io
import logging
from typing import Union, Optional

from remotior_sensus.core import configurations as cfg


class Log(object):
    log = None

    def __init__(
            self, file_path: Optional[str] = None,
            directory: Optional[str] = None, level: Union[int, str] = None,
            multiprocess=False, time=True
    ):
        """Manages logs.

        This module allows for managing logs of processes.

        Attributes:
            file_path: path of a log file.
            directory: directory path where a log file is created if file_path is None.
            level: level of logging (10 for DEBUG, 20 for INFO).
            multiprocess: if True, sets logging for parallel processes.
            time: if True, time is saved in log file.

        Examples:
            Create a log file and starts logging.
                >>> Log(file_path='file.txt', level=20)
        """  # noqa: E501
        if file_path is None:
            if directory is None:
                raise Exception('file path or directory missing')
            else:
                file_path = '{}/{}.log'.format(directory, cfg.root_name)
        # create logger
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG)
        if level is None:
            level = logging.DEBUG
        if not multiprocess:
            # create file handler
            fh = logging.FileHandler(file_path)
            fh.setLevel(level)
            fhf = logging.Formatter(
                '%(levelname)s|%(asctime)s.%(msecs)03d|%(module)s|%(funcName)s'
                '|%(lineno)s|%(message)s', '%Y-%m-%dT%H:%M:%S'
            )
            fh.setFormatter(fhf)
            logger.addHandler(fh)
            # create console handler
            ch = logging.StreamHandler()
            ch.setLevel(level)
            if time:
                chf = logging.Formatter(
                    '%(levelname)s[%(asctime)s.%(msecs)03d] '
                    '%(module)s.%(funcName)s[%(lineno)s] %(message)s',
                    '%Y-%m-%dT%H:%M:%S'
                )
            else:
                chf = logging.Formatter(
                    '%(levelname)s %(module)s.%(funcName)s[%(lineno)s] '
                    '%(message)s'
                )
            ch.setFormatter(chf)
            logger.addHandler(ch)
            logger.propagate = False
            self.log = logger
            self.file_path = file_path
            self.stream = None
            self.level = level
        # multiprocess number
        else:
            # create stream handler
            stream = io.StringIO()
            ch = logging.StreamHandler(stream)
            ch.setLevel(level)
            if time:
                chf = logging.Formatter(
                    '%(levelname)s_p{}|%(asctime)s.%(msecs)03d|%(module)s'
                    '|%(funcName)s|%(lineno)s|%(message)s'.format(
                        multiprocess
                    ), '%Y-%m-%dT%H:%M:%S'
                )
            else:
                chf = logging.Formatter(
                    '%(levelname)s_p{}|%(module)s|%(funcName)s|%(lineno)s'
                    '|%(message)s'.format(multiprocess)
                )
            ch.setFormatter(chf)
            logger.addHandler(ch)
            logger.propagate = False
            self.log = logger
            self.stream = stream
            self.file_path = None
            self.level = level
