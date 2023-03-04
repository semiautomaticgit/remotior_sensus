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
Tools to manage dates and times
"""

import datetime
from remotior_sensus.core import configurations as cfg


# get date string from name
def date_string_from_directory_name(directory_name):
    date = False
    # format YYYY-MM-DD
    try:
        datetime.datetime.strptime(directory_name[-10:], '%Y-%m-%d')
        date = directory_name[-10:]
    except Exception as err:
        str(err)
        # format YYYYMMDD
        try:
            dir_part = directory_name.split('_')
            for d_p in dir_part:
                d_p_part = d_p.lower().split('t')[0]
                try:
                    date_string = datetime.datetime.strptime(d_p_part,
                                                             '%Y%m%d')
                    d_p_part_string = date_string.strftime('%Y-%m-%d')
                    date = d_p_part_string
                    break
                except Exception as err:
                    str(err)
        except Exception as err:
            str(err)
    cfg.logger.log.debug('date: %s' % date)
    return date


# get time
def get_time_string():
    time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')
    return time


# get date
def get_date_string():
    time = datetime.datetime.now().strftime('%m%d')
    return time


# create date
def create_date(string: str):
    # format YYYY-MM-DD
    try:
        date = datetime.datetime.strptime(string, '%Y-%m-%d')
    except Exception as err:
        cfg.logger.log.error(str(err))
        date = None
    return date
