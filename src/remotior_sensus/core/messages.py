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

"""Manager of messages.

Console messages during processes.

Typical usage example:

    >>> # display a warning message
    >>> warning('warning message')
"""


def warning(message: str):
    """Warning message.

        Prints a warning message.

        Args:
            message: message.

        Examples:
            Display a message
                >>> warning('warning message')
        """
    print('⚠ warning: %s' % message)


def error(message: str):
    """Error message.

        Prints an error message.

        Args:
            message: message.

        Examples:
            Display a message
                >>> error('error message')
        """
    print('▲ error: %s' % message)


def info(message: str):
    """Info message.

        Prints an info message.

        Args:
            message: message.

        Examples:
            Display a message
                >>> info('error message')
        """
    print('info: %s' % message)
