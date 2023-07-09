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

import unittest

# protect the entry point for multiprocess
if __name__ == '__main__':
    testLoader = unittest.TestLoader()
    test_dir = '.'
    # units to test
    pattern = 'test*.py'
    # pattern = 'test_progress.py'
    d = testLoader.discover(test_dir, pattern=pattern)
    textTestRunner = unittest.TextTestRunner(verbosity=2)
    textTestRunner.run(d)
