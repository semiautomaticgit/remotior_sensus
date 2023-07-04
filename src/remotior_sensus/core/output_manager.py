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
"""Output manager.

Core class that manages several types of output, mainly intended for tools
that have several outputs.

Typical usage example:

    >>> # process output is checked
    >>> OutputManager()
"""

from remotior_sensus.core.bandset_catalog import BandSetCatalog


class OutputManager(object):
    """Manages output.

    This class manages several types of output, mainly intended for tools
    that have several outputs.
    Check argument is False if output failed.
    Single output raster or multiple file paths can be defined as arguments.
    Additional output files or tables are managed with an extra argument.
    The type of the extra argument can be flexible depending on the process 
    output.

    Attributes:
        check: True if output is as expected, False if process failed.
        path: path of the first output.
        paths: list of output paths in case of multiple outputs.
        extra: additional output elements depending on the process.

    Examples:
        Output failed
            >>> OutputManager(check=False)
            
        Output is checked and file path is provided
            >>> OutputManager(path='file.tif')
    """  # noqa: E501

    def __init__(
            self, check: bool = True, path: str = None, paths: list = None,
            extra=None
    ):
        """Initializes an Output.

        Initializes an Output. 

        Args:   
            check: True if output is as expected, False if process failed.
            path: path of the first output.
            paths: list of output paths in case of multiple outputs.
            extra: additional output elements depending on the process.

        Examples:
            Create an object with a single file path
                >>> OutputManager(path='file.tif')
                
            Create an object with several output file paths in a list and an extra argument for a dictionary
                >>> OutputManager(
                ... paths=['file1.tif', 'file2.tif'],
                ... extra={'additional_output': 'file.csv'}
                ... )
        )
        """  # noqa: E501
        self.check = check
        self.paths = paths
        if path is None:
            if paths is None:
                self.path = None
            elif len(paths) == 0:
                self.path = None
            else:
                self.path = paths[0]
        else:
            self.path = path
        self.extra = extra

    def add_to_bandset(
            self, bandset_catalog: BandSetCatalog, bandset_number=None,
            band_number=None, raster_band=None, band_name=None, date=None,
            unit=None, root_directory=None, multiplicative_factor=None,
            additive_factor=None, wavelength=None
            ):
        """Adds output to BandSet.

        Adds the OutputManager.path as a band to a BandSet in a BandSetCatalog.

        Args:
            bandset_catalog: BandSetCatalog object.
            band_name: raster name used for identifying the bands.
            wavelength: center wavelengths of band.
            unit: wavelength unit as string
            multiplicative_factor: multiplicative factor for bands during calculations.
            additive_factor: additive factors for band during calculations.
            date: date string (format YYYY-MM-DD).
            bandset_number: number of the BandSet; if None, the band is added to the current BandSet.
            root_directory: root directory for relative path.
            raster_band: raster band number.
            band_number: number of band in BandSet.

        Examples:
            Add the output to BandSet 1 as band 1.
                >>> catalog = BandSetCatalog()
                >>> OutputManager.add_to_bandset(
                ... bandset_catalog=catalog, bandset_number=1, band_number=1
                ... )
        """  # noqa: E501
        if type(bandset_catalog) is BandSetCatalog:
            bandset_catalog.add_band_to_bandset(
                path=self.path, bandset_number=bandset_number,
                band_number=band_number, raster_band=raster_band,
                band_name=band_name, date=date, unit=unit,
                root_directory=root_directory,
                multiplicative_factor=multiplicative_factor,
                additive_factor=additive_factor, wavelength=wavelength
            )
        else:
            raise Exception('bandset catalog not found')
