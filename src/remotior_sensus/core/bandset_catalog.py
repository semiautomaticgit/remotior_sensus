# Remotior Sensus , software to process remote sensing and GIS data.
# Copyright (C) 2022-2023 Luca Congedo.
# Author: Luca Congedo
# Email: ing.congedoluca@gmail.com
#
# This file is part of Remotior Sensus.
# Remotior Sensus is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.napoleon_type_aliases
# Remotior Sensus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Remotior Sensus. If not, see <https://www.gnu.org/licenses/>.
"""BandSet Catalog manager.

Core class that manages a catalog of BandSets.
A :func:`~remotior_sensus.core.bandset.BandSet` is an object that includes
information about single bands.

The BandSet Catalog allows for the definition of multimple BandSets
by a reference number, and the access to BandSet functions.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # set lists of files
    >>> file_list = ['file_1.tif', 'file_2.tif', 'file_3.tif']
    >>> # create BandSet Catalog
    >>> catalog = rs.bandset_catalog()
    >>> catalog.create_bandset(paths=file_list)
    >>> # for instance get BandSet count
    >>> print(catalog.get_bandset_count())
    1
"""
from copy import deepcopy
from typing import Union, Optional

import numpy

from remotior_sensus.core import (
    configurations as cfg, messages, table_manager as tm
)
from remotior_sensus.core.bandset import BandSet


class BandSetCatalog(object):
    """Manages BandSets.

    This class manages BandSets through a catalog, defining attributes and
    properties of BandSets.

    Attributes:
        bandsets: dictionary of the actual BandSets
        bandsets_table: table of BandSets containing the properties thereof
        get: alias for get_bandset

    Examples:
        Create a BandSet Catalog
            >>> import remotior_sensus
            >>> rs = remotior_sensus.Session()
            >>> catalog = rs.bandset_catalog()
    """  # noqa: E501

    def __init__(self):
        """Initializes the catalog with an empty BandSet"""
        self.bandsets = {}
        self.bandsets_table = None
        self.current_bandset = 1
        # alias for get_bandset
        self.get = self.get_bandset
        # empty BandSet
        self._empty_bandset()

    def get_bandset_count(self) -> int:
        """Gets count of BandSets in the catalog.

        This function gets the count of BandSets present in the catalog.

        Returns:
            The integer number of BandSets.

        Examples:
            Count of BandSets present.
                >>> catalog = BandSetCatalog()
                >>> count = catalog.get_bandset_count()
                >>> print(count)
                1
        """
        return len(self.bandsets)

    @property
    def current_bandset(self) -> int:
        """Property that defines the current BandSet in the catalog.

        This property identifies a BandSet number which is considered
        current (i.e. active) in several other tools when
        no BandSet is specified.

        Returns:
            The integer number of current BandSet.

        Examples:
            Get current BandSet.
                >>> catalog = BandSetCatalog()
                >>> bandset_number = catalog.current_bandset
                >>> print(bandset_number)
                1

            Set current BandSet.
                >>> catalog = BandSetCatalog()
                >>> print(catalog.current_bandset)
                1
                >>> catalog.current_bandset = 2
                >>> print(catalog.current_bandset)
                2
         """
        return self._current_bandset

    @current_bandset.setter
    def current_bandset(self, bandset_number: int):
        if bandset_number > self.get_bandset_count():
            bandset_number = self.get_bandset_count()
        if bandset_number == 0:
            bandset_number = 1
        self._current_bandset = bandset_number

    def get_bandset_by_number(self, number: int) -> Union[None, BandSet]:
        """Get BandSet by number thereof.

        This function gets the BandSet by the number thereof.

        Args:
            number: number of the BandSet

        Returns:
            The BandSet identified by the number.

        Examples:
            Get the first BandSet.
                >>> catalog = BandSetCatalog()
                >>> bandset_1 = catalog.get_bandset_by_number(1)
                >>> print(bandset_1)
                BandSet object
        """
        cfg.logger.log.debug('number: %s' % number)
        bandset = self.get_bandset(number)
        return bandset

    def get_bandset_by_name(
            self, bandset_name: str, output_number: Optional[bool] = False
    ) -> Union[None, int, BandSet]:
        """Get BandSet by name thereof.

        This function gets the BandSet by the name thereof.

        Args:
            bandset_name: name of the BandSet
            output_number: if True then the output is the BandSet number,
                if False then the output is the BandSet

        Returns:
            The BandSet identified by the name.

        Examples:
            Get the number of the BandSet having the name 'example'.
                >>> catalog = BandSetCatalog()
                >>> bandset_number = catalog.get_bandset_by_name(
                bandset_name='example', output_number=True)
                >>> print(bandset_number)
                3

            Get the BandSet having the name 'example'.
                >>> catalog = BandSetCatalog()
                >>> bandset = catalog.get_bandset_by_name(
                bandset_name='example', output_number=False)
                >>> print(bandset)
                BandSet object
        """
        cfg.logger.log.debug(
            'bandset_name: %s; output_number: %s' % (
                bandset_name, output_number)
        )
        uid = self.bandsets_table[
            self.bandsets_table['bandset_name'] == bandset_name]['uid']
        if len(uid) == 0:
            return None
        else:
            if output_number:
                bandset_array = self.bandsets_table['bandset_number'][
                    self.bandsets_table['uid'] == str(uid[0])]
                if len(bandset_array) > 0:
                    return bandset_array[0]
            else:
                return self.bandsets[str(uid[0])]

    def get_bandset_catalog_attributes(
            self, bandset_number: int,
            attribute: Optional[str] = None
    ) -> Union[None, str, list]:
        """Get BandSet Catalog attributes by number of the BandSet.

        This function gets the BandSet Catalog attributes by number of the
            BandSet.

        Args:
            bandset_number: number of the BandSet
            attribute: string of the attribute name, if None then the list
                of attributes is returned

        Returns:
            The BandSet attribute or the list of attributes
                ('bandset_number', 'bandset_name', 'date', 'root_directory',
                'uid').

        Examples:
            Get the 'date' attribute of the BandSet 1.
                >>> catalog = BandSetCatalog()
                >>> date = catalog.get_bandset_catalog_attributes(
                bandset_number=1, attribute='date')
                >>> print(date)
                2000-12-31

            Get the list of attributes of the BandSet 1.
                >>> catalog = BandSetCatalog()
                >>> attributes = catalog.get_bandset_catalog_attributes(
                bandset_number=1)
                >>> print(attributes)
                [(1, 'example', '2000-12-31', 'None', '20000101_1605495327_293')]
        """  # noqa: E501
        if bandset_number is None:
            bandset_number = self.current_bandset
        cfg.logger.log.debug(
            'number: %s; attribute: %s' % (bandset_number, attribute)
        )
        record = self.bandsets_table[
            self.bandsets_table['bandset_number'] == bandset_number]
        if len(record) == 0:
            return None
        else:
            if attribute is None:
                return record.tolist()
            else:
                result = record[attribute][0]
                cfg.logger.log.debug('result: %s' % result)
                return result

    def get_bandset(
            self, bandset_number: int, attribute: Optional[str] = None
    ) -> Union[None, str, list, BandSet]:
        """Get BandSet or BandSet attributes by number of the BandSet.

        This function gets the BandSet or BandSet attributes by the number
            of the BandSet.

        Args:
            bandset_number: number of the BandSet
            attribute: string of the attribute name, if None then the
                BandSet is returned

        Returns:
            The BandSet band attributes or the BandSet.

        Examples:
            Get the 'name' attribute of bands of the BandSet 1.
                >>> catalog = BandSetCatalog()
                >>> names = catalog.get_bandset(bandset_number=1,
                attribute='name')
                >>> print(names)
                ['file1', 'file2', 'file3']

            Get the BandSet 1.
                >>> catalog = BandSetCatalog()
                >>> bandset_1 = catalog.get_bandset(bandset_number=1)
                >>> print(bandset_1)
                BandSet
        """
        if bandset_number is None:
            bandset_number = self.current_bandset
        record = self.bandsets_table[
            self.bandsets_table['bandset_number'] == bandset_number]
        if len(record['uid']) == 0:
            cfg.logger.log.debug('get_bandset: None')
            return None
        uid = record['uid'][0]
        bs = self.bandsets[uid]
        if attribute is None:
            result = bs
        else:
            result = bs.get_band_attributes(attribute)
        cfg.logger.log.debug('bandset_number: %i' % bandset_number)
        return result

    def get_bandset_bands_by_attribute(
            self, bandset_number: int, attribute: str,
            attribute_value: Union[float, int, str],
            output_number: Optional[bool] = False
    ) -> Union[None, list, numpy.recarray]:
        """Get BandSet bands from the attributes thereof.

         This function gets BandSet bands or the band number of bands whose
            attributes match an attribute value.

         Args:
             bandset_number: number of the BandSet
             attribute: string of the attribute name
             attribute_value: value of the band attribute to find
             output_number: if True then the output is the band number list,
                if False then the output is the band array

         Returns:
             The band number list or the array of bands.

         Examples:
             Get the 'name' attribute of bands of the BandSet 1.
                 >>> catalog = BandSetCatalog()
                 >>> band_number = catalog.get_bandset_bands_by_attribute(bandset_number=1,attribute='wavelength',attribute_value=0.443,output_number=True)
                 >>> print(band_number)
                 [2]

             Get the BandSet 1.
                 >>> catalog = BandSetCatalog()
                 >>> band_x = catalog.get_bandset_bands_by_attribute(bandset_number=1,attribute='wavelength',attribute_value=0.443,output_number=True)
                 >>> print(band_x)
                 [(1, 1, '/data/file1.tif', '/data/file1.tif', 'file1', 0.443, 'µm (1 E-6m)', ...]
         """  # noqa: E501
        cfg.logger.log.debug('bandset_number: %s' % bandset_number)
        bs = self.get(bandset_number)
        result = bs.get_bands_by_attributes(
            attribute=attribute, attribute_value=attribute_value,
            output_as_number=output_number
        )
        return result

    def reset(self):
        """Resets the BandSets Catalog.

        This function resets the BandSets Catalog removing all BandSets and
            creating an empty one.

        Examples:
            Reset the BandSets catalog.
                >>> catalog = BandSetCatalog()
                >>> catalog.reset()
        """
        self.current_bandset = 1
        self.bandsets = {}
        self.bandsets_table = None
        # empty BandSet
        self._empty_bandset()
        cfg.logger.log.debug('reset')

    def create_bandset(
            self, paths: Union[list, str] = None,
            band_names: Optional[list] = None,
            wavelengths: Optional[list or str] = None,
            unit: Optional[str] = None,
            multiplicative_factors: Optional[list] = None,
            additive_factors: Optional[list] = None,
            date: Optional[list or str] = None,
            bandset_number: Optional[int] = None,
            insert: Optional[bool] = False,
            root_directory: Optional[str] = None,
            bandset_name: Optional[str] = None,
            box_coordinate_list: Optional[list] = None
    ) -> BandSet:
        """Creates a BandSet adding it to the catalog.

        This function creates a BandSet adding it to the catalog,
        by inserting or replacing the BandSet at specific BandSet number.
        Wavelength is defined by providing a list of values for each band,
        or a string of sensors names as defined in configurations
        satWlList such as:
         
        - Sentinel-2;
        - Landsat8;
        - Landsat5;
        - ASTER.

        Wavelength unit is defined by a string such as:
        
        - band number unitless band order as defined in the BandSet;
        - µm (1 E-6m) micrometers;
        - nm (1 E-9m) nanometers.

        Args:
            paths: list of file paths or a directory path.
            band_names: list of raster names used for identifying the bands, 
                if None then the names are automatically extracted from file names.
            wavelengths: list of center wavelengths of bands or string of sensor names (also partial).
            unit: wavelength unit as string
            multiplicative_factors: multiplicative factors for bands during calculations.
            additive_factors: additive factors for bands during calculations.
            date: list of date strings, or single date string (format YYYY-MM-DD) 
                or string defined in configurations date_auto to detect date from directory name.
            bandset_number: number of the BandSet, replacing an existing BandSet with the same number.
            insert: if True insert the BandSet at bandset_number, if False replace the BandSet number.
            root_directory: root directory for relative path.
            bandset_name: name of the BandSet.
            box_coordinate_list: list of coordinates [left, top, right, bottom] 
                to create a virtual subset.

        Returns:
            The created BandSet.

        Examples:
            Create a first BandSet from a file list with files inside a data directory, setting root_directory, defining the BandSet date.
                >>> catalog = BandSetCatalog()
                >>> file_list = ['file1.tif', 'file2.tif', 'file3.tif']
                >>> bandset_date = '2021-01-01'
                >>> data_directory = 'data'
                >>> bandset = catalog.create_bandset(
                ... paths=file_list, wavelengths=['Sentinel-2'], date=bandset_date, 
                ... root_directory=data_directory
                ... )

            Create a new BandSet from a file list with files inside a data directory (setting root_directory), defining the BandSet date, and explicitly defining the BandSet number.
                >>> catalog = BandSetCatalog()
                >>> file_list = ['file1.tif', 'file2.tif', 'file3.tif']
                >>> bandset_date = '2021-01-01'
                >>> data_directory = 'data'
                >>> bandset = catalog.create_bandset(
                ... paths=file_list, wavelengths=['Sentinel-2'], date=bandset_date, 
                ... bandset_number=2, root_directory=data_directory
                ... )
        """  # noqa: E501
        cfg.logger.log.debug('start')
        bst = BandSet.create(
            paths, band_names=band_names, wavelengths=wavelengths, unit=unit,
            multiplicative_factors=multiplicative_factors,
            additive_factors=additive_factors, dates=date,
            root_directory=root_directory, name=bandset_name,
            box_coordinate_list=box_coordinate_list
        )
        self.add_bandset(
            bandset=bst, bandset_number=bandset_number, insert=insert
        )
        cfg.logger.log.debug('end')
        return bst

    def _load(self, bandset_catalog, current_bandset: int = 1):
        """Loads a BandSet Catalog.

        This function loads a BandSet Catalog, and optionally defines the
        current BandSet.

        Args:
            bandset_catalog: BandSet Catalog
            current_bandset: optional integer number of current BandSet
        """
        cfg.logger.log.debug('start')
        self.reset()
        # copy BandSet Catalog
        bandset_catalog_copy = deepcopy(bandset_catalog)
        if bandset_catalog_copy.bandsets is not None:
            self.bandsets = bandset_catalog_copy.bandsets
        if bandset_catalog_copy.bandsets_table is None:
            # empty BandSet
            self._empty_bandset()
        else:
            self.bandsets_table = bandset_catalog_copy.bandsets_table
        self.current_bandset = current_bandset
        cfg.logger.log.debug('end')

    def move_band_in_bandset(
            self, band_number_input: int, band_number_output: int,
            bandset_number: Optional[int] = None, wavelength=True
    ):
        """Moves band in Bandset.

         This function reorders a band in a Bandset.

         Args:
            band_number_input: number of band to be moved.
            band_number_output: position of the moved band.
            bandset_number: number of BandSet; if None, current BandSet is used.
            wavelength: if True, keep the wavelength attributes of the band order.

         Examples:
             Move the second band of the BandSet 1 to position 5.
                 >>> catalog = BandSetCatalog()
                 >>> catalog.move_band_in_bandset(
                 ... bandset_number=1, band_number_input=2, band_number_output=5
                 ... )
         """  # noqa: E501
        cfg.logger.log.debug('start')
        if bandset_number is None:
            bandset_number = self.current_bandset
        bandset = self.get_bandset_by_number(bandset_number)
        wavelength_list = bandset.get_wavelengths()
        # temporary set -1 to moved band
        bandset.bands['band_number'][
            bandset.bands['band_number'] == band_number_input] = -1
        # move down bands between band_number_input and band_number_output
        if band_number_input < band_number_output:
            bandset.bands['band_number'][
                (bandset.bands['band_number'] > band_number_input) & (
                        bandset.bands['band_number'] <= band_number_output)] \
                = bandset.bands['band_number'][
                      (bandset.bands['band_number'] > band_number_input) & (
                              bandset.bands[
                                  'band_number'] <= band_number_output)] - 1
        # move up bands between band_number_input and band_number_output
        elif band_number_input > band_number_output:
            bandset.bands['band_number'][
                (bandset.bands['band_number'] >= band_number_input) & (
                        bandset.bands['band_number'] < band_number_output)] = \
                bandset.bands['band_number'][
                    (bandset.bands['band_number'] >= band_number_input) & (
                            bandset.bands[
                                'band_number'] < band_number_output)] + 1
        # set band_number_output to moved band
        bandset.bands['band_number'][
            bandset.bands['band_number'] == -1] = band_number_output
        bandset.bands.sort(order='band_number')
        if wavelength:
            bandset.bands['wavelength'] = wavelength_list
        cfg.logger.log.debug('end')

    def remove_band_in_bandset(
            self, band_number: int, bandset_number: Optional[int] = None
    ):
        """Removes band in Bandset.

         This function removes a band in Bandset identified by a number.
         Automatically reorders other band numbers.

         Args:
            band_number: number of band to be removed.
            bandset_number: number of BandSet; if None, current BandSet is used.

         Examples:
             Remove the second band of the BandSet.
                 >>> catalog = BandSetCatalog()
                 >>> catalog.remove_band_in_bandset(
                 ... bandset_number=1, band_number=2
                 ... )
         """  # noqa: E501
        cfg.logger.log.debug('start')
        if bandset_number is None:
            bandset_number = self.current_bandset
        bandset = self.get_bandset(bandset_number)
        # remove from table
        bandset.bands = bandset.bands[
            bandset.bands['band_number'] != band_number]
        # reorder other band numbers
        if band_number < bandset.get_band_count():
            bandset.bands['band_number'][
                bandset.bands['band_number'] > band_number] = \
                bandset.bands['band_number'][
                    bandset.bands['band_number'] > band_number] - 1
        cfg.logger.log.debug('end')

    def add_band_to_bandset(
            self, path: str, bandset_number: Optional[int] = None,
            band_number: Optional[int] = None,
            raster_band: Optional[int] = None, band_name: Optional[str] = None,
            date: Optional[str] = None, unit: Optional[str] = None,
            root_directory: Optional[str] = None,
            multiplicative_factor: Optional[int] = None,
            additive_factor: Optional[int] = None,
            wavelength: Optional[float] = None
    ):
        """Adds a new band to BandSet.

        This function creates a new band and adds it to a BandSet.

        Args:
            path: file path.
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
            Add a band to BandSet 1.
                >>> catalog = BandSetCatalog()
                >>> catalog.add_band_to_bandset(
                    path='file1.tif', bandset_number=1, band_number=1, raster_band=1
                ... )
        """  # noqa: E501
        cfg.logger.log.debug('start')
        if bandset_number is None:
            bandset_number = self.current_bandset
        self.get_bandset_by_number(bandset_number).add_new_band(
            path=path, band_number=band_number, raster_band=raster_band,
            band_name=band_name, date=date,
            root_directory=root_directory,
            multiplicative_factor=multiplicative_factor,
            additive_factor=additive_factor,
            wavelength=wavelength, unit=unit
        )
        cfg.logger.log.debug('end')

    def sort_bands_by_wavelength(self, bandset_number: Optional[int] = None):
        """Sorts bands by wavelength.

         This function numerically sorts bands in a BandSet by wavelength center.

         Args:
            bandset_number: number of BandSet; if None, current BandSet is used.

         Examples:
            Sort bands in BandSet 1.
                >>> catalog = BandSetCatalog()
                >>> catalog.sort_bands_by_wavelength(bandset_number=1)
         """  # noqa: E501
        cfg.logger.log.debug('bandset_number: %s' % bandset_number)
        if bandset_number is None:
            bandset_number = self.current_bandset
        self.get_bandset(bandset_number).sort_bands_by_wavelength()

    def add_bandset(
            self, bandset: BandSet, bandset_number: Optional[int] = None,
            insert=False
    ):
        """Adds a BandSet to Catalog.

         This function adds a previously created BandSet to BandSet Catalog.

         Args:
            bandset: the BandSet to be added.
            bandset_number: number of BandSet; if None, current BandSet is used.
            insert: if True insert the BandSet at bandset_number (other BandSets are moved), 
                if False replace the BandSet number.
                
         Examples:
            Insert a BandSet as BandSet 1.
                >>> catalog = BandSetCatalog()
                >>> catalog.add_bandset(bandset_number=1, insert=True
                ... )
         """  # noqa: E501
        cfg.logger.log.debug('bandset_number: %s' % bandset_number)
        # copy BandSet
        bandset_copy = deepcopy(bandset)
        bandset_count = self.get_bandset_count()
        if not insert:
            if bandset_number is None:
                bandset_number = self.current_bandset
            # new BandSet
            else:
                # add new BandSet if bandset_number > BandSet count
                if bandset_number > bandset_count:
                    bandset_number = bandset_count + 1
            if bandset_number <= bandset_count:
                # temporary set -1 to replaced bandset
                self.bandsets_table['bandset_number'][
                    (self.bandsets_table[
                         'bandset_number'] == bandset_number)] = -1
        # check existing BandSet
        else:
            try:
                if bandset_number is None:
                    bandset_number = self.current_bandset
                # reorder band sets
                if (bandset_number <= bandset_count
                        and self.bandsets_table is not None):
                    self.bandsets_table['bandset_number'][
                        self.bandsets_table[
                            'bandset_number'] >= bandset_number] = \
                        self.bandsets_table['bandset_number'][
                            self.bandsets_table[
                                'bandset_number'] >= bandset_number] + 1
                elif bandset_number >= bandset_count + 1:
                    bandset_number = bandset_count + 1
            except Exception as err:
                cfg.logger.log.error(str(err))
                messages.error(str(err))
        # add BandSet to dictionary by uid
        bandset_copy.uid = bandset_copy.generate_uid()
        self.bandsets[bandset_copy.uid] = bandset_copy
        if bandset_copy.box_coordinate_list is not None:
            box_coordinate_left = bandset_copy.box_coordinate_list[0]
            box_coordinate_top = bandset_copy.box_coordinate_list[1]
            box_coordinate_right = bandset_copy.box_coordinate_list[2]
            box_coordinate_bottom = bandset_copy.box_coordinate_list[3]
        else:
            box_coordinate_left = box_coordinate_top = box_coordinate_right \
                = box_coordinate_bottom = None
        self.bandsets_table = tm.create_bandset_catalog_table(
            bandset_number=bandset_number,
            root_directory=bandset_copy.root_directory, date=bandset_copy.date,
            bandset_uid=bandset_copy.uid, bandset_name=bandset_copy.name,
            previous_catalog=self.bandsets_table,
            crs=bandset_copy.crs, box_coordinate_left=box_coordinate_left,
            box_coordinate_top=box_coordinate_top,
            box_coordinate_right=box_coordinate_right,
            box_coordinate_bottom=box_coordinate_bottom
        )
        if not insert and bandset_number <= bandset_count:
            self._remove_bandset(-1, reorder=False)

    def move_bandset(
            self, bandset_number_input: int, bandset_number_output: int
    ):
        """Moves a BandSet.

         This function reorders a BandSet in the BandSet Catalog.

         Args:
            bandset_number_input: the BandSet number to be moved.
            bandset_number_output: the new poistion of BandSet.

         Examples:
            Move BandSet 1 to position 3.
                >>> catalog = BandSetCatalog()
                >>> catalog.move_bandset(bandset_number_input=1, bandset_number_output=3)
         """  # noqa: E501
        cfg.logger.log.debug(
            'bandset_number_input: %s' % str(bandset_number_input)
        )
        # temporary set -1 to moved bandset
        self.bandsets_table['bandset_number'][
            (self.bandsets_table[
                 'bandset_number'] == bandset_number_input)] = -1
        # move down BandSets between bandset_number_input and
        # bandset_number_output
        if bandset_number_input < bandset_number_output:
            self.bandsets_table['bandset_number'][
                (self.bandsets_table['bandset_number'] >
                 bandset_number_input) & (
                        self.bandsets_table['bandset_number'] <=
                        bandset_number_output)] = \
                self.bandsets_table['bandset_number'][
                    (self.bandsets_table['bandset_number'] >
                     bandset_number_input) & (
                            self.bandsets_table['bandset_number'] <=
                            bandset_number_output)] - 1
        # move up BandSets between bandset_number_input and
        # bandset_number_output
        elif bandset_number_input > bandset_number_output:
            self.bandsets_table['bandset_number'][
                (self.bandsets_table['bandset_number'] >=
                 bandset_number_output) & (
                        self.bandsets_table['bandset_number']
                        < bandset_number_input)] = \
                self.bandsets_table[
                    'bandset_number'][
                    (self.bandsets_table['bandset_number'] >=
                     bandset_number_output) & (
                            self.bandsets_table['bandset_number'] <
                            bandset_number_input)] + 1
        cfg.logger.log.info(
            'bandset bandset_number_input: %s; bandset '
            'bandset_number_output: %s'
            % (str(bandset_number_input), str(bandset_number_output))
        )
        # set bandset_number to moved BandSet
        self.bandsets_table['bandset_number'][
            (self.bandsets_table['bandset_number'] == -1)] = \
            bandset_number_output
        # sort
        self.bandsets_table.sort(order='bandset_number')

    def remove_bandset(self, bandset_number: int):
        """Removes a BandSet.

         This function removes a BandSet by the number thereof.

         Args:
            bandset_number: the BandSet number to be removed.

         Examples:
            Remove BandSet 2.
                >>> catalog = BandSetCatalog()
                >>> catalog.remove_bandset(2)
         """  # noqa: E501
        self._remove_bandset(bandset_number, reorder=True)

    def _remove_bandset(self, bandset_number: int, reorder=True):
        """Function to remove a BandSet by number and optional reorder."""
        cfg.logger.log.debug('bandset_number: %s' % str(bandset_number))
        if bandset_number > self.get_bandset_count():
            cfg.logger.log.warning('bandset_number: %s' % str(bandset_number))
            messages.warning('bandset_number: %s' % str(bandset_number))
        else:
            record = self.bandsets_table[
                self.bandsets_table['bandset_number'] == bandset_number]
            uid = record['uid'][0]
            # remove from table
            self.bandsets_table = self.bandsets_table[
                self.bandsets_table['uid'] != uid]
            # move down BandSets above removed BandSet
            if reorder and bandset_number < self.get_bandset_count():
                self.bandsets_table['bandset_number'][
                    (self.bandsets_table['bandset_number'] >
                     bandset_number)] = \
                    self.bandsets_table[
                        'bandset_number'][(self.bandsets_table[
                                               'bandset_number'] >
                                           bandset_number)] - 1
            # remove from dictionary
            self.bandsets.pop(str(uid), None)
            if self.get_bandset_count() == 0:
                # empty BandSet
                self._empty_bandset()

    def _empty_bandset(self):
        """Function to create an empty BandSet."""
        bandset = BandSet.create()
        bandset_number = 1
        # add BandSet to dictionary by uid
        bandset.uid = bandset.generate_uid()
        self.bandsets[bandset.uid] = bandset
        self.bandsets_table = tm.create_bandset_catalog_table(
            bandset_number=bandset_number,
            root_directory=bandset.root_directory, date=bandset.date,
            bandset_uid=bandset.uid, bandset_name=bandset.name,
            previous_catalog=self.bandsets_table
        )
        cfg.logger.log.debug('empty bandset')

    @staticmethod
    def get_band_list(
            bandset: Union[int, list, BandSet] = None,
            bandset_catalog: Optional = None
            ) -> list:
        """Gets band list.

        This function gets band list from several types of input.

        Args:
            bandset: input of type BandSet or list of paths or integer number of BandSet.
            bandset_catalog: optional type BandSetCatalog if bandset argument is a number.
            
        Returns:
            If argument bandset is a BandSet or a BandSet number, 
                returns list of bands in the BandSet; if argument bandset is 
                already a list, returns the same list.
            
        Examples:
            Get band list of BandSet 1.
                >>> # import Remotior Sensus and start the session
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> catalog = BandSetCatalog()
                >>> list_1 = rs.bandset_catalog.get_band_list(1, catalog)
                >>> # which is equivalent to
                >>> bandset_1 = bandset_catalog.get_bandset(1)
                >>> list_2 = bandset_1.get_absolute_paths()
        """  # noqa: E501
        # BandSet
        if type(bandset) is BandSet:
            band_list = bandset.get_absolute_paths()
        # list of paths
        elif type(bandset) is list:
            band_list = bandset
        # number of BandSet
        elif bandset_catalog is not None:
            b = bandset_catalog.get_bandset(bandset)
            band_list = b.get_absolute_paths()
        else:
            band_list = None
        cfg.logger.log.debug('band list: %s' % str(band_list))
        return band_list

    def iterate_bandset_bands(self, attribute: str) -> list:
        """Iterates BandSet attributes.

        This function gets BandSet attributes iterating all the BandSets 
            in the Catalog.

        Args:
            attribute: attribute name.

        Returns:
            List of attributes

        Examples:
            Get BandSet names.
                >>> catalog = BandSetCatalog()
                >>> names = catalog.iterate_bandset_bands('name')
        """  # noqa: E501
        attributes = []
        for i in range(1, self.get_bandset_count() + 1):
            attributes.extend(self.get_bandset(i, attribute))
        return attributes

    def find_bandset_names_in_list(
            self, names: list, lower=True, output_number=True,
            exact_match=False
    ) -> list:
        """Finds BandSet names.

        This function finds BandSet names in a list and return BandSets 
            or BandSet numbers.

        Args:
            names: list of string names.
            lower: if True, finds by lowering all the names.
            output_number: if True returns the number of the band, if False returns the BandSet.
            exact_match: if True, names are compared by equality.

        Returns:
            if output_number is True, returns list of band number;
            if output_number is False, returns list of BandSets.

        Examples:
            Find BandSet from name list.
                >>> catalog = BandSetCatalog()
                >>> name_list = ['name1', 'name2']
                >>> bandsets = catalog.find_bandset_names_in_list(names=name_list)
        """  # noqa: E501
        bandset_names = self.bandsets_table['bandset_name']
        bandset_list = []
        # iterate names
        for bandset_name in bandset_names.tolist():
            bandset_name_check = bandset_name
            for name in names:
                if lower:
                    name = name.lower()
                    bandset_name_check = bandset_name.lower()
                if exact_match:
                    if name == bandset_name_check:
                        bandset_list.append(
                            self.get_bandset_by_name(
                                bandset_name, output_number=output_number
                            )
                        )
                        break
                else:
                    if name in bandset_name_check:
                        bandset_list.append(
                            self.get_bandset_by_name(
                                bandset_name, output_number=output_number
                            )
                        )
                        break
        return bandset_list

    def set_root_directory(
            self, root_directory: str, bandset_number: Optional[int] = None
    ):
        """Sets BandSet root directory.

        Sets BandSet root directory for relative path.

        Args:
            root_directory: root directory path.
            bandset_number: number of BandSet; if None, current BandSet is used.

        Examples:
            Set BandSet 1 root directory.
                >>> catalog = BandSetCatalog()
                >>> catalog.set_root_directory(bandset_number=1, root_directory='data_directory')
        """  # noqa: E501
        if bandset_number is None:
            bandset_number = self.current_bandset
        self.get_bandset(bandset_number).root_directory = root_directory
        self.bandsets_table['root_directory'][self.bandsets_table[
                                                  'bandset_number'] ==
                                              bandset_number] = root_directory

    def set_date(self, date: str, bandset_number: Optional[int] = None):
        """Sets BandSet date.

        Sets BandSet date.

        Args:
            date: date string (format YYYY-MM-DD).
            bandset_number: number of BandSet; if None, current BandSet is used.

        Examples:
            Set BandSet 1 date.
                >>> catalog = BandSetCatalog()
                >>> catalog.set_date(bandset_number=1, date='2021-01-01')
        """  # noqa: E501
        if bandset_number is None:
            bandset_number = self.current_bandset
        self.get_bandset(bandset_number).date = date
        self.bandsets_table['date'][
            self.bandsets_table['bandset_number'] == bandset_number] = date

    def set_name(self, name: str, bandset_number: int = None):
        """Sets BandSet name.

        Sets BandSet name.

        Args:
            name: name.
            bandset_number: number of BandSet; if None, current BandSet is used.

        Examples:
            Set BandSet 1 name.
                >>> catalog = BandSetCatalog()
                >>> catalog.set_name(bandset_number=1, name='example')
        """  # noqa: E501
        if bandset_number is None:
            bandset_number = self.current_bandset
        self.get_bandset(bandset_number).name = name
        self.bandsets_table['bandset_name'][
            self.bandsets_table['bandset_number'] == bandset_number] = name

    def set_box_coordinate_list(
            self, box_coordinate_list: list,
            bandset_number: Optional[int] = None
    ):
        """Sets BandSet box coordinate list.

        Sets BandSet box coordinate list  for virtual subset.

        Args:
            box_coordinate_list: list of coordinates [left, top, right, bottom] to create a virtual subset.
            bandset_number: number of BandSet; if None, current BandSet is used.

        Examples:
            Set BandSet 1 coordinate list.
                >>> catalog = BandSetCatalog()
                >>> catalog.set_box_coordinate_list(
                ... bandset_number=1, box_coordinate_list=[230000, 4680000, 232000, 4670000]
                ... )
        """  # noqa: E501
        if bandset_number is None:
            bandset_number = self.current_bandset
        self.get_bandset(
            bandset_number
        ).box_coordinate_list = box_coordinate_list
        self.bandsets_table['box_coordinate_left'][
            self.bandsets_table['bandset_number'] == bandset_number] = \
            box_coordinate_list[0]
        self.bandsets_table['box_coordinate_top'][
            self.bandsets_table['bandset_number'] == bandset_number] = \
            box_coordinate_list[1]
        self.bandsets_table['box_coordinate_right'][
            self.bandsets_table['bandset_number'] == bandset_number] = \
            box_coordinate_list[2]
        self.bandsets_table['box_coordinate_bottom'][
            self.bandsets_table['bandset_number'] == bandset_number] = \
            box_coordinate_list[3]

    def get_root_directory(self, bandset_number: Optional[int] = None) -> str:
        """Gets BandSet root directory.

        Gets BandSet root directory.

        Args:
            bandset_number: number of BandSet; if None, current BandSet is used.

        Returns:
            Root directory string.
            
        Examples:
            Get BandSet 1 root directory.
                >>> catalog = BandSetCatalog()
                >>> catalog.get_root_directory(1)
        """  # noqa: E501
        if bandset_number is None:
            bandset_number = self.current_bandset
        return self.get_bandset_catalog_attributes(
            bandset_number=bandset_number, attribute='root_directory'
        )

    def get_date(self, bandset_number: Optional[int] = None) -> str:
        """Gets BandSet date.

        Gets BandSet date.

        Args:
            bandset_number: number of BandSet; if None, current BandSet is used.

        Returns:
            Date string.
            
        Examples:
            Get BandSet 1 date.
                >>> catalog = BandSetCatalog()
                >>> catalog.get_date(1)
        """  # noqa: E501
        if bandset_number is None:
            bandset_number = self.current_bandset
        return str(
            self.get_bandset_catalog_attributes(
                bandset_number=bandset_number, attribute='date'
            )
        )

    def get_name(self, bandset_number: Optional[int] = None) -> str:
        """Gets BandSet name.

        Gets BandSet name.

        Args:
            bandset_number: number of BandSet; if None, current BandSet is used.

        Returns:
            BandSet name.
            
        Examples:
            Get BandSet 1 date.
                >>> catalog = BandSetCatalog()
                >>> catalog.get_name(1)
        """  # noqa: E501
        if bandset_number is None:
            bandset_number = self.current_bandset
        return str(
            self.get_bandset_catalog_attributes(
                bandset_number=bandset_number, attribute='bandset_name'
            )
        )

    def get_band_count(
            self, bandset_number: Optional[int] = None
    ) -> Union[None, int]:
        """Gets band count.

        Gets band count of a BandSet.

        Args:
            bandset_number: number of BandSet; if None, current BandSet is used.

        Returns:
            count of bands.
            
        Examples:
            Get BandSet 1 band count.
                >>> catalog = BandSetCatalog()
                >>> catalog.get_band_count(1)
         """  # noqa: E501
        if bandset_number is None:
            bandset_number = self.current_bandset
        bandset = self.get(bandset_number=bandset_number)
        if bandset is None:
            return None
        else:
            count = bandset.get_band_count()
            cfg.logger.log.debug('band_count: %s' % str(count))
            return count

    def get_box_coordinate_list(
            self, bandset_number: int = None
    ) -> Union[None, list]:
        """Gets BandSet box coordinate list.

        Gets BandSet box coordinate list.

        Args:
            bandset_number: number of BandSet; if None, current BandSet is used.

        Returns:
            List of box coordinates [left, top, right, bottom].
            
        Examples:
            Get BandSet 1 band count.
                >>> catalog = BandSetCatalog()
                >>> catalog.get_box_coordinate_list(1)
        """  # noqa: E501
        if bandset_number is None:
            bandset_number = self.current_bandset
        bandset = self.get(bandset_number=bandset_number)
        if bandset is None:
            return None
        else:
            box_coordinate_list = [
                self.bandsets_table['box_coordinate_left'][
                    self.bandsets_table['bandset_number'] == bandset_number][
                    0],
                self.bandsets_table['box_coordinate_top'][
                    self.bandsets_table['bandset_number'] == bandset_number][
                    0],
                self.bandsets_table['box_coordinate_right'][
                    self.bandsets_table['bandset_number'] == bandset_number][
                    0],
                self.bandsets_table['box_coordinate_bottom'][
                    self.bandsets_table['bandset_number'] == bandset_number][
                    0]]
            return box_coordinate_list

    def create_band_string_list(self, bandset_number: int = None) -> list:
        """Creates band string list.

        Creates band string list such as "bandset1b1" for all the bands in a BandSet. 
        Used in tool :func:`~remotior_sensus.tools.band_calc`.

        Args:
            bandset_number: number of BandSet; if None, current BandSet is used.
            
        Returns:
            List of band string such as ["bandset1b1", "bandset1b2"].
            
        Examples:
            Get BandSet 1 band string list.
                >>> catalog = BandSetCatalog()
                >>> catalog.create_band_string_list(1)
         """  # noqa: E501
        number = self.get(bandset_number).bands['band_number']
        string_1 = numpy.char.add(
            '%s%s%s%s' % (cfg.variable_band_quotes, cfg.variable_bandset_name,
                          str(bandset_number), cfg.variable_band_name),
            number.astype('<U16')
        )
        string_2 = numpy.char.add(string_1, cfg.variable_band_quotes)
        return string_2.tolist()
