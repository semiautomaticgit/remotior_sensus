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
"""BandSet manager.

Core class that manages BandSets.
A BandSet is an object that includes information about single bands
(from the file path to the spatial and spectral characteristics).
Bands in a BandSet can be referenced by the properties thereof,
such as order number or center wavelength.

BandSets can be used as input for operations on multiple bands
such as Principal Components Analysis, classification, mosaic,
or band calculation.
Multimple BandSets can be defined and identified by their reference number
in the BandSet Catalog.

Most BandSet functions can be accessed through the BandSet Catalog.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # set lists of files, names and center wavelength
    >>> file_list = ['file_1.tif', 'file_2.tif', 'file_3.tif']
    >>> band_names = ['name_1', 'name_2', 'name_3']
    >>> wavelengths = [0.6, 0.7, 0.8]
    >>> # set optional date
    >>> date = '2020-01-01'
    >>> # create a BandSet
    >>> bandset = rs.bandset.create(
    ... file_list, band_names=band_names, wavelengths=wavelengths, dates=date
    ... )
    >>> # get list of absolute paths of bands
    >>> absolute_paths = bandset.get_absolute_paths()
    >>> # bandset wavelengths
    >>> wavelengths = bandset.get_wavelengths()
    >>> # create a BandSet with a root directory for files in relative paths
    >>> root_directory = 'root_directory'
    >>> relative_file_list = ['file_1.tif', 'file_2.tif', 'file_3.tif']
    >>> bandset = rs.bandset.create(file_list,root_directory=root_directory)
    >>> # get list of relative paths of bands
    >>> relative_paths = bandset.get_paths()
    >>> # get band by nearest wavelength
    >>> band_x = bandset.get_band_by_wavelength(0.8,threshold=0.1)
"""

import random
from typing import Union, Optional

import numpy as np

from remotior_sensus.core import (
    configurations as cfg, messages, table_manager as tm
)
from remotior_sensus.util import dates_times, files_directories, raster_vector


class BandSet(object):
    """Manages band sets.

    This module allows for managing bands in a BandSet.

    Attributes:
        bands: BandSet of band tables.
        crs: BandSet coordinate reference system.
        get: alias for get_band function.
        
    Examples:
        Create a BandSet
            >>> import remotior_sensus
            >>> rs = remotior_sensus.Session()
            >>> file_list = ['file_1.tif', 'file_2.tif', 'file_3.tif']
            >>> bandset = rs.bandset.create(file_list)
            
        Get the first band
            >>> band_1 = bandset.get_band(1)
    """  # noqa: E501

    def __init__(
            self, bandset_uid, bands_list=None, name=None, date=None,
            root_directory=None, crs=None, box_coordinate_list=None
    ):
        """Initializes a BandSet.

        Initializes a BandSet defining a unique ID.
        One should use the create method for creating new BandSets.       

        Args:
            bandset_uid: unique BandSet ID.
            bands_list: list of band tables.
            name: optional BandSet name.
            date: optional BandSet date.
            root_directory: optional BandSet root directory for relative path.
            crs: BandSet coordinate reference system.
            box_coordinate_list: list of coordinates [left, top, right, bottom] 
                to create a virtual subset.

        Examples:
            Initialize a BandSet
                >>> BandSet.create(name=name,box_coordinate_list=box_coordinate_list)
        """  # noqa: E501
        self.bands = tm.create_bandset_table(bands_list)
        # unique ID
        self.uid = bandset_uid
        self.date = date
        self.root_directory = root_directory
        self.crs = crs
        # alias for get_band
        self.get = self.get_band
        if name is None:
            names = self.get_band_attributes('name')
            try:
                self.name = names[0][:-2]
            except Exception as err:
                str(err)
                if names is not None:
                    if len(names) > 0:
                        self.name = names[0]
                    else:
                        self.name = 'BandSet'
                else:
                    self.name = 'BandSet'
        else:
            self.name = name
        self.box_coordinate_list = box_coordinate_list

    @property
    def date(self):
        """Optional date."""
        return self._date

    @date.setter
    def date(self, date):
        self._date = date

    @property
    def uid(self):
        """str: Unique BandSet ID."""
        return self._uid

    @uid.setter
    def uid(self, uid):
        self._uid = uid

    @property
    def root_directory(self):
        """str: Optional BandSet root directory for relative path."""
        return self._root_directory

    @root_directory.setter
    def root_directory(self, root_directory):
        self._root_directory = root_directory

    @property
    def name(self):
        """str: Optional BandSet name."""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def box_coordinate_list(self):
        """
        list: Optional BandSet box coordinate list for virtual subset.
        """  # noqa: E501
        return self._box_coordinate_list

    @box_coordinate_list.setter
    def box_coordinate_list(self, box_coordinate_list):
        self._box_coordinate_list = box_coordinate_list

    def get_band_count(self) -> int:
        """Gets the count of bands."""
        if self.bands is None:
            return 0
        else:
            return self.bands.shape[0]

    def get_paths(self) -> list:
        """Gets the list of bands."""
        return self.bands['path'].tolist()

    def get_absolute_paths(self) -> list:
        """Gets the list of absolute paths."""
        paths = self.bands['absolute_path'].tolist()
        cfg.logger.log.debug('absolute_paths: %s' % str(paths))
        return paths

    def get_wavelengths(self) -> list:
        """Gets the list of center wavelength."""
        return self.bands['wavelength'].tolist()

    def get_wavelength_units(self) -> list:
        """Gets the list of wavelength units."""
        return self.bands['wavelength_unit'].tolist()

    def get_raster_band_list(self) -> list:
        """Gets the list of raster bands."""
        raster_band = self.bands['raster_band'].tolist()
        band_list = []
        for n in raster_band:
            band_list.append([n])
        return band_list

    def get_band_attributes(self, attribute: str = None) -> list:
        """Gets band attributes.

        Gets an attribute of bands.

        Args:
            attribute: attribute name.
            
        Returns:
            returns list of attributes; if attribute is None returns the bands;  
            if attribute is not found returns None.

        Examples:
            Get band names
            >>> bandset = BandSet()
            >>> names = bandset.get_band_attributes('name')
        """  # noqa: E501
        cfg.logger.log.debug('attribute: %s' % str(attribute))
        if attribute is None:
            result = self.bands
            cfg.logger.log.debug('get bands')
        else:
            if self.bands is not None:
                record = self.bands[attribute]
                if len(record) == 0:
                    result = None
                else:
                    result = record.tolist()
            else:
                result = None
        return result

    def get_bands_by_attributes(
            self, attribute: str, attribute_value, output_as_number=False
    ) -> np.array:
        """Gets bands by attribute.

        Gets bands identified by an attribute value (identified bands have 
        the attribute value equal to the attribute_value).

        Args:
            attribute: attribute name.
            attribute_value: attribute value.
            output_as_number: if True returns the number of the band, if False returns the band table.
            
        Returns:
            if output_number is True, returns list of band number;
            if output_number is False, returns bands identified by attribute;
            returns None if no band has the attribute value.

        Examples:
            Get bands by attributes
            >>> bandset = BandSet()
            >>> band_x = bandset.get_bands_by_attributes('wavelength',0.8)
        """  # noqa: E501
        cfg.logger.log.debug(
            'attribute: %s; attribute_value: %s; output_number: %s' % (
                attribute, attribute_value, output_as_number)
        )
        result = None
        if self.bands is not None:
            bands = self.bands[self.bands[attribute] == attribute_value]
            if not output_as_number:
                result = bands
            else:
                result = bands['band_number'].tolist()
        return result

    def get_band_alias(self) -> list:
        """Gets bands alias (with band number) used in band_calc

        Returns:
            the list of band aliases.
        """
        result = []
        if self.bands is not None:
            for i in range(1, len(self.bands) + 1):
                result.append('%s%s' % (cfg.variable_band_name, i))
        cfg.logger.log.debug('result: %s' % str(result))
        return result

    def get_band(self, number=None) -> np.array:
        """Gets a band.

        Gets a band by number.

        Args:
            number: band number.

        Returns:
            the band table; if number is None returns all the bands;
            returns None if band number is not found.

        Examples:
            Get bands by number
            >>> bandset = BandSet()
            >>> # get first band
            >>> band = bandset.get_band(1)
            >>> # band x size
            >>> band_x_size = band.x_size
            >>> # get band nodata directly
            >>> band_nodata = bandset.get_band(1).nodata
        """  # noqa: E501
        cfg.logger.log.debug('number: %s' % number)
        if number is None:
            result = self.bands
        elif not number:
            result = None
        else:
            try:
                result = self.bands[self.bands['band_number'] == number]
            except Exception as err:
                str(err)
                result = None
        return result

    def reset(self):
        """Resets a BandSet.

        Resets a BandSet destroyng bands and attributes.

        Examples:
            Reset a BandSet
            >>> bandset = BandSet()
            >>> # reset
            >>> bandset.reset()
        """  # noqa: E501
        self.bands = self.name = self.date = self.root_directory = None
        self.box_coordinate_list = self.crs = None
        cfg.logger.log.debug('reset')

    @classmethod
    def create(
            cls, paths: Union[list, str] = None,
            band_names: Optional[list] = None,
            wavelengths: Optional[list or str] = None,
            unit: Optional[str] = None,
            multiplicative_factors: Optional[list] = None,
            additive_factors: Optional[list] = None,
            dates: Optional[list or str] = None,
            root_directory: Optional[str] = None,
            box_coordinate_list: Optional[list] = None,
            name: Optional[str] = None
    ):
        """Creates a BandSet.

        This method creates a BandSet defined by input files.
        Raster properties are derived from files to populate 
        the table of bands.

        Args:
            name: BandSet name.
            paths: list of file paths or a directory path.
            band_names: list of raster names used for identifying the bands, 
                if None then the names are automatically extracted from file names.
            wavelengths: list of center wavelengths of bands or string of sensor names (also partial).
            unit: wavelength unit.
            multiplicative_factors: multiplicative factors for bands during calculations.
            additive_factors: additive factors for bands during calculations.
            dates: list of date strings, or single date string (format YYYY-MM-DD) 
                or string defined in configurations date_auto to detect date from directory name.
            root_directory: root directory for relative path.
            box_coordinate_list: list of coordinates [left, top, right, bottom] 
                to create a virtual subset.

        Returns:
            returns BandSet.
            
        Examples:
            Given that a session was previously started
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> # set lists of files, names and center wavelength
                >>> file = ['file_1.tif', 'file_2.tif', 'file_3.tif']
                >>> names = ['name_1', 'name_2', 'name_3']
                >>> wavelength = [0.6, 0.7, 0.8]
                >>> # create a BandSet
                >>> bandset = rs.bandset.create(file,band_names=names,wavelengths=wavelength)
        """  # noqa: E501
        cfg.logger.log.info('start')
        bandset_uid = cls.generate_uid()
        satellite = None
        multiband = None
        # directory
        if paths is None:
            cfg.logger.log.info('end')
            return cls(bandset_uid=bandset_uid, name=name)
        elif len(paths) <= 2 and files_directories.is_directory(
                files_directories.relative_to_absolute_path(
                    paths[0], root_directory
                )
        ):
            try:
                filters = paths[1].strip("'").strip()
            except Exception as err:
                cfg.logger.log.error(str(err))
                filters = None
            file_list = files_directories.files_in_directory(
                paths[0], sort_files=True, path_filter=filters,
                root_directory=root_directory
            )
            # date from directory name
            if dates is not None and type(dates) is not list:
                if dates.lower() == cfg.date_auto:
                    dates = dates_times.date_string_from_directory_name(
                        paths[0]
                    )
        # one file
        elif len(paths) == 1:
            b_count = raster_vector.get_number_bands(paths[0])
            if b_count > 1:
                multiband = b_count
            file_list = paths.copy()
        # single band rasters
        else:
            file_list = paths.copy()
        # wavelength
        try:
            if wavelengths is not None and len(wavelengths) == 1:
                for sat in cfg.satWlList:
                    if (wavelengths[0].lower() in sat.lower()
                            or wavelengths[0].lower().replace(' ', '')
                            in sat.lower().replace(' ', '')):
                        satellite = sat
                        wavelengths = None
                        break
        except Exception as err:
            cfg.logger.log.error(str(err))
            wavelengths = None
        if unit is None:
            unit = cfg.no_unit
        if satellite is not None and satellite != cfg.no_satellite:
            sat_wl, sat_unit, sat_bands = cfg.satellites[satellite]
        else:
            sat_wl = sat_unit = sat_bands = None
        # create band list
        bands_list = []
        date = None
        counter = 1
        for f in range(len(file_list)):
            # band names from raster
            if band_names is None:
                band_name = _raster_to_band_names(file_list[f])[0]
            else:
                band_name = band_names[f]
            # path
            path = file_list[f]
            cfg.logger.log.debug('band %s path: %s' % (str(counter), path))
            # date
            if dates is not None:
                if type(dates) is not list:
                    if dates.lower() == cfg.date_auto:
                        date = None
                    else:
                        date = dates
                else:
                    date = dates[f]
            else:
                date = None
            # get wavelength
            if wavelengths is None:
                if satellite is not None and satellite != cfg.no_satellite:
                    wl = sat_wl[f]
                    unit = sat_unit
                    # get band number from names
                    try:
                        # numbers in format 01, 02, ...
                        b = band_name.lower()[-2:]
                        wl = float(sat_wl[sat_bands.index(b)])
                    except Exception as err:
                        e = str(err)
                        try:
                            # numbers in format 1, 2, ...
                            b = band_name.lower()[-2:].lstrip('0')
                            wl = float(sat_wl[sat_bands.index(b)])
                        except Exception as err:
                            e = '{}; {}'.format(err, e)
                            # get values from list
                            try:
                                wl = float(sat_wl[f])
                            except Exception as err:
                                cfg.logger.log.error('%s: %s', (err, e))
                else:
                    wl = f + 1
                    unit = cfg.no_unit
            else:
                wl = wavelengths[f]
            # single band
            if multiband is None:
                # multiplicative factors
                if multiplicative_factors is None:
                    multiplicative_factors = [1] * len(file_list)
                # additive factors
                if additive_factors is None:
                    additive_factors = [0] * len(file_list)
                new_band = _create_table_of_bands(
                    path, band_number=counter, raster_band=1, name=band_name,
                    multiplicative_factor=multiplicative_factors[f],
                    additive_factor=additive_factors[f], date=date,
                    root_directory=root_directory, wavelength=wl,
                    wavelength_unit=unit
                )
                bands_list.append(new_band)
                counter += 1
            # multi band raster
            else:
                # multiplicative factors
                if multiplicative_factors is None:
                    multiplicative_factors = [1] * multiband
                # additive factors
                if additive_factors is None:
                    additive_factors = [0] * multiband
                # band names from raster
                band_names = _raster_to_band_names(file_list[0], multiband)
                for b in range(multiband):
                    new_band = _create_table_of_bands(
                        path, band_number=counter, raster_band=b + 1,
                        name=band_names[b],
                        multiplicative_factor=multiplicative_factors[b],
                        additive_factor=additive_factors[b], date=date,
                        root_directory=root_directory, wavelength=wl,
                        wavelength_unit=unit
                    )
                    bands_list.append(new_band)
                    counter += 1
        if len(bands_list) > 0:
            crs = bands_list[0]['crs'][0]
        else:
            crs = None
        cfg.logger.log.info('end; file list: %s' % file_list)
        return cls(
            bands_list=bands_list, name=name, date=date,
            bandset_uid=bandset_uid, crs=crs,
            box_coordinate_list=box_coordinate_list
        )

    def get_band_by_wavelength(
            self, wavelength: float, threshold: float = None,
            output_as_number=False
    ) -> np.array:
        """Gets band by wavelength.

        Gets band by nearest wavelength.

        Args:
            wavelength: value of wavelength.
            threshold: threshold for wavelength identification, identifying bands within wavelength +- threshold; if None, the nearest value is returned.
            output_as_number: if True returns the number of the band, if False returns the band table.

        Returns:
            returns band table.
            
        Examples:
            Get band by wavelength
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> bandset = BandSet()
                >>> # get band by nearest wavelength
                >>> band_x = bandset.get_band_by_wavelength(
                >>> wavelength=0.7, threshold=0.1
                    )
        """  # noqa: E501
        try:
            wl = float(wavelength)
        except Exception as err:
            cfg.logger.log.error(str(err))
            return None
        band_table = tm.find_nearest_value(
            self.bands, 'wavelength', wl, threshold
        )
        if not output_as_number and output_as_number is not None:
            band = band_table
        elif band_table is not None:
            band = band_table['band_number']
        else:
            band = None
        cfg.logger.log.debug(
            'wavelength %s; band: %s'
            % (str(wavelength), str(band))
        )
        return band

    def spectral_range_bands(self, output_as_number=True) -> list:
        """Gets bands from spectral range.

        Gets bands from spectral range blue, green, red, nir, swir_1, swir_2.

        Args:
            output_as_number: if True returns the number of the band, if False returns the band table.

        Returns:
            returns list of bands.
            
        Examples:
            Gets bands from spectral range
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> bandset = BandSet()
                >>> # spectral range bands
                >>> (blue_band, green_band, red_band, nir_band, swir_1_band,
                ... swir_2_band) = bandset.spectral_range_bands(
                ... output_band_order=False)
        """  # noqa: E501
        cfg.logger.log.debug('start')
        try:
            unit = self.get_wavelength_units()[0]
        except Exception as err:
            str(err)
            return [None, None, None, None, None, None]
        # scale unit
        if unit == cfg.wl_micro:
            m = 1
        elif unit == cfg.wl_nano:
            m = 1000
        else:
            m = 0
        # centers and threshold list blue, green, red, nir, swir_1, swir_2
        c_list = [[cfg.blue_center * m, cfg.blue_threshold * m],
                  [cfg.green_center * m, cfg.green_threshold * m],
                  [cfg.red_center * m, cfg.red_threshold * m],
                  [cfg.nir_center * m, cfg.nir_threshold * m],
                  [cfg.swir_1_center * m, cfg.swir_1_threshold * m],
                  [cfg.swir_2_center * m, cfg.swir_2_threshold * m]]
        # get bands
        bands = []
        for c in c_list:
            band = self.get_band_by_wavelength(
                wavelength=c[0], threshold=c[1],
                output_as_number=output_as_number
            )
            bands.append(band)
        cfg.logger.log.debug('end; bands: %s' % str(bands))
        return bands

    def add_new_band(
            self, path: str, band_number: int = None, raster_band=None,
            band_name=None, date=None, root_directory=None,
            multiplicative_factor=None, additive_factor=None, wavelength=None,
            unit=None
    ):
        """Adds new band to a BandSet.

        Adds a new band to a BandSet with the option to set the band number
        for the position in the BandSet order.

        Args:
            path: file path.
            band_number: sets the position of the band in BandSet order.
            raster_band: number of band for multiband rasters.
            band_name: name of raster used for band.
            date: single date (as YYYY-MM-DD).
            multiplicative_factor: multiplicative factor.
            additive_factor: additive factor.
            wavelength: center wavelength.
            unit: wavelength unit.
            root_directory: root directory for relative path.

        Examples:
            Add a new band
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> bandset = BandSet()
                >>> bandset.add_new_band(
                ... path=path, band_number=band_number, raster_band=raster_band,
                ... band_name=band_name, date=date, root_directory=root_directory,
                ... multiplicative_factor=multiplicative_factor,
                ... additive_factor=additive_factor, wavelength=wavelength, unit=unit
                ... )
        """  # noqa: E501
        if band_number is None:
            band_number = self.get_band_count() + 1
        if band_name is None:
            band_name = _raster_to_band_names(path)[0]
        if self.root_directory != root_directory:
            cfg.logger.log.warning('root_directory: %s' % root_directory)
            messages.warning('root_directory: %s' % root_directory)
        if multiplicative_factor is None:
            multiplicative_factor = 1
        if additive_factor is None:
            additive_factor = 0
        if wavelength is None:
            wavelength = band_number
            unit = cfg.no_unit
        if unit is None:
            unit = cfg.no_unit
        # move up bands above added band
        if band_number <= self.get_band_count():
            self.bands['band_number'][
                self.bands['band_number'] >= band_number] = \
                self.bands['band_number'][
                    self.bands['band_number'] >= band_number] + 1
        band = _create_table_of_bands(
            path, band_number=band_number, raster_band=raster_band,
            name=band_name, multiplicative_factor=multiplicative_factor,
            additive_factor=additive_factor, date=date,
            root_directory=root_directory, wavelength=wavelength,
            wavelength_unit=unit
        )
        self.bands = tm.create_bandset_table([self.bands, band])

    def sort_bands_by_wavelength(self):
        """Sorts band order by wavelength"""
        self.bands.sort(order='wavelength')
        order_list = list(range(1, self.bands.shape[0] + 1))
        self.bands['band_number'] = np.array(order_list)

    def find_values_in_list(
            self, attribute, value_list, output_attribute=None
    ) -> list:
        """Adds new band to a BandSet.

        Finds BandSet values in a list and return a band attribute or band numbers.

        Args:
            attribute: attribute name for identification.
            value_list: attribute value list for indentification.
            output_attribute: attribute name of desired output; if None returns the band number.

        Returns:
            returns list of band attributes.
            
        Examples:
            Find values from wavelength attribute
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> bandset = BandSet()
                >>> bandset_values = bandset.find_values_in_list(
                ... attribute='wavelength', value_list=[0.6, 0.7],
                ... output_attribute='path'
                ... )
        """  # noqa: E501
        band_list = []
        for value in value_list:
            if output_attribute is None:
                bandset_attributes = self.bands['band_number'][
                    self.bands[attribute] == value]
            else:
                bandset_attributes = self.bands[output_attribute][
                    self.bands[attribute] == value]
            if len(bandset_attributes) > 0:
                for attr in bandset_attributes:
                    band_list.append(attr)
        return band_list

    @staticmethod
    def generate_uid() -> str:
        """Generates unique ID for BandSet

        Returns:
            returns table of bands.
        """
        times = dates_times.get_time_string()
        r = str(random.randint(0, 1000))
        uid = '{}_{}'.format(times, r)
        return uid


def _create_table_of_bands(
        path, band_number=None, raster_band=None, name=None,
        multiplicative_factor=None, additive_factor=None, date=None,
        root_directory=None, wavelength=None, wavelength_unit=None
) -> np.array:
    """Creates table of bands

    This method creates a BandSet defined by input files.
    Raster properties are derived from files to populate 
    the table of bands.

    Args:
        name: optional BandSet name string
        path: file path
        band_number: sets the position of the band in BandSet order.
        raster_band: number of band for multiband rasters.
        name: name of raster used for band.
        date: single date (as YYYY-MM-DD).
        multiplicative_factor: multiplicative factor.
        additive_factor: additive factor.
        wavelength: center wavelength.
        wavelength_unit: wavelength unit.
        root_directory: root directory for relative path.
                    
    Returns:
        returns table of bands.
    """  # noqa: E501
    cfg.logger.log.debug('start')
    try:
        absolute_path = files_directories.relative_to_absolute_path(
            path, root_directory
        )
    except Exception as err:
        cfg.logger.log.error(str(err))
        if files_directories.is_file(path):
            absolute_path = path
        else:
            cfg.logger.log.error('path: %s' % path)
            messages.error('path: %s' % path)
            return None
    (gt, crs, crs_unit, xy_count, nd, number_of_bands, block_size,
     scale_offset, data_type) = raster_vector.raster_info(absolute_path)
    x_size = abs(gt[1])
    y_size = abs(gt[5])
    top = gt[3]
    left = gt[0]
    bottom = gt[3] + gt[5] * xy_count[1] + gt[4] * xy_count[0]
    right = gt[0] + gt[1] * xy_count[0] + gt[2] * xy_count[1]
    table = tm.create_band_table(
        band_number=band_number, raster_band=raster_band, path=path,
        absolute_path=absolute_path, name=name, wavelength=wavelength,
        wavelength_unit=wavelength_unit, additive_factor=additive_factor,
        multiplicative_factor=multiplicative_factor, date=date, x_size=x_size,
        y_size=y_size, top=top, left=left, bottom=bottom, right=right,
        x_count=xy_count[0], y_count=xy_count[1], nodata=nd,
        data_type=data_type, crs=crs, number_of_bands=number_of_bands,
        x_block_size=block_size[0], y_block_size=block_size[1],
        scale=scale_offset[0], offset=scale_offset[1]
    )
    cfg.logger.log.debug('end')
    return table


def _raster_to_band_names(path: str, raster_band=1) -> list:
    """Creates band name list from raster name

    Creates band name list from raster name

    Args:
        path: file path.
        raster_band: raster band number.

    Returns:
        returns list of band names.
    """
    cfg.logger.log.debug('start')
    raster_name = files_directories.file_name(path, False)
    name_list = []
    if raster_band == 1:
        name_list.append(raster_name)
    else:
        for i in range(raster_band):
            name_list.append(
                '%s%s%s' % (raster_name, cfg.band_name_suf, str(i))
            )
    cfg.logger.log.debug('end; name_list: %s' % str(name_list))
    return name_list
