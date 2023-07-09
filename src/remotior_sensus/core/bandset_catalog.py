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

from copy import deepcopy
from xml.etree import cElementTree
from xml.dom import minidom
import random
from typing import Union, Optional

import numpy as np

from remotior_sensus.core import configurations as cfg, table_manager as tm
from remotior_sensus.util import (
    dates_times, files_directories, raster_vector, read_write_files)

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


class BandSet(object):
    """Manages band sets.

    This module allows for managing bands in a BandSet.

    Attributes:
        bands: BandSet of band tables.
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
            root_directory=None, crs=None, box_coordinate_list=None,
            catalog=None
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
            catalog: BandSet Catalog

        Examples:
            Initialize a BandSet
                >>> BandSet.create(name=name,box_coordinate_list=box_coordinate_list)
        """  # noqa: E501
        if bands_list is None:
            # create an empty band table ('empty' to cause exception)
            self.bands = tm.create_band_table(band_number='empty')
        else:
            self.bands = tm.create_bandset_table(bands_list)
        # unique ID
        self.uid = bandset_uid
        if date is None:
            date = 'NaT'
        self.date = np.array(date, dtype='datetime64[D]')
        self.root_directory = root_directory
        self.crs = crs
        self.catalog = catalog
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
    def uid(self):
        """str: Unique BandSet ID."""
        return self._uid

    @uid.setter
    def uid(self, uid):
        self._uid = uid

    @property
    def name(self):
        """str: Optional BandSet name."""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        if self.catalog is not None:
            if self.catalog.bandsets_table is not None:
                self.catalog.bandsets_table['bandset_name'][
                    self.catalog.bandsets_table['uid'] == self.uid] = name

    @property
    def crs(self):
        """str: crs."""
        return self._crs

    @crs.setter
    def crs(self, crs):
        self._crs = crs
        if self.catalog is not None:
            if self.catalog.bandsets_table is not None:
                self.catalog.bandsets_table['crs'][
                    self.catalog.bandsets_table['uid'] == self.uid] = crs

    @property
    def date(self):
        """Optional date."""
        return self._date

    @date.setter
    def date(self, date):
        if date is None:
            date = 'NaT'
        self._date = np.array(date, dtype='datetime64[D]')
        if self.catalog is not None:
            if self.catalog.bandsets_table is not None:
                self.catalog.bandsets_table['date'][
                    self.catalog.bandsets_table['uid'] == self.uid] = date

    @property
    def root_directory(self):
        """str: Optional BandSet root directory for relative path."""
        return self._root_directory

    @root_directory.setter
    def root_directory(self, root_directory):
        self._root_directory = root_directory
        if self.catalog is not None:
            if self.catalog.bandsets_table is not None:
                self.catalog.bandsets_table['root_directory'][
                    self.catalog.bandsets_table[
                        'uid'] == self.uid] = root_directory

    @property
    def box_coordinate_list(self):
        """
        list: Optional BandSet box coordinate list for virtual subset.
        """  # noqa: E501
        return self._box_coordinate_list

    @box_coordinate_list.setter
    def box_coordinate_list(self, box_coordinate_list):
        self._box_coordinate_list = box_coordinate_list
        if box_coordinate_list is not None:
            box_coordinate_left = box_coordinate_list[0]
            box_coordinate_top = box_coordinate_list[1]
            box_coordinate_right = box_coordinate_list[2]
            box_coordinate_bottom = box_coordinate_list[3]
        else:
            box_coordinate_left = box_coordinate_top = box_coordinate_right \
                = box_coordinate_bottom = None
        if self.catalog is not None:
            if self.catalog.bandsets_table is not None:
                self.catalog.bandsets_table['box_coordinate_left'][
                    self.catalog.bandsets_table[
                        'uid'] == self.uid] = box_coordinate_left
                self.catalog.bandsets_table['box_coordinate_top'][
                    self.catalog.bandsets_table[
                        'uid'] == self.uid] = box_coordinate_top
                self.catalog.bandsets_table['box_coordinate_right'][
                    self.catalog.bandsets_table[
                        'uid'] == self.uid] = box_coordinate_right
                self.catalog.bandsets_table['box_coordinate_bottom'][
                    self.catalog.bandsets_table[
                        'uid'] == self.uid] = box_coordinate_bottom

    def get_band_count(self) -> int:
        """Gets the count of bands."""
        if self.bands is None:
            return 0
        else:
            return self.bands.shape[0]

    def get_paths(self) -> list:
        """Gets the list of bands."""
        if self.bands is not None:
            return self.bands['path'].tolist()
        else:
            return []

    def get_absolute_paths(self) -> list:
        """Gets the list of absolute paths."""
        absolute_paths = []
        if self.bands is not None:
            paths = self.bands['path'].tolist()
            for path in paths:
                absolute_path = files_directories.relative_to_absolute_path(
                    path, self.root_directory
                )
                absolute_paths.append(absolute_path)
        cfg.logger.log.debug('absolute_paths: %s' % str(absolute_paths))
        return absolute_paths

    def get_path(self, band_number) -> Union[str, None]:
        """Gets the absolute path of band."""
        if self.bands is not None:
            return self.bands[self.bands['band_number'] == band_number]['path']
        else:
            return None

    def get_absolute_path(self, band_number) -> str:
        """Gets the absolute path of band."""
        absolute_path = None
        if self.bands is not None:
            path = self.bands[self.bands['band_number'] == band_number]['path']
            if path is not None:
                if len(path) > 0:
                    absolute_path = (
                        files_directories.relative_to_absolute_path(
                            path[0], self.root_directory
                        )
                    )
        cfg.logger.log.debug('absolute_path: %s%s'
                             % (str(absolute_path), str(band_number)))
        return absolute_path

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

    def export_as_xml(self, output_path=None):
        """Exports a BandSet as xml.

        Exports a BandSet bands and attributes.

        Examples:
            Export a BandSet.
                >>> bandset = BandSet()
                >>> # reset
                >>> bandset.export_as_xml()
        """  # noqa: E501
        root = cElementTree.Element('bandset')
        root.set('version', str(cfg.version))
        root.set('name', str(self.name))
        root.set('date', str(self.date))
        root.set('root_directory', str(self.root_directory))
        root.set('crs', str(self.crs))
        if self.box_coordinate_list is not None:
            root.set('box_coordinate_left', str(self.box_coordinate_list[0]))
            root.set('box_coordinate_top', str(self.box_coordinate_list[1]))
            root.set('box_coordinate_right', str(self.box_coordinate_list[2]))
            root.set('box_coordinate_bottom', str(self.box_coordinate_list[3]))
        if self.bands is not None:
            for band in self.bands:
                band_element = cElementTree.SubElement(root, 'band')
                band_element.set('band_number', str(band['band_number']))
                for attribute in self.bands.dtype.names:
                    if attribute != 'band_number':
                        element = cElementTree.SubElement(band_element,
                                                          attribute)
                        element.text = str(band[attribute])
        cfg.logger.log.debug('export bandset')
        if output_path is None:
            return cElementTree.tostring(root)
        else:
            # save to file
            pretty_xml = minidom.parseString(
                cElementTree.tostring(root)).toprettyxml()
            read_write_files.write_file(pretty_xml, output_path)
            return output_path

    def print(self):
        """Prints a BandSet.

        Prints a BandSet bands and attributes.

        Examples:
            Print a BandSet.
                >>> bandset = BandSet()
                >>> # reset
                >>> bandset.print()
        """  # noqa: E501
        text = []
        nl = cfg.new_line
        sep = '│ '
        text.append('name: %s %s' % (str(self.name), nl))
        text.append('date: %s %s' % (str(self.date), nl))
        text.append('root directory: %s %s'
                    % (str(self.root_directory), nl))
        text.append('crs: %s %s' % (str(self.crs), nl))
        if self.box_coordinate_list is not None:
            text.append('box coordinate left %s %s %s'
                        % (sep, str(self.box_coordinate_list[0]), nl))
            text.append('box coordinate top %s %s %s'
                        % (sep, str(self.box_coordinate_list[1]), nl))
            text.append('box coordinate right %s %s %s'
                        % (sep, str(self.box_coordinate_list[2]), nl))
            text.append('box coordinate bottom %s %s %s'
                        % (sep, str(self.box_coordinate_list[3]), nl))
        if self.bands is not None:
            max_widths = {}
            for attribute in self.bands.dtype.names:
                max_widths[attribute] = len(attribute)
                for band in self.bands:
                    max_widths[attribute] = max(
                        max_widths[attribute], len(str(band[attribute])))
            attributes = []
            for band in self.bands:
                attributes.append('│ ')
                for attribute in self.bands.dtype.names:
                    attributes.append(
                        '%s %s'
                        % (str(band[attribute]).ljust(
                            max_widths[attribute]), sep)
                    )
                attributes.append(nl)
            # field names
            names = ['│ ']
            first_line = ['┌']
            lines = ['├']
            last_line = ['└']
            for attribute in self.bands.dtype.names:
                names.append('%s %s'
                             % (attribute.ljust(max_widths[attribute]), sep))
                first_line.append('%s%s'
                                  % ('─' * (max_widths[attribute] + 2), '┬'))
                lines.append('%s%s'
                             % ('─' * (max_widths[attribute] + 2), '┼'))
                last_line.append('%s%s'
                                 % ('─' * (max_widths[attribute] + 2), '┴'))
            names.append(nl)
            first_line[-1] = first_line[-1][:-1]
            first_line.append('%s %s' % ('┐', nl))
            lines[-1] = lines[-1][:-1]
            lines.append('%s %s' % ('┤', nl))
            last_line[-1] = last_line[-1][:-1]
            last_line.append('%s %s' % ('┘', nl))
            text.append(''.join(first_line))
            text.append(''.join(names))
            text.append(''.join(lines))
            text.append(''.join(attributes))
            text.append(''.join(last_line))
        cfg.logger.log.debug('print bandset')
        # print output
        print(''.join(text))

    def import_as_xml(self, xml_path):
        """Imports a BandSet as xml.

        Imports a BandSet bands and attributes.

        Examples:
            Import a BandSet
                >>> bandset = BandSet()
                >>> # reset
                >>> bandset.import_as_xml('xml_path')
        """  # noqa: E501

        cfg.logger.log.debug('import bandset: %s' % xml_path)
        tree = cElementTree.parse(xml_path)
        root = tree.getroot()
        version = root.get('version')
        if version is None:
            cfg.logger.log.error('failed importing bandset: %s' % xml_path)
            cfg.messages.error('failed importing bandset: %s' % xml_path)
        else:
            name = root.get('name')
            date = root.get('date')
            root_directory = root.get('root_directory')
            crs = root.get('crs')
            box_coordinate_left = root.get('box_coordinate_left')
            box_coordinate_top = root.get('box_coordinate_top')
            box_coordinate_right = root.get('box_coordinate_right')
            box_coordinate_bottom = root.get('box_coordinate_bottom')
            if box_coordinate_left is None:
                box_coordinate_list = None
            else:
                box_coordinate_list = [
                    float(box_coordinate_left), float(box_coordinate_top),
                    float(box_coordinate_right), float(box_coordinate_bottom)
                ]
            bands_list = []
            for child in root:
                band_number = child.get('band_number')
                attributes = {}
                for attribute in self.bands.dtype.names:
                    if attribute != 'band_number':
                        element = child.find(attribute).text
                        if element == 'None':
                            element = None
                        attributes[attribute] = element
                new_band = tm.create_band_table(
                    band_number=band_number,
                    raster_band=attributes['raster_band'],
                    path=attributes['path'],
                    name=attributes['name'],
                    wavelength=attributes['wavelength'],
                    wavelength_unit=attributes['wavelength_unit'],
                    additive_factor=attributes['additive_factor'],
                    multiplicative_factor=attributes['multiplicative_factor'],
                    date=attributes['date'], x_size=attributes['x_size'],
                    y_size=attributes['y_size'], top=attributes['top'],
                    left=attributes['left'], bottom=attributes['bottom'],
                    right=attributes['right'], x_count=attributes['x_count'],
                    y_count=attributes['y_count'], nodata=attributes['nodata'],
                    data_type=attributes['data_type'], crs=attributes['crs'],
                    number_of_bands=attributes['number_of_bands'],
                    x_block_size=attributes['x_block_size'],
                    y_block_size=attributes['y_block_size'],
                    root_directory=root_directory,
                    scale=attributes['scale'], offset=attributes['offset']
                )
                bands_list.append(new_band)
            self.bands = tm.create_bandset_table(bands_list)
            if date is None:
                date = 'NaT'
            self.date = np.array(date, dtype='datetime64[D]')
            if root_directory == 'None':
                root_directory = None
            self.root_directory = root_directory
            if crs == 'None':
                crs = None
            self.crs = crs
            if name == 'None':
                name = None
            self.name = name
            self.box_coordinate_list = box_coordinate_list
            cfg.logger.log.debug('import bandset: %s' % xml_path)

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
            name: Optional[str] = None, catalog=None
    ):
        """Creates a BandSet.

        This method creates a BandSet defined by input files.
        Raster properties are derived from files to populate 
        the table of bands.

        Args:
            catalog: BandSet Catalog object.
            name: BandSet name.
            paths: list of file paths or a string of directory path; also, a list of directory path 
                and name filter is accepted.
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

           Passing a directory with file name filter and the wavelenght from satellite name.
                >>> bandset = rs.bandset.create(['directory_path', 'tif'], wavelengths='Sentinel-2')
        """  # noqa: E501
        cfg.logger.log.info('start')
        bandset_uid = cls.generate_uid()
        cls.catalog = catalog
        satellite = None
        multiband = None
        # directory
        if paths is None:
            cfg.logger.log.info('end')
            return cls(bandset_uid=bandset_uid, name=name, catalog=catalog)
        elif len(paths) <= 2 and files_directories.is_directory(
                files_directories.relative_to_absolute_path(
                    paths[0], root_directory
                )
        ):
            filters = None
            try:
                if len(paths) == 2:
                    filters = paths[1].strip("'").strip()
            except Exception as err:
                cfg.logger.log.error(str(err))
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
        elif type(paths) is str and files_directories.is_directory(
                files_directories.relative_to_absolute_path(
                    paths, root_directory
                )
        ):
            file_list = files_directories.files_in_directory(
                paths, sort_files=True, root_directory=root_directory
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
            bands_list = None
        cfg.logger.log.info('end; file list: %s' % file_list)
        return cls(
            bands_list=bands_list, name=name, date=date,
            bandset_uid=bandset_uid, crs=crs,
            box_coordinate_list=box_coordinate_list,
            catalog=catalog, root_directory=root_directory
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
                ... output_as_number=False)
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
            cfg.messages.warning('root_directory: %s' % root_directory)
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
        if band is not None:
            if self.bands is None:
                self.bands = tm.create_bandset_table([band])
            else:
                self.bands = tm.create_bandset_table([self.bands, band])

    def sort_bands_by_wavelength(self):
        """Sorts band order by wavelength"""
        self.bands.sort(order='wavelength')
        order_list = list(range(1, self.bands.shape[0] + 1))
        self.bands['band_number'] = np.array(order_list)

    def sort_bands_by_name(self, keep_wavelength_order=True):
        """Sorts band order by name"""
        bandset_wavelength = deepcopy(self.bands['wavelength'])
        self.bands.sort(order='name')
        order_list = list(range(1, self.bands.shape[0] + 1))
        self.bands['band_number'] = np.array(order_list)
        if keep_wavelength_order:
            self.bands['wavelength'] = bandset_wavelength

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

# Tools #######################################################################

    def execute(self, function, *args, **kwargs):
        """Executes a function.

        Executes a functions directly from the BandSet, passing the argument input_bands.
        The arguments are related to the function.

        Args:
            function: the function to be executed.

        Returns:
            The output of the function.

        Examples:
            Given that a session was previously started
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> bandset = rs.bandset.create(
                ...     ['file1.tif', 'file2.tif'], wavelengths=['Sentinel-2'],
                ... )

            Calculation of sum of the first two bands 
                >>> output_object = bandset.execute(
                ...     rs.band_calc, output_path='output.tif', expression_string='"b1" + "b2"'
                ... )

            Calculation of NDVI
                >>> output_object = bandset.execute(
                ...     rs.band_calc, output_path='output.tif', 
                ...     expression_string='("#NIR#" - "#RED#") / ("#NIR#" + "#RED#")'
                ... )

            Calculation of band combination
                >>> output_object = bandset.execute(rs.band_combination, output_path='output.tif')
        """  # noqa: E501
        kwargs['input_bands'] = self
        try:
            return function(*args, **kwargs)
        except Exception as err:
            cfg.logger.log.error(str(err))
            cfg.messages.error(str(err))
            return None

    def calc(self, *args, **kwargs):
        """Executes a calculation.

        Executes a calculation directly from the BandSet, passing the argument input_bands.
        The arguments are related to the function.
        Bands in the BandSet can be reffered in the expressions such as "b1" or "b2";
        also band alias such as "#RED#" and expression alias such as #NDVI# can be used.

        Args:
            kwargs: See :func:`~remotior_sensus.tools.band_calc`.

        Returns:
            The output of the function.

        Examples:
            Given that a session was previously started
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> bandset = rs.bandset.create(
                ...     ['file1.tif', 'file2.tif'], wavelengths=['Sentinel-2'],
                ... )

            Calculation of sum of the first two bands and saving the output in temporary directory
                >>> output_object = bandset.calc('"b1" + "b2"')

            Calculation of NDVI
                >>> output_object = bandset.calc(output_path='output.tif', 
                ...     expression_string='("#NIR#" - "#RED#") / ("#NIR#" + "#RED#")'
                ... )
        """  # noqa: E501
        kwargs['input_bands'] = self
        return cfg.band_calc(*args, **kwargs)

    def classification(self, *args, **kwargs):
        """Executes a classification.

        Executes a classification directly from the BandSet, passing the argument input_bands.
        The arguments are related to the function.

        Args:
            kwargs: See :func:`~remotior_sensus.tools.band_classification`.

        Returns:
            The output of the function.
        """  # noqa: E501
        kwargs['input_bands'] = self
        return cfg.band_classification(*args, **kwargs)

    def combination(self, *args, **kwargs):
        """Band combination.

        Combines classifications directly from the BandSet, in order to get a raster where each value 
        corresponds to a combination of class values.
        The arguments are related to the function.

        Args:
            kwargs: See :func:`~remotior_sensus.tools.band_combination`.

        Returns:
            The output of the function.

        Examples:
            Given that a session was previously started
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> bandset = rs.bandset.create(['file1.tif', 'file2.tif'])

            Combination using two rasters having paths path_1 and path_2
                >>> combination = bandset.combination(output_path='output_path')
        """  # noqa: E501
        kwargs['input_bands'] = self
        return cfg.band_combination(*args, **kwargs)

    def dilation(self, *args, **kwargs):
        """Band dilation.

        Band dilation directly from the BandSet, passing the argument input_bands.
        The arguments are related to the function.

        Args:
            kwargs: See :func:`~remotior_sensus.tools.band_dilation`.

        Returns:
            The output of the function.

        Examples:
            Given that a session was previously started
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> bandset = rs.bandset.create(['file1.tif', 'file2.tif'])

            Dilation
                >>> dilation = bandset.dilation(output_path='directory_path',
                ...     value_list=[1, 2], size=5)
        """  # noqa: E501
        kwargs['input_bands'] = self
        return cfg.band_dilation(*args, **kwargs)

    def erosion(self, *args, **kwargs):
        """Band erosion.

        Band erosion directly from the BandSet, passing the argument input_bands.
        The arguments are related to the function.

        Args:
            kwargs: See :func:`~remotior_sensus.tools.band_erosion`.

        Returns:
            The output of the function.

        Examples:
            Given that a session was previously started
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> bandset = rs.bandset.create(['file1.tif', 'file2.tif'])

            Erosion
                >>> erosion = bandset.erosion(output_path='directory_path',
                ...     value_list=[1, 2], size=1)
        """  # noqa: E501
        kwargs['input_bands'] = self
        return cfg.band_erosion(*args, **kwargs)

    def pca(self, *args, **kwargs):
        """Band PCA.

        Band PCA directly from the BandSet, passing the argument input_bands.
        The arguments are related to the function.

        Args:
            kwargs: See :func:`~remotior_sensus.tools.band_pca`.

        Returns:
            The output of the function.

        Examples:
            Given that a session was previously started
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> bandset = rs.bandset.create(['file1.tif', 'file2.tif'])

            PCA
                >>> pca = bandset.pca(output_path='directory_path')
        """  # noqa: E501
        kwargs['input_bands'] = self
        return cfg.band_pca(*args, **kwargs)

    def sieve(self, *args, **kwargs):
        """Band sieve.

        Band sieve directly from the BandSet, passing the argument input_bands.
        The arguments are related to the function.

        Args:
            kwargs: See :func:`~remotior_sensus.tools.band_sieve`.

        Returns:
            The output of the function.

        Examples:
            Given that a session was previously started
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> bandset = rs.bandset.create(['file1.tif', 'file2.tif'])

            Sieve
                >>> sieve = bandset.sieve(output_path='directory_path', size=3)
        """  # noqa: E501
        kwargs['input_bands'] = self
        return cfg.band_sieve(*args, **kwargs)

    def neighbor_pixels(self, *args, **kwargs):
        """Band neighbor pixels.

        Band neighbor pixels directly from the BandSet, passing the argument input_bands.
        The arguments are related to the function.

        Args:
            kwargs: See :func:`~remotior_sensus.tools.band_neighbor_pixels`.

        Returns:
            The output of the function.

        Examples:
            Given that a session was previously started
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> bandset = rs.bandset.create(['file1.tif', 'file2.tif'])

            Neighbor pixels
                >>> neighbor = bandset.neighbor_pixels(output_path='directory_path',
                ... size=10, circular_structure=True, stat_name='Sum')
        """  # noqa: E501
        kwargs['input_bands'] = self
        return cfg.band_neighbor_pixels(*args, **kwargs)

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
            cfg.messages.error('path: %s' % path)
            return None
    info = raster_vector.raster_info(absolute_path)
    if info is False:
        cfg.logger.log.error('path: %s' % path)
        cfg.messages.error('path: %s' % path)
        return None
    else:
        (gt, crs, crs_unit, xy_count, nd, number_of_bands, block_size,
         scale_offset, data_type) = info
    x_size = abs(gt[1])
    y_size = abs(gt[5])
    top = gt[3]
    left = gt[0]
    bottom = gt[3] + gt[5] * xy_count[1] + gt[4] * xy_count[0]
    right = gt[0] + gt[1] * xy_count[0] + gt[2] * xy_count[1]
    table = tm.create_band_table(
        band_number=band_number, raster_band=raster_band, path=path,
        root_directory=root_directory, name=name, wavelength=wavelength,
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

    def get_bandsets_by_date(
            self, date_list: list, output_number: Optional[bool] = False
    ) -> list:
        """Get BandSets by date.

        This function gets the BandSets by date.

        Args:
            date_list: list of date strings, or single date string (format YYYY-MM-DD);
                it can also include ranges using > or <, multiple conditions including &
            output_number: if True then the output is the BandSet number,
                if False then the output is the BandSet

        Returns:
            The BandSet identified by the name.

        Examples:
            Get the number of the BandSet by date.
                >>> catalog = BandSetCatalog()
                >>> bandset_number = catalog.get_bandsets_by_date(date_list=['2020-01-01'], output_number=True)
                >>> print(bandset_number)
                [1]
                
            Get the number of the BandSet by range of dates.
                >>> catalog = BandSetCatalog()
                >>> bandset_number = catalog.get_bandsets_by_date(date_list=['2020-01-01', '>=2021-01-01 & <=2022-01-02'], output_number=True)
                >>> print(bandset_number)
                [1, 3]
        """  # noqa: E501
        cfg.logger.log.debug(
            'date_list: %s; output_number: %s' % (
                str(date_list), output_number)
        )
        if output_number:
            field = 'bandset_number'
        else:
            field = 'uid'
        uids = []
        b_uids = []
        for d in date_list:
            date_l = date_le = date_g = date_ge = date_eq = None
            try:
                if '&' in d:
                    date_ranges = d.split('&')
                    date_l = date_le = date_g = date_ge = None
                    for r in date_ranges:
                        if '<=' in r:
                            date_le = np.datetime64(
                                r.replace(' ', '').replace('<=', '')
                            )
                        elif '<' in r:
                            date_l = np.datetime64(
                                r.replace(' ', '').replace('<', '')
                            )
                        elif '>=' in r:
                            date_ge = np.datetime64(
                                r.replace(' ', '').replace('>=', '')
                            )
                        elif '>' in r:
                            date_g = np.datetime64(
                                r.replace(' ', '').replace('>', '')
                            )
                elif '<=' in d:
                    date_le = np.datetime64(
                        d.replace(' ', '').replace('<=', '')
                    )
                elif '<' in d:
                    date_l = np.datetime64(
                        d.replace(' ', '').replace('<', '')
                    )
                elif '>=' in d:
                    date_ge = np.datetime64(
                        d.replace(' ', '').replace('>=', '')
                    )
                elif '>' in d:
                    date_g = np.datetime64(
                        d.replace(' ', '').replace('>', '')
                    )
                else:
                    date_eq = np.datetime64(
                        d.replace(' ', '')
                    )
            except Exception as err:
                cfg.logger.log.error(str(err))
                cfg.messages.error(str(err))
            # get uids
            if date_eq is not None:
                uids = self.bandsets_table[
                    self.bandsets_table['date'] == date_eq][field]
            elif date_g is not None:
                if date_l is not None:
                    uids = self.bandsets_table[
                        (self.bandsets_table['date'] > date_g)
                        & (self.bandsets_table['date'] < date_l)][field]
                elif date_le is not None:
                    uids = self.bandsets_table[
                        (self.bandsets_table['date'] > date_g)
                        & (self.bandsets_table['date'] <= date_le)][field]
                else:
                    uids = self.bandsets_table[
                        self.bandsets_table['date'] > date_g][field]
            elif date_ge is not None:
                if date_l is not None:
                    uids = self.bandsets_table[
                        (self.bandsets_table['date'] >= date_ge)
                        & (self.bandsets_table['date'] < date_l)][field]
                elif date_le is not None:
                    uids = self.bandsets_table[
                        (self.bandsets_table['date'] >= date_ge)
                        & (self.bandsets_table['date'] <= date_le)][field]
                else:
                    uids = self.bandsets_table[
                        self.bandsets_table['date'] >= date_ge][field]
            elif date_l is not None:
                uids = self.bandsets_table[
                    self.bandsets_table['date'] < date_l][field]
            elif date_le is not None:
                uids = self.bandsets_table[
                    self.bandsets_table['date'] <= date_le][field]
            b_uids.extend(uids)
        if output_number:
            bandset_list = list(set(b_uids))
        else:
            bandset_list = []
            for b in set(b_uids):
                bandset_list.append(self.bandsets[b])
        return bandset_list

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
                >>> date = catalog.get_bandset_catalog_attributes(bandset_number=1, attribute='date')
                >>> print(date)
                2000-12-31

            Get the list of attributes of the BandSet 1.
                >>> catalog = BandSetCatalog()
                >>> attributes = catalog.get_bandset_catalog_attributes(bandset_number=1)
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
    ) -> Union[None, list, np.recarray]:
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
            paths: list of file paths or a string of directory path; also, a list of directory path 
                and name filter is accepted.
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
            Create a first BandSet from a file list with files inside a data directory, setting root_directory, defining the BandSet date, and the wavelenght from satellite name.
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
                
           Passing a directory with file name filter and the wavelenght from satellite name.
                >>> bandset = catalog.create_bandset(['directory_path', 'tif'], wavelengths='Sentinel-2')
        """  # noqa: E501
        cfg.logger.log.debug('start')
        bst = BandSet.create(
            paths, band_names=band_names, wavelengths=wavelengths, unit=unit,
            multiplicative_factors=multiplicative_factors,
            additive_factors=additive_factors, dates=date,
            root_directory=root_directory, name=bandset_name,
            box_coordinate_list=box_coordinate_list, catalog=self
        )
        self.add_bandset(
            bandset=bst, bandset_number=bandset_number, insert=insert,
            keep_uid=True
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
            # set current catalog
            for b in self.bandsets:
                self.bandsets[b].catalog = self
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
                (bandset.bands['band_number'] >= band_number_output) & (
                        bandset.bands['band_number'] < band_number_input)] = \
                bandset.bands['band_number'][
                    (bandset.bands['band_number'] >= band_number_output) & (
                            bandset.bands[
                                'band_number'] < band_number_input)] + 1
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
        if band_number <= bandset.get_band_count():
            bandset.bands['band_number'][
                bandset.bands['band_number'] >= band_number] = \
                bandset.bands['band_number'][
                    bandset.bands['band_number'] >= band_number] - 1
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

    def sort_bands_by_name(self, bandset_number: Optional[int] = None,
                           keep_wavelength_order: Optional[bool] = True):
        """Sorts bands by name.

         This function numerically sorts bands in a BandSet by name.

         Args:
            bandset_number: number of BandSet; if None, current BandSet is used.
            keep_wavelength_order: if True, keep wavelength_order.

         Examples:
            Sort bands in BandSet 1.
                >>> catalog = BandSetCatalog()
                >>> catalog.sort_bands_by_name(bandset_number=1)
         """  # noqa: E501
        cfg.logger.log.debug('bandset_number: %s' % bandset_number)
        if bandset_number is None:
            bandset_number = self.current_bandset
        self.get_bandset(bandset_number).sort_bands_by_name(
            keep_wavelength_order=keep_wavelength_order)

    def add_bandset(
            self, bandset: BandSet, bandset_number: Optional[int] = None,
            insert=False, keep_uid=False
    ):
        """Adds a BandSet to Catalog.

         This function adds a previously created BandSet to BandSet Catalog.

         Args:
            bandset: the BandSet to be added.
            bandset_number: number of BandSet; if None, current BandSet is used.
            insert: if True insert the BandSet at bandset_number (other BandSets are moved), 
                if False replace the BandSet number.
            keep_uid: if True, keeps uid from the original bandset to the added one. 
                
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
                cfg.messages.error(str(err))
        # add BandSet to dictionary by uid
        if keep_uid is False:
            bandset_copy.uid = bandset_copy.generate_uid()
        # replace catalog with current one
        bandset_copy.catalog = self
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

    def sort_bandsets_by_date(self):
        """Sort BandSets by BandSet date.

         This function sorts BandSets in the BandSet Catalog based on the date.

         Examples:
            Sort BandSets.
                >>> catalog = BandSetCatalog()
                >>> catalog.sort_bandsets_by_date()
         """  # noqa: E501
        cfg.logger.log.debug('sort_bandsets_by_date')
        bandset_numbers = deepcopy(self.bandsets_table['bandset_number'])
        # sort
        self.bandsets_table.sort(order='date')
        self.bandsets_table['bandset_number'] = bandset_numbers
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
            cfg.messages.warning('bandset_number: %s' % str(bandset_number))
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
        bandset = BandSet.create(catalog=self)
        bandset_number = 1
        # add BandSet to dictionary by uid
        self.bandsets[bandset.uid] = bandset
        box_coordinate_left = box_coordinate_top = box_coordinate_right = None
        box_coordinate_bottom = None
        self.bandsets_table = tm.create_bandset_catalog_table(
            bandset_number=bandset_number,
            root_directory=bandset.root_directory, date=bandset.date,
            bandset_uid=bandset.uid, bandset_name=bandset.name,
            previous_catalog=self.bandsets_table,
            box_coordinate_left=box_coordinate_left,
            box_coordinate_top=box_coordinate_top,
            box_coordinate_right=box_coordinate_right,
            box_coordinate_bottom=box_coordinate_bottom
        )
        cfg.logger.log.debug('empty bandset')

    def clear_bandset(self, bandset_number=None):
        """Function to clear a BandSet."""
        if bandset_number is None:
            bandset_number = self.current_bandset
        bandset = self.get_bandset(bandset_number)
        bandset.reset()
        cfg.logger.log.debug('clear bandset')

    def print_bandset(self, bandset_number=None):
        """Function to print a BandSet bands and attributes."""
        cfg.logger.log.debug('print bandset')
        if bandset_number is None:
            bandset_number = self.current_bandset
        bandset = self.get_bandset(bandset_number)
        # print bandset
        bandset.print()

    def export_bandset_as_xml(self, bandset_number, output_path=None):
        """Function to export a BandSet as xml."""
        cfg.logger.log.debug('export bandset as xml')
        bandset = self.get_bandset(bandset_number)
        xml = bandset.export_as_xml()
        pretty_xml = minidom.parseString(xml).toprettyxml()
        if output_path is None:
            return pretty_xml
        else:
            # save to file
            read_write_files.write_file(pretty_xml, output_path)
            return output_path

    def import_bandset_from_xml(self, bandset_number, xml_path):
        """Function to import a BandSet from xml."""
        cfg.logger.log.debug('import bandset from xml')
        bandset = self.get_bandset(bandset_number)
        bandset.import_as_xml(xml_path)

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

    def get_bandsets_by_list(
            self, bandset_list: Optional[list] = None,
            output_number: Optional[bool] = False
    ) -> list:
        """Gets BandSets by list.

        This function gets all the BandSets in the Catalog or filtered 
        using a list of BandSet numbers.

        Args:
            bandset_list: list of integers BandSet numbers.
            output_number: if True, returns the list of BandSet number; if False, returns the list of BandSet object.

        Returns:
            List of BandSets, or list of BandSets numbers if output_number is True.

        Examples:
            Iterate BandSets.
                >>> catalog = BandSetCatalog()
                >>> bandset_t = catalog.get_bandsets_by_list()
                >>> for bandset in bandset_t:
                >>>     print(bandset)
        """  # noqa: E501
        bandsets = []
        if bandset_list is None:
            bandset_list = range(1, self.get_bandset_count() + 1)
        for i in bandset_list:
            if output_number:
                bandsets.append(i)
            else:
                bandsets.append(self.get_bandset(i))
        return bandsets

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
        self.bandsets_table['root_directory'][
            self.bandsets_table[
                'bandset_number'] == bandset_number] = root_directory

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

    def set_crs(self, crs: str, bandset_number: Optional[int] = None):
        """Sets BandSet crs.

        Sets BandSet crs from Bandset information.

        Args:
            crs: crs string.
            bandset_number: number of BandSet; if None, current BandSet is used.

        Examples:
            Set BandSet 1 crs.
                >>> catalog = BandSetCatalog()
                >>> catalog.set_crs(bandset_number=1, crs='PROJCS["WGS 84 / UTM zone 33N"...')
        """  # noqa: E501
        if bandset_number is None:
            bandset_number = self.current_bandset
        self.get_bandset(bandset_number).crs = crs
        self.bandsets_table['crs'][
            self.bandsets_table['bandset_number'] == bandset_number] = crs

    def update_crs(self, bandset_number: Optional[int] = None):
        """Updates BandSet crs.

        Updates BandSet crs from Bandset first band.

        Args:
            bandset_number: number of BandSet; if None, current BandSet is used.

        Examples:
            Update BandSet 1 crs.
                >>> catalog = BandSetCatalog()
                >>> catalog.update_crs(bandset_number=1)
        """  # noqa: E501
        if bandset_number is None:
            bandset_number = self.current_bandset
        crs = self.get_bandset(bandset_number).get_band_attributes('crs')
        if crs is not None:
            crs = crs[0]
        self.get_bandset(bandset_number).crs = crs
        self.bandsets_table['crs'][
            self.bandsets_table['bandset_number'] == bandset_number] = crs

    def get_crs(self, bandset_number: Optional[int] = None):
        """Gets BandSet crs.

        Gets BandSet crs from Bandset information.

        Args:
            bandset_number: number of BandSet; if None, current BandSet is used.

        Examples:
            Get BandSet 1 crs.
                >>> catalog = BandSetCatalog()
                >>> catalog.get_crs(bandset_number=1)
        """  # noqa: E501
        if bandset_number is None:
            bandset_number = self.current_bandset
        crs = self.get_bandset(bandset_number).crs
        return crs

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
            self.bandsets_table[
                'bandset_number'] == bandset_number] = box_coordinate_list[0]
        self.bandsets_table['box_coordinate_top'][
            self.bandsets_table[
                'bandset_number'] == bandset_number] = box_coordinate_list[1]
        self.bandsets_table['box_coordinate_right'][
            self.bandsets_table[
                'bandset_number'] == bandset_number] = box_coordinate_list[2]
        self.bandsets_table['box_coordinate_bottom'][
            self.bandsets_table[
                'bandset_number'] == bandset_number] = box_coordinate_list[3]

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
        """  # noqa: E501get_
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
            Count of bands.
            
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
            box_coordinate_list = bandset.box_coordinate_list
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
        string_1 = np.char.add(
            '%s%s%s%s' % (cfg.variable_band_quotes, cfg.variable_bandset_name,
                          str(bandset_number), cfg.variable_band_name),
            number.astype('<U16')
        )
        string_2 = np.char.add(string_1, cfg.variable_band_quotes)
        return string_2.tolist()

    def create_virtual_raster(
            self, bandset_number: int = None, output_path: str = None,
            nodata_value: int = None, intersection: bool = False
    ) -> str:
        """Creates the virtual raster of a bandset.

        Creates the virtual raster of a bandset.

        Args:
            output_path: output path of the virtual raster; if None, use temporary path.
            bandset_number: number of BandSet; if None, current BandSet is used.
            nodata_value: nodata value.
            intersection: if True get minimum extent from input intersection, if False get maximum extent from union.
            
        Returns:
            Path of the output virtual raster.
            
        Examples:
            Create BandSet 1 virtual raster.
                >>> catalog = BandSetCatalog()
                >>> catalog.create_virtual_raster(1)
         """  # noqa: E501
        bandset = self.get(bandset_number)
        if output_path is None:
            output_path = cfg.temp.temporary_file_path(
                name_suffix=cfg.vrt_suffix)
        raster_vector.create_virtual_raster(
            output=output_path, nodata_value=nodata_value,
            intersection=intersection, bandset=bandset
        )
        cfg.logger.log.debug('output_path: %s' % str(output_path))
        return output_path

    def create_bandset_stack(
            self, bandset_number: int = None, output_path: str = None,
            nodata_value: int = None, intersection: bool = False
    ) -> str:
        """Creates a raster stack of a bandset.

        Stacks the raster bands of a bandset in a multiband raster.

        Args:
            output_path: output path of the raster; if None, use temporary path.
            bandset_number: number of BandSet; if None, current BandSet is used.
            nodata_value: nodata value.
            intersection: if True get minimum extent from input intersection, if False get maximum extent from union.

        Returns:
            Path of the output raster.

        Examples:
            Create BandSet 1 raster.
                >>> catalog = BandSetCatalog()
                >>> catalog.create_bandset_stack(1)
         """  # noqa: E501
        bandset = self.get(bandset_number)
        if output_path is None:
            output_path = cfg.temp.temporary_file_path(
                name_suffix=cfg.tif_suffix
            )
        virtual_path = cfg.temp.temporary_file_path(name_suffix=cfg.vrt_suffix)
        raster_vector.create_virtual_raster(
            output=virtual_path, nodata_value=nodata_value,
            intersection=intersection, bandset=bandset
        )
        raster_vector.gdal_copy_raster(input_raster=virtual_path,
                                       output=output_path)
        cfg.logger.log.debug('output_path: %s' % str(output_path))
        return output_path
