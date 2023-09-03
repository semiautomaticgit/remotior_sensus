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

import os
import random
import zipfile
from xml.dom import minidom
from xml.etree import cElementTree

import numpy as np

from remotior_sensus.core import (
    configurations as cfg, table_manager as tm
)
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.processor_functions import (
    spectral_signature, get_values_for_scatter_plot
)
from remotior_sensus.util import (
    raster_vector, dates_times, files_directories, read_write_files,
    plot_tools, shared_tools
)

try:
    if cfg.gdal_path is not None:
        os.add_dll_directory(cfg.gdal_path)
except Exception as error:
    cfg.logger.log.error(str(error))
try:
    from osgeo import ogr
    from osgeo import osr
except Exception as error:
    cfg.logger.log.error(str(error))
try:
    from osgeo import gdal
except Exception as error:
    cfg.logger.log.error(str(error))


class SpectralSignaturesCatalog(object):
    """A class to manage Spectral Signatures and ROIs.

    """

    def __init__(
            self, bandset: BandSet = None, catalog_table=None,
            geometry_file_path=None, macroclass_field=None, class_field=None
    ):
        # relative BandSet
        self.bandset = bandset
        # spectral signatures catalog table (value, wavelength, and sd)
        self.table = catalog_table
        # dictionary of spectral signature tables (macroclass, class, selected)
        self.signatures = {}
        # dictionary of names of macroclasses
        self.macroclasses = {}
        # dictionary of color strings of macroclasses
        self.macroclasses_color_string = {}
        # vector file linked to signature table containing ROIs
        if geometry_file_path is None:
            geometry_file_path = cfg.temp.temporary_file_path(
                name_suffix=cfg.gpkg_suffix
            )
        if not macroclass_field:
            macroclass_field = cfg.macroclass_field_name
        if not class_field:
            class_field = cfg.class_field_name
        # default macroclass field name
        self.macroclass_field = macroclass_field
        # default class field name
        self.class_field = class_field
        self.geometry_file = geometry_file_path
        self.crs = None
        # create geometry vector
        if bandset:
            self.crs = bandset.crs
            if self.crs is not None:
                raster_vector.create_geometry_vector(
                    output_path=self.geometry_file, crs_wkt=self.crs,
                    macroclass_field_name=self.macroclass_field,
                    class_field_name=self.class_field
                )
            else:
                cfg.logger.log.debug('bandset without crs')

    # add spectral signature to Spectral Signatures Catalog
    def add_spectral_signature(
            self, value_list, macroclass_id=None, class_id=None,
            macroclass_name=None, class_name=None, wavelength_list=None,
            standard_deviation_list=None, signature_id=None, selected=1,
            min_dist_thr=0, max_like_thr=0, spec_angle_thr=0, geometry=0,
            signature=0, color_string=None, pixel_count=0, unit=None
    ):
        """Adds a spectral signature.

        This method adds spectral signature to Spectral Signatures Catalog.

        Args:
            value_list:
            macroclass_id:
            class_id:
            macroclass_name:
            class_name:
            wavelength_list:
            standard_deviation_list:
            signature_id:
            selected:
            min_dist_thr:
            max_like_thr:
            spec_angle_thr:
            geometry:
            signature:
            color_string:
            pixel_count: pixel count
            unit: unit

        Returns:
            object OutputManger

        """
        cfg.logger.log.debug('start')
        if macroclass_id is None:
            macroclass_id = 1
        if class_id is None:
            class_id = 1
        if color_string is None:
            color_string = shared_tools.random_color()
        # signature id
        if signature_id is None:
            signature_id = generate_signature_id()
        if wavelength_list is None:
            wavelength_list = self.bandset.get_wavelengths()
        # create signature table
        self.signatures[signature_id] = tm.create_spectral_signature_table(
            value_list=value_list, wavelength_list=wavelength_list,
            standard_deviation_list=standard_deviation_list
        )
        self.signature_to_catalog(
            signature_id=signature_id, macroclass_id=macroclass_id,
            class_id=class_id, macroclass_name=macroclass_name,
            class_name=class_name, selected=selected,
            min_dist_thr=min_dist_thr, max_like_thr=max_like_thr,
            spec_angle_thr=spec_angle_thr, geometry=geometry,
            signature=signature, color_string=color_string,
            pixel_count=pixel_count, unit=unit
        )
        cfg.logger.log.debug('end')

    # sets crs from bandset
    def set_crs_from_bandset(self, bandset):
        self.crs = bandset.crs

    # sets macroclass color string
    def set_macroclass_color(self, macroclass_id, color_string):
        self.macroclasses_color_string[macroclass_id] = color_string

    # add spectral signature reference to Spectral Signatures Catalog
    def signature_to_catalog(
            self, signature_id, macroclass_id, class_id, macroclass_name=None,
            class_name=None, selected=1, min_dist_thr=0, max_like_thr=0,
            spec_angle_thr=0, geometry=0, signature=0, color_string=None,
            pixel_count=0, unit=None
    ):
        # add signature to catalog
        self.table = tm.add_spectral_signature_to_catalog_table(
            signature_id=signature_id, macroclass_id=macroclass_id,
            class_id=class_id, class_name=class_name,
            previous_catalog=self.table, selected=selected,
            min_dist_thr=min_dist_thr, max_like_thr=max_like_thr,
            spec_angle_thr=spec_angle_thr, geometry=geometry,
            signature=signature, color_string=color_string,
            pixel_count=pixel_count, unit=unit
        )
        # add or update macroclass name
        if macroclass_name is not None:
            self.macroclasses[macroclass_id] = str(macroclass_name)
        # check macroclass name
        if macroclass_id not in self.macroclasses:
            self.macroclasses[macroclass_id] = '%s%s' % (
                cfg.macroclass_default, str(len(self.macroclasses) + 1))
        if macroclass_id not in self.macroclasses_color_string:
            self.macroclasses_color_string[macroclass_id] = color_string

    # calculate scatter plot for geometry from Spectral Signatures Catalog
    def calculate_scatter_plot_by_id(
            self, signature_id: str, band_x, band_y,
            decimal_round: int = None, plot=False, n_processes: int = None
    ):
        cfg.logger.log.debug('start')
        cfg.logger.log.debug(
            'signature_id: %s; decimal_round: %s'
            % (signature_id, decimal_round)
        )
        geometry = self.table[
            self.table['signature_id'] == signature_id].geometry[0]
        # geometry
        if geometry == 1:
            if not files_directories.is_file(self.geometry_file):
                cfg.logger.log.error(
                    'geometry file not found: %s' % self.geometry_file
                )
                raise Exception('geometry file not found')
            vector = raster_vector.get_polygon_from_vector(
                vector_path=self.geometry_file,
                attribute_filter="%s = '%s'" % (
                    cfg.uid_field_name, signature_id)
            )
            value_list = self.calculate_scatter_plot(
                roi_path=vector, band_x=band_x, band_y=band_y,
                n_processes=n_processes
            )
            if value_list is False:
                cfg.logger.log.error('unable to calculate')
                cfg.messages.error('unable to calculate')
                return False
            # calculate histogram
            histogram = shared_tools.calculate_2d_histogram(
                x_values=value_list[0], y_values=value_list[1],
                decimal_round=decimal_round
            )
            if plot is True:
                ax = plot_tools.prepare_scatter_plot()
                plots = plot_tools.add_values_to_scatter_plot(
                    histogram=histogram, ax=ax
                )
                macroclass_value = self.table[
                    self.table['signature_id'] == signature_id].macroclass_id[
                    0]
                macroclass_name = self.macroclasses[macroclass_value]
                class_value = self.table[
                    self.table['signature_id'] == signature_id].class_id[
                    0]
                class_name = self.table[
                    self.table['signature_id'] == signature_id].class_name[
                    0]
                plot_names = '%s#%s %s#%s' % (
                    macroclass_value, macroclass_name, class_value, class_name
                )
                plot_tools.create_scatter_plot(
                    ax=ax, plots=[plots], plot_names=[plot_names]
                )
            cfg.logger.log.debug('end')
            return histogram
        # not geometry
        else:
            cfg.logger.log.error('unable to calculate, missing geometry')
            cfg.messages.error('unable to calculate, missing geometry')
            return False

    # remove spectral signature and geometry from Spectral Signatures Catalog
    def remove_signature_by_id(self, signature_id: str):
        cfg.logger.log.debug('start')
        # remove signature
        try:
            del self.signatures[signature_id]
        except Exception as err:
            str(err)
        try:
            geometry = self.table[
                self.table['signature_id'] == signature_id].geometry[0]
            macroclass_value = self.table[
                self.table['signature_id'] == signature_id].macroclass_id[0]
        except Exception as err:
            str(err)
            cfg.logger.log.error('signature not found: %s' % signature_id)
            cfg.messages.error('signature not found: %s' % signature_id)
            return False
        # remove signature from table
        self.table = self.table[self.table['signature_id'] != signature_id]
        if macroclass_value not in self.table.macroclass_id.tolist():
            try:
                del self.macroclasses[macroclass_value]
                del self.macroclasses_color_string[macroclass_value]
            except Exception as err:
                str(err)
        if geometry == 1:
            if not files_directories.is_file(self.geometry_file):
                cfg.logger.log.error(
                    'geometry file not found: %s' % self.geometry_file
                )
                raise Exception('geometry file not found')
            # remove geometry
            raster_vector.remove_polygon_from_vector(
                vector_path=self.geometry_file,
                attribute_field=cfg.uid_field_name,
                attribute_value=signature_id
            )
        cfg.logger.log.debug('end')

    # merge spectral signatures and geometry from Spectral Signatures Catalog
    def merge_signatures_by_id(
            self, signature_id_list, calculate_signature=True,
            macroclass_id=None, class_id=None, macroclass_name=None,
            class_name=None, color_string=None
    ):
        cfg.logger.log.debug('start')
        geometry_check = True
        macroclass_value = 0
        class_value = 0
        geometry_ids = []
        signature_ids = []
        # get signatures and geometries from list
        count = 0
        for signature_id in signature_id_list:
            if cfg.action is False:
                break
            count += 1
            try:
                geometry = self.table[
                    self.table['signature_id'] == signature_id].geometry[0]
                # not geometry
                if geometry == 0:
                    geometry_check = False
                    signature_ids.append(signature_id)
                # geometry
                elif geometry == 1:
                    geometry_ids.append(signature_id)
                    signature_check = self.table[
                        self.table['signature_id'] == signature_id
                        ].signature[0]
                    # calculate signature
                    if calculate_signature is True and signature_check == 0:
                        if not files_directories.is_file(self.geometry_file):
                            cfg.logger.log.error(
                                'geometry file not found: %s'
                                % self.geometry_file
                            )
                            raise Exception('geometry file not found')
                        vector = raster_vector.get_polygon_from_vector(
                            vector_path=self.geometry_file,
                            attribute_filter="%s = '%s'" % (
                                cfg.uid_field_name, signature_id)
                        )
                        (value_list, standard_deviation_list, wavelength_list,
                         pixel_count) = self.calculate_signature(vector)
                        mc_value = self.table[
                            self.table['signature_id'] == signature_id
                            ].macroclass_id[0]
                        c_value = self.table[
                            self.table['signature_id'] == signature_id
                            ].class_id[0]
                        c_name = self.table[
                            self.table['signature_id'] == signature_id
                            ].class_name[0]
                        unit = self.table[
                            self.table['signature_id'] == signature_id
                            ].unit[0]
                        if color_string is None:
                            color_string = self.table[
                                self.table['signature_id'] == signature_id
                                ].color[0]
                        mc_name = self.macroclasses[mc_value]
                        self.add_spectral_signature(
                            value_list=value_list, macroclass_id=mc_value,
                            class_id=c_value, macroclass_name=mc_name,
                            class_name=c_name,
                            standard_deviation_list=standard_deviation_list,
                            signature_id=signature_id, geometry=1, signature=1,
                            color_string=color_string, pixel_count=pixel_count,
                            unit=unit
                        )
                    signature_ids.append(signature_id)
                # get first element class and macroclass
                if count == 1:
                    macroclass_value = self.table[
                        self.table['signature_id'] == signature_id
                        ].macroclass_id[0]
                    class_value = self.table[
                        self.table['signature_id'] == signature_id
                        ].class_id[0]
            except Exception as err:
                cfg.logger.log.error(str(err))
        if macroclass_id is not None:
            macroclass_value = macroclass_id
        if macroclass_name is None:
            macroclass_name = self.macroclasses[macroclass_value]
        if class_id is not None:
            class_value = class_id
        if class_name is None:
            class_name = 'merged'
        if color_string is None:
            color_string = shared_tools.random_color()
        # merge geometries if geometry == 1 for whole signature_id_list
        if geometry_check is True:
            temp_path = cfg.temp.temporary_file_path(
                name_suffix=cfg.gpkg_suffix
            )
            merged = raster_vector.merge_polygons(
                input_layer=self.geometry_file, value_list=signature_id_list,
                target_layer=temp_path
            )
            # import vector
            self.import_vector(
                file_path=merged, macroclass_value=macroclass_value,
                class_value=class_value, macroclass_name=macroclass_name,
                class_name=class_name, calculate_signature=calculate_signature,
                color_string=color_string
            )
        # merge signatures if not geometry
        else:
            wavelength = None
            value_arrays = []
            std_arrays = []
            for signature in signature_ids:
                if cfg.action is False:
                    break
                value_arrays.append(self.signatures[signature].value)
                std_arrays.append(
                    self.signatures[signature].standard_deviation
                )
                wavelength = self.signatures[signature].wavelength
            unit = self.table[
                self.table['signature_id'] == signature_ids[0]].unit[0]
            wavelength_list = wavelength.tolist()
            values = np.column_stack(value_arrays)
            stds = np.column_stack(std_arrays)
            values_mean = np.mean(values, axis=1)
            stds_squared = np.square(stds)
            stds_squared_sum = np.sum(stds_squared, axis=1)
            stds_variance = np.divide(stds_squared_sum, stds_squared.shape[1])
            stds_mean = np.sqrt(stds_variance)
            self.add_spectral_signature(
                value_list=values_mean.tolist(),
                macroclass_id=macroclass_value, class_id=class_value,
                macroclass_name=macroclass_name, class_name='merged',
                wavelength_list=wavelength_list,
                standard_deviation_list=stds_mean.tolist(), geometry=0,
                signature=1, color_string=color_string, pixel_count=0,
                unit=unit
            )
        cfg.logger.log.debug('end')

    # export signatures as csv
    def export_signatures_as_csv(
            self, signature_id_list, output_directory, separator=','
    ):
        cfg.logger.log.debug(
            'export_signatures_as_csv: %s' % str(signature_id_list)
        )
        files_directories.create_directory(output_directory)
        output_list = []
        for signature_id in signature_id_list:
            if cfg.action is False:
                break
            try:
                values = self.signatures[signature_id].value.tolist()
                wavelength = self.signatures[signature_id].wavelength.tolist()
                standard_deviation = (
                    self.signatures[signature_id].standard_deviation.tolist()
                )
                macroclass_id = self.table[
                    self.table['signature_id'] == signature_id
                    ].macroclass_id[0]
                macroclass_name = self.macroclasses[macroclass_id]
                class_id = self.table[
                    self.table['signature_id'] == signature_id
                    ].class_id[0]
                class_name = self.table[
                    self.table['signature_id'] == signature_id
                    ].class_name[0]
                output_file = '%s/%s_%s_%s_%s.csv' % (
                    output_directory, macroclass_id, macroclass_name,
                    class_id, class_name
                )
                text = ''
                for v in range(len(values)):
                    if cfg.action is False:
                        break
                    text += '%s%s%s%s%s\n' % (
                        values[v], separator, wavelength[v], separator,
                        standard_deviation[v]
                    )
                read_write_files.write_file(text, output_file)
                output_list.append(output_file)
            except Exception as err:
                cfg.logger.log.error(str(err))
        return output_list

    # import spectral signature csv to Spectral Signatures Catalog
    def import_spectral_signature_csv(
            self, csv_path, macroclass_id=None, class_id=None,
            macroclass_name=None, class_name=None, separator=',',
            color_string=None
    ):
        cfg.logger.log.debug('start')
        # import csv as comma separated with fields value, wavelength,
        # standard_deviation (optional)
        if files_directories.is_file(csv_path):
            csv = tm.open_file(
                file_path=csv_path, separators=separator,
                field_names=['value', 'wavelength',
                             'standard_deviation'],
                progress_message=False, skip_first_line=False
            )
            bandset_wavelength = self.bandset.bands['wavelength']
            unit = self.bandset.get_wavelength_units()[0]
            value_list = []
            wavelength_list = []
            standard_deviation_list = []
            if 'wavelength' in tm.columns(csv):
                for b in bandset_wavelength.tolist():
                    if cfg.action is False:
                        break
                    arg_min = np.abs(csv.wavelength - b).argmin()
                    wavelength_list.append(b)
                    value_list.append(csv.value[arg_min])
                    if 'standard_deviation' in tm.columns(csv):
                        standard_deviation_list.append(
                            csv.standard_deviation[arg_min]
                        )
            if len(value_list) == 0:
                cfg.logger.log.error('file: %s' % csv_path)
                cfg.messages.error('error importing file %s' % csv_path)
                return
            if len(wavelength_list) == 0:
                wavelength_list = None
            if len(standard_deviation_list) == 0:
                standard_deviation_list = None
            if color_string is None:
                color_string = shared_tools.random_color()
            self.add_spectral_signature(
                value_list=value_list, macroclass_id=macroclass_id,
                class_id=class_id, macroclass_name=macroclass_name,
                class_name=class_name, wavelength_list=wavelength_list,
                standard_deviation_list=standard_deviation_list, geometry=0,
                signature=1, color_string=color_string, pixel_count=0,
                unit=unit
            )
            cfg.logger.log.debug('end; imported: %s' % csv_path)
        else:
            cfg.logger.log.error('error file not found: %s' % csv_path)
            cfg.messages.error('error file not found: %s' % csv_path)

    # import Spectral Signatures Catalog file
    def import_file(self, file_path):
        cfg.logger.log.debug(
            'import_file: %s' % file_path
        )
        # create temporary directory
        temp_dir = cfg.temp.create_temporary_directory()
        file_list = files_directories.unzip_file(file_path, temp_dir)
        # list of new ids
        signature_ids = {}
        geometry_ids = {}
        geometry_file = None
        table = None
        for f in file_list:
            if cfg.action is True:
                f_name = files_directories.file_name(f, suffix=True)
                if f_name == 'geometry.gpkg':
                    geometry_file = f
                elif f_name == 'table':
                    table = np.core.records.fromfile(
                        f, dtype=cfg.spectral_dtype_list
                    )
                    # remove file
                    files_directories.remove_file(f)
                elif f_name == 'macroclasses.xml':
                    tree = cElementTree.parse(f)
                    root = tree.getroot()
                    version = root.get('version')
                    if version is None:
                        cfg.logger.log.error(
                            'failed loading signatures: %s'
                            % file_path
                        )
                        cfg.messages.error(
                            'failed loading signatures: %s'
                            % file_path
                        )
                    else:
                        for child in root:
                            if cfg.action is False:
                                break
                            macroclass_id = child.get('id')
                            macroclass_name = child.get('name')
                            macroclass_color = child.get('color')
                            if int(macroclass_id) not in self.macroclasses:
                                self.macroclasses[int(macroclass_id)] = str(
                                    macroclass_name
                                )
                            if (int(macroclass_id)
                                    not in self.macroclasses_color_string):
                                self.macroclasses_color_string[
                                    int(macroclass_id)] = str(macroclass_color)
                    # remove file
                    files_directories.remove_file(f)
                else:
                    signature_id = generate_signature_id()
                    signature_ids[f_name] = signature_id
                    f_name = signature_id
                    self.signatures[f_name] = np.core.records.fromfile(
                        f, dtype=cfg.signature_dtype_list
                    )
                    # remove file
                    files_directories.remove_file(f)
            else:
                cfg.logger.log.error('cancel')
                return
        cfg.logger.log.debug('signature_ids: %s' % signature_ids)
        # import vector
        if geometry_file is not None:
            # get vector crs
            vector_crs = raster_vector.get_crs(geometry_file)
            if self.crs is None:
                cfg.logger.log.error('crs not defined')
                raise Exception('crs not defined')
            # check crs
            catalog_sr = osr.SpatialReference()
            catalog_sr.ImportFromWkt(self.crs)
            vector_sr = osr.SpatialReference()
            vector_sr.ImportFromWkt(vector_crs)
            # required by GDAL 3 coordinate order
            try:
                catalog_sr.SetAxisMappingStrategy(
                    osr.OAMS_TRADITIONAL_GIS_ORDER
                )
                vector_sr.SetAxisMappingStrategy(
                    osr.OAMS_TRADITIONAL_GIS_ORDER
                )
            except Exception as err:
                str(err)
            if catalog_sr.IsSame(vector_sr) == 1:
                coord_transform = None
            else:
                # coordinate transformation
                coord_transform = osr.CoordinateTransformation(
                    vector_sr, catalog_sr
                )
            # open input vector
            i_vector = ogr.Open(geometry_file)
            i_layer = i_vector.GetLayer()
            catalog_vector = ogr.Open(self.geometry_file, 1)
            catalog_layer = catalog_vector.GetLayer()
            catalog_layer_definition = catalog_layer.GetLayerDefn()
            # import geometries
            i_feature = i_layer.GetNextFeature()
            while i_feature:
                if cfg.action is True:
                    # get geometry
                    geom = i_feature.GetGeometryRef()
                    if coord_transform is not None:
                        # project feature
                        geom.Transform(coord_transform)
                    o_feature = ogr.Feature(catalog_layer_definition)
                    o_feature.SetGeometry(geom)
                    sig_id = i_feature.GetField(cfg.uid_field_name)
                    mc_value = i_feature.GetField(self.macroclass_field)
                    c_value = i_feature.GetField(self.class_field)
                    if sig_id in signature_ids:
                        o_feature.SetField(
                            cfg.uid_field_name, signature_ids[sig_id]
                        )
                    else:
                        signature_id = generate_signature_id()
                        geometry_ids[sig_id] = signature_id
                        o_feature.SetField(
                            cfg.uid_field_name, signature_id
                        )
                    o_feature.SetField(cfg.class_field_name, int(c_value))
                    o_feature.SetField(
                        cfg.macroclass_field_name, int(mc_value)
                    )
                    catalog_layer.CreateFeature(o_feature)
                    o_feature.Destroy()
                    i_feature.Destroy()
                    i_feature = i_layer.GetNextFeature()
                else:
                    # close files
                    i_vector.Destroy()
                    catalog_vector.Destroy()
                    i_feature = False
                    cfg.logger.log.error('cancel')
        # import table
        if table is not None:
            for sig_id in signature_ids:
                table['signature_id'][
                    table['signature_id'] == sig_id] = signature_ids[sig_id]
            for sig_id in geometry_ids:
                table['signature_id'][
                    table['signature_id'] == sig_id] = geometry_ids[sig_id]
            self.table = tm.append_tables(self.table, table)

    # import vector to Spectral Signatures Catalog
    def import_vector(
            self, file_path, macroclass_value=None, class_value=None,
            macroclass_name=None, class_name=None, macroclass_field=None,
            class_field=None, macroclass_name_field=None,
            class_name_field=None, calculate_signature=True,
            color_string=None
    ):
        cfg.logger.log.debug('start')
        if files_directories.is_file(file_path):
            # check geometry vector
            if not files_directories.is_file(self.geometry_file):
                if self.bandset is None:
                    cfg.logger.log.error('bandset not found')
                    raise Exception('bandset not found')
                if self.crs is not None:
                    raster_vector.create_geometry_vector(
                        output_path=self.geometry_file, crs_wkt=self.crs,
                        macroclass_field_name=self.macroclass_field,
                        class_field_name=self.class_field
                    )
                else:
                    cfg.logger.log.error('crs not defined')
                    raise Exception('crs not defined')
            # get vector crs
            vector_crs = raster_vector.get_crs(file_path)
            if self.crs is None:
                cfg.logger.log.error('crs not defined')
                raise Exception('crs not defined')
            # check crs
            catalog_sr = osr.SpatialReference()
            catalog_sr.ImportFromWkt(self.crs)
            vector_sr = osr.SpatialReference()
            vector_sr.ImportFromWkt(vector_crs)
            unit = self.bandset.get_wavelength_units()[0]
            # required by GDAL 3 coordinate order
            try:
                catalog_sr.SetAxisMappingStrategy(
                    osr.OAMS_TRADITIONAL_GIS_ORDER
                )
                vector_sr.SetAxisMappingStrategy(
                    osr.OAMS_TRADITIONAL_GIS_ORDER
                )
            except Exception as err:
                str(err)
            if catalog_sr.IsSame(vector_sr) == 1:
                coord_transform = None
            else:
                # coordinate transformation
                coord_transform = osr.CoordinateTransformation(
                    vector_sr, catalog_sr
                )
            # open input vector
            i_vector = ogr.Open(file_path)
            i_layer = i_vector.GetLayer()
            catalog_vector = ogr.Open(self.geometry_file, 1)
            catalog_layer = catalog_vector.GetLayer()
            catalog_layer_definition = catalog_layer.GetLayerDefn()
            # import geometries
            i_feature = i_layer.GetNextFeature()
            while i_feature:
                if cfg.action is True:
                    signature_id = generate_signature_id()
                    # get geometry
                    geom = i_feature.GetGeometryRef()
                    if coord_transform is not None:
                        # project feature
                        geom.Transform(coord_transform)
                    o_feature = ogr.Feature(catalog_layer_definition)
                    o_feature.SetGeometry(geom)
                    if macroclass_value is None:
                        mc_value = i_feature.GetField(macroclass_field)
                    else:
                        mc_value = macroclass_value
                    if class_value is None:
                        c_value = i_feature.GetField(class_field)
                    else:
                        c_value = class_value
                    if macroclass_name is None:
                        mc_name = i_feature.GetField(macroclass_name_field)
                    else:
                        mc_name = macroclass_name
                    if class_name is None:
                        c_name = i_feature.GetField(class_name_field)
                    else:
                        c_name = class_name
                    o_feature.SetField(cfg.uid_field_name, signature_id)
                    o_feature.SetField(cfg.class_field_name, int(c_value))
                    o_feature.SetField(
                        cfg.macroclass_field_name, int(mc_value)
                    )
                    catalog_layer.CreateFeature(o_feature)
                    if color_string is None:
                        color_string = shared_tools.random_color()
                    if calculate_signature:
                        temp_path = cfg.temp.temporary_file_path(
                            name_suffix=cfg.gpkg_suffix
                        )
                        raster_vector.create_geometry_vector(
                            output_path=temp_path, crs_wkt=self.crs,
                            macroclass_field_name=self.macroclass_field,
                            class_field_name=self.class_field
                        )
                        temp_vector = ogr.Open(temp_path, 1)
                        temp_layer = temp_vector.GetLayer()
                        temp_layer.CreateFeature(o_feature)
                        temp_vector.Destroy()
                        try:
                            (value_list, standard_deviation_list,
                             wavelength_list,
                             pixel_count) = self.calculate_signature(temp_path)
                        except Exception as err:
                            cfg.logger.log.error(str(err))
                            return False
                        self.add_spectral_signature(
                            value_list=value_list, macroclass_id=mc_value,
                            class_id=c_value, macroclass_name=mc_name,
                            class_name=c_name,
                            standard_deviation_list=standard_deviation_list,
                            signature_id=signature_id, geometry=1, signature=1,
                            color_string=color_string, pixel_count=pixel_count,
                            unit=unit
                        )
                    else:
                        self.signature_to_catalog(
                            signature_id=signature_id, macroclass_id=mc_value,
                            class_id=c_value, macroclass_name=mc_name,
                            class_name=c_name, geometry=1, signature=0,
                            color_string=color_string, unit=unit
                        )
                    o_feature.Destroy()
                    i_feature.Destroy()
                    i_feature = i_layer.GetNextFeature()
                else:
                    # close files
                    i_vector.Destroy()
                    catalog_vector.Destroy()
                    i_feature = False
                    cfg.logger.log.error('cancel')
            # close files
            i_vector.Destroy()
            catalog_vector.Destroy()
            cfg.logger.log.debug('end; imported: %s' % file_path)
        else:
            cfg.logger.log.error('error file not found: %s' % file_path)
            cfg.messages.error('error file not found: %s' % file_path)

    # calculate spectral signatures
    def calculate_signature(self, roi_path, n_processes: int = None):
        cfg.logger.log.debug('calculate_signature: %s' % roi_path)
        if n_processes is None:
            n_processes = cfg.n_processes
        min_x, max_x, min_y, max_y = raster_vector.get_layer_extent(roi_path)
        path_list = self.bandset.get_absolute_paths()
        virtual_path_list = []
        for p in path_list:
            if cfg.action is False:
                break
            temp_path = cfg.temp.temporary_file_path(
                name_suffix=cfg.tif_suffix
            )
            virtual = raster_vector.create_virtual_raster(
                input_raster_list=[p], output=temp_path,
                box_coordinate_list=[min_x, max_y, max_x, min_y]
            )
            virtual_path_list.append(virtual)
        roi_paths = [roi_path] * len(path_list)
        cfg.multiprocess.run_separated(
            raster_path_list=virtual_path_list, function=spectral_signature,
            function_argument=roi_paths, function_variable=virtual_path_list,
            n_processes=n_processes, keep_output_argument=True,
            progress_message='calculate signature', min_progress=1,
            max_progress=80
        )
        wavelength_list = self.bandset.get_wavelengths()
        cfg.multiprocess.multiprocess_spectral_signature()
        if cfg.multiprocess.output is False:
            return False
        else:
            (value_list, standard_deviation_list,
             count_list) = cfg.multiprocess.output
            return (
                value_list, standard_deviation_list, wavelength_list,
                count_list
            )

    # import vector for scatter plot
    def calculate_scatter_plot(
            self, roi_path, band_x, band_y, n_processes: int = None
    ):
        cfg.logger.log.debug(
            'calculate_scatter_plot: %s; band x: %s; band y: %s'
            % (roi_path, band_x, band_y)
        )
        if n_processes is None:
            n_processes = cfg.n_processes
        min_x, max_x, min_y, max_y = raster_vector.get_layer_extent(roi_path)
        band_x_path = self.bandset.get_absolute_path(band_number=band_x)
        band_y_path = self.bandset.get_absolute_path(band_number=band_y)
        if band_x_path is None or band_y_path is None:
            cfg.logger.log.error('failed to get bands')
            cfg.messages.error('failed to get bands')
            return False
        path_list = [band_x_path, band_y_path]
        virtual_path_list = []
        for p in path_list:
            if cfg.action is False:
                break
            temp_path = cfg.temp.temporary_file_path(
                name_suffix=cfg.tif_suffix
            )
            virtual = raster_vector.create_virtual_raster(
                input_raster_list=[p], output=temp_path,
                box_coordinate_list=[min_x, max_y, max_x, min_y]
            )
            virtual_path_list.append(virtual)
        roi_paths = [roi_path] * len(path_list)
        if n_processes > 2:
            n_processes = 2
        cfg.multiprocess.run_separated(
            raster_path_list=virtual_path_list,
            function=get_values_for_scatter_plot,
            function_argument=roi_paths, function_variable=virtual_path_list,
            n_processes=n_processes, keep_output_argument=True,
            progress_message='calculate band values', min_progress=1,
            max_progress=80
        )
        cfg.multiprocess.multiprocess_scatter_values()
        if cfg.multiprocess.output is False:
            return False
        value_list = cfg.multiprocess.output
        return value_list

    # save Spectral Signatures Catalog to file
    def save(self, output_path, signature_id_list=None):
        cfg.logger.log.debug(
            'save Spectral Signatures Catalog: %s' % output_path
        )
        # file list
        file_list = []
        # create temporary directory
        temp_dir = cfg.temp.create_temporary_directory()
        # geometry file
        if files_directories.is_file(self.geometry_file):
            if signature_id_list is None:
                files_directories.copy_file(
                    self.geometry_file, '%s/geometry.gpkg' % temp_dir
                )
            else:
                self.export_vector(
                    signature_id_list, '%s/geometry.gpkg' % temp_dir
                )
            file_list.append('%s/geometry.gpkg' % temp_dir)
        # create xml file
        root = cElementTree.Element('macroclasses')
        root.set('version', str(cfg.version))
        root.set('macroclass_field', str(self.macroclass_field))
        root.set('class_field', str(self.class_field))
        if self.signatures is not None:
            for signature_id in self.signatures:
                if cfg.action is False:
                    break
                if (signature_id_list is None
                        or signature_id in signature_id_list):
                    # create file inside temporary directory
                    self.signatures[signature_id].tofile(
                        file='%s/%s' % (temp_dir, signature_id)
                    )
                    file_list.append('%s/%s' % (temp_dir, signature_id))
        if signature_id_list is None:
            macroclass_list = []
        else:
            macroclass_list = self.table[
                np.in1d(self.table['signature_id'], signature_id_list)
            ].macroclass_id.tolist()
        if self.macroclasses is not None:
            for macroclass in self.macroclasses:
                if signature_id_list is None or macroclass in macroclass_list:
                    macroclass_element = cElementTree.SubElement(
                        root, 'macroclass'
                    )
                    macroclass_element.set('id', str(macroclass))
                    macroclass_element.set(
                        'name',
                        str(self.macroclasses[macroclass])
                    )
                    macroclass_element.set(
                        'color',
                        str(self.macroclasses_color_string[macroclass])
                    )
        # save to file
        pretty_xml = minidom.parseString(
            cElementTree.tostring(root)
        ).toprettyxml()
        # create file inside temporary directory
        read_write_files.write_file(
            pretty_xml,
            '%s/macroclasses.xml' % temp_dir
        )
        file_list.append('%s/macroclasses.xml' % temp_dir)
        # create file inside temporary directory
        if self.table is not None:
            if signature_id_list is None:
                self.table.tofile(file='%s/table' % temp_dir)
            else:
                self.table[
                    np.in1d(self.table['signature_id'], signature_id_list)
                ].tofile(file='%s/table' % temp_dir)
            file_list.append('%s/table' % temp_dir)
        # zip files
        files_directories.zip_files(
            file_list, output_path,
            compression=zipfile.ZIP_STORED
        )
        cfg.logger.log.debug('saved signature catalog')
        return output_path

    # load Spectral Signatures Catalog from file
    def load(self, file_path):
        cfg.logger.log.debug(
            'load Spectral Signatures Catalog: %s' % file_path
        )
        self.table = None
        self.signatures = {}
        # create temporary directory
        temp_dir = cfg.temp.create_temporary_directory()
        file_list = files_directories.unzip_file(file_path, temp_dir)
        for f in file_list:
            if cfg.action is False:
                break
            f_name = files_directories.file_name(f, suffix=True)
            if f_name == 'geometry.gpkg':
                self.geometry_file = f
            elif f_name == 'table':
                self.table = np.core.records.fromfile(
                    f, dtype=cfg.spectral_dtype_list
                )
                # remove file
                files_directories.remove_file(f)
            elif f_name == 'macroclasses.xml':
                tree = cElementTree.parse(f)
                root = tree.getroot()
                version = root.get('version')
                if version is None:
                    cfg.logger.log.error(
                        'failed loading signatures: %s'
                        % file_path
                    )
                    cfg.messages.error(
                        'failed loading signatures: %s'
                        % file_path
                    )
                else:
                    self.macroclass_field = root.get('macroclass_field')
                    self.class_field = root.get('class_field')
                    self.macroclasses = {}
                    self.macroclasses_color_string = {}
                    for child in root:
                        if cfg.action is False:
                            break
                        macroclass_id = child.get('id')
                        macroclass_name = child.get('name')
                        macroclass_color = child.get('color')
                        self.macroclasses[int(macroclass_id)] = str(
                            macroclass_name
                        )
                        self.macroclasses_color_string[
                            int(macroclass_id)] = str(macroclass_color)
                # remove file
                files_directories.remove_file(f)
            else:
                self.signatures[f_name] = np.core.records.fromfile(
                    f, dtype=cfg.signature_dtype_list
                )
                # remove file
                files_directories.remove_file(f)

    # prepare signature values for plot
    def export_signature_values_for_plot(
            self, signature_id, plot_catalog=None
    ):
        cfg.logger.log.debug(
            'export_signature_values_for_plot: %s'
            % signature_id
        )
        # check signature
        try:
            signature = self.table[
                self.table['signature_id'] == signature_id].signature[0]
        except Exception as err:
            str(err)
            cfg.logger.log.error('signature not found: %s' % signature_id)
            cfg.messages.error('signature not found: %s' % signature_id)
            return False
        if signature == 0:
            geometry = self.table[
                self.table['signature_id'] == signature_id].geometry[0]
            # not geometry
            if geometry == 0:
                cfg.logger.log.error('signature not found: %s' % signature_id)
                cfg.messages.error('signature not found: %s' % signature_id)
                return False
            # geometry
            else:
                if files_directories.is_file(self.geometry_file):
                    # calculate signature
                    vector = raster_vector.get_polygon_from_vector(
                        vector_path=self.geometry_file,
                        attribute_filter="%s = '%s'" % (
                            cfg.uid_field_name, signature_id
                        )
                    )
                    try:
                        (value_list, standard_deviation_list, wavelength_list,
                         pixel_count) = self.calculate_signature(vector)
                    except Exception as err:
                        cfg.logger.log.error(str(err))
                        return False
                else:
                    cfg.logger.log.error(
                        'geometry file not found: %s'
                        % self.geometry_file
                    )
                    raise Exception('geometry file not found')
        else:
            value_list = self.signatures[signature_id].value
            wavelength_list = self.signatures[signature_id].wavelength
            standard_deviation_list = self.signatures[
                signature_id].standard_deviation
            pixel_count = self.table[
                self.table['signature_id'] == signature_id].pixel_count[0]
        mc_value = self.table[
            self.table['signature_id'] == signature_id].macroclass_id[0]
        c_value = self.table[
            self.table['signature_id'] == signature_id].class_id[0]
        c_name = self.table[
            self.table['signature_id'] == signature_id].class_name[0]
        color_string = self.table[
            self.table['signature_id'] == signature_id].color[0]
        mc_name = self.macroclasses[mc_value]
        signature_plot = SpectralSignaturePlot(
            value=value_list, wavelength=wavelength_list,
            standard_deviation=standard_deviation_list,
            pixel_count=pixel_count, signature_id=signature_id,
            macroclass_id=mc_value, class_id=c_value, macroclass_name=mc_name,
            class_name=c_name, color_string=color_string
        )
        if plot_catalog is not None:
            plot_catalog.add_signature(signature_plot)
        return signature_plot

    # export geometries to vector
    def export_vector(
            self, signature_id_list, output_path, vector_format=None
    ):
        cfg.logger.log.debug(
            'export_vector: %s' % str(signature_id_list)
        )
        if vector_format is None:
            vector_format = 'GPKG'
        raster_vector.save_polygons(
            input_layer=self.geometry_file, value_list=signature_id_list,
            target_layer=output_path, vector_format=vector_format
        )
        return output_path

    # display plot of signatures using Matplotlib
    def add_signatures_to_plot_by_id(self, signature_id_list):
        cfg.logger.log.debug(
            'add_signatures_to_plot_by_id: %s'
            % str(signature_id_list)
        )
        try:
            plot_catalog = SpectralSignaturePlotCatalog()
            ax = plot_tools.prepare_plot()
            for signature in signature_id_list:
                if cfg.action is False:
                    break
                self.export_signature_values_for_plot(
                    signature_id=signature, plot_catalog=plot_catalog
                )
            name_list = plot_catalog.get_signature_names()
            value_list = plot_catalog.get_signature_values()
            wavelength_list = plot_catalog.get_signature_wavelength()
            color_list = plot_catalog.get_signature_color()
            plots, plot_names, x_ticks, y_ticks, v_lines = (
                plot_tools.add_lines_to_plot(
                    name_list=name_list, wavelength_list=wavelength_list,
                    value_list=value_list, color_list=color_list
                )
            )
            plot_tools.create_plot(
                ax=ax, plots=plots, plot_names=plot_names, x_ticks=x_ticks,
                y_ticks=y_ticks, v_lines=v_lines
            )
        except Exception as err:
            cfg.logger.log.error(str(err))
            return False

    # calculate Bray-Curtis similarity
    # (100 - 100 * sum(abs(x[ki]-x[kj]) / (sum(x[ki] + x[kj])))
    def calculate_bray_curtis_similarity(self, signature_id_x, signature_id_y):
        cfg.logger.log.debug(
            'calculate_bray_curtis_similarity i: %s; j: %s'
            % (str(signature_id_x), str(signature_id_y))
        )
        value = shared_tools.calculate_bray_curtis_similarity(
            values_x=self.signatures[signature_id_x].value,
            values_y=self.signatures[signature_id_y].value
        )
        return value

    # calculate Euclidean distance sqrt(sum((x[ki] - x[kj])^2))
    def calculate_euclidean_distance(self, signature_id_x, signature_id_y):
        cfg.logger.log.debug(
            'calculate_euclidean_distance i: %s; j: %s'
            % (str(signature_id_x), str(signature_id_y))
        )
        value = shared_tools.calculate_euclidean_distance(
            values_x=self.signatures[signature_id_x].value,
            values_y=self.signatures[signature_id_y].value
        )
        return value

    # calculate Spectral angle
    # [ arccos( sum(r_i * s_i) / sqrt( sum(r_i**2) * sum(s_i**2) ) ) ]
    def calculate_spectral_angle(self, signature_id_x, signature_id_y):
        cfg.logger.log.debug(
            'Spectral angle i: %s; j: %s'
            % (str(signature_id_x), str(signature_id_y))
        )
        value = shared_tools.calculate_spectral_angle(
            values_x=self.signatures[signature_id_x].value,
            values_y=self.signatures[signature_id_y].value
        )
        return value


class SpectralSignaturePlot(object):
    """A class to manage Spectral Signatures for plots.

    """

    def __init__(
            self, value, wavelength, standard_deviation=None, pixel_count=None,
            signature_id=None, macroclass_id=None, class_id=None,
            macroclass_name=None, class_name=None, color_string=None,
            selected=None
    ):
        self.value = value
        self.wavelength = wavelength
        self.standard_deviation = standard_deviation
        if pixel_count is None:
            pixel_count = 0
        self.pixel_count = pixel_count
        if signature_id is None:
            signature_id = generate_signature_id()
        self.signature_id = signature_id
        if macroclass_id is None:
            macroclass_id = 0
        self.macroclass_id = macroclass_id
        if class_id is None:
            class_id = 0
        self.class_id = class_id
        if macroclass_name is None:
            macroclass_name = cfg.macroclass_default
        self.macroclass_name = macroclass_name
        if class_name is None:
            class_name = cfg.class_default
        self.class_name = class_name
        if color_string is None:
            color_string = '#000000'
        self.color = color_string
        if selected is None:
            selected = 1
        self.selected = selected
        # generic attributes
        self.attributes = {}


class SpectralSignaturePlotCatalog(object):
    """A class to manage Spectral Signatures Catalog for plots.

    """

    def __init__(
            self, signature: SpectralSignaturePlot = None
    ):
        self.catalog = {}
        if signature is not None:
            self.catalog[signature.signature_id] = signature

    # add signature to catalog
    def add_signature(self, signature: SpectralSignaturePlot):
        self.catalog[signature.signature_id] = signature
        cfg.logger.log.debug('add_signature: %s' % signature.signature_id)
        return True

    # add signature to catalog
    def remove_signature(self, signature_id: str):
        try:
            del self.catalog[signature_id]
            cfg.logger.log.debug('remove_signature: %s' % signature_id)
            return True
        except Exception as err:
            str(err)
            cfg.logger.log.error('signature not found: %s' % signature_id)
            cfg.messages.error('signature not found: %s' % signature_id)
            return False

    def get_signature_count(self) -> int:
        """Gets count of signatures in the catalog.

        This function gets the count of signatures present in the catalog.

        Returns:
            The integer number of signatures.

        Examples:
            Count of signatures present.
                >>> catalog = SpectralSignaturePlotCatalog()
                >>> count = catalog.get_signature_count()
                >>> print(count)
                1
        """
        return len(self.catalog)

    def get_signature(self, signature_id) -> SpectralSignaturePlot:
        """Gets signature in the catalog.

        This function gets signature by id from the catalog.

        Returns:
            The SpectralSignaturePlot.

        Examples:
            Get signature.
                >>> catalog = SpectralSignaturePlotCatalog()
                >>> signature = catalog.get_signature(signature_id='signature_id')
                >>> print(signature.signature_id)
                'signature_id'
        """  # noqa: E501
        return self.catalog[signature_id]

    def get_signature_ids(self) -> list:
        """Gets signature ids in the catalog.

        This function gets signature ids from the catalog.

        Returns:
            The list of ids.

        Examples:
            Get signature.
                >>> catalog = SpectralSignaturePlotCatalog()
                >>> signature_ids = catalog.get_signature_ids()
                >>> print(signature_ids)
                ['signature_id']
        """
        return list(self.catalog.keys())

    def get_signature_names(self, selected=True) -> list:
        """Gets signature names in the catalog.

        This function gets signature names from the catalog,
        derived from SpectralSignaturePlot.

        Returns:
            The list of values.

        Examples:
            Get signature.
                >>> catalog = SpectralSignaturePlotCatalog()
                >>> signature_names = catalog.get_signature_names()
                >>> print(signature_names)
                ['name']
        """
        property_list = []
        for signature in self.catalog:
            if cfg.action is False:
                break
            if selected is True:
                if self.catalog[signature].selected == 1:
                    property_list.append(
                        '%s#%s %s#%s' % (
                            self.catalog[signature].macroclass_id,
                            self.catalog[signature].macroclass_name,
                            self.catalog[signature].class_id,
                            self.catalog[signature].class_name)
                    )
            else:
                property_list.append(
                    '%s#%s %s#%s' % (
                        self.catalog[signature].macroclass_id,
                        self.catalog[signature].macroclass_name,
                        self.catalog[signature].class_id,
                        self.catalog[signature].class_name)
                )
        return property_list

    def get_signature_values(self, selected=True) -> list:
        """Gets signature values in the catalog.

        This function gets signature values from the catalog.

        Returns:
            The list of values.

        """
        property_list = []
        for signature in self.catalog:
            if cfg.action is False:
                break
            if selected is True:
                if self.catalog[signature].selected == 1:
                    property_list.append(
                        self.catalog[signature].value
                    )
            else:
                property_list.append(self.catalog[signature].value)
        return property_list

    def get_signature_wavelength(self, selected=True) -> list:
        """Gets signature wavelength in the catalog.

        This function gets signature wavelength from the catalog.

        Returns:
            The list of values.

        """
        property_list = []
        for signature in self.catalog:
            if cfg.action is False:
                break
            if selected is True:
                if self.catalog[signature].selected == 1:
                    property_list.append(
                        self.catalog[signature].wavelength
                    )
            else:
                property_list.append(self.catalog[signature].wavelength)
        return property_list

    def get_signature_standard_deviation(self, selected=True) -> list:
        """Gets signature standard deviation in the catalog.

        This function gets signature standard deviation from the catalog.

        Returns:
            The list of values.

        """
        property_list = []
        for signature in self.catalog:
            if cfg.action is False:
                break
            if selected is True:
                if self.catalog[signature].selected == 1:
                    property_list.append(
                        self.catalog[signature].standard_deviation
                    )
            else:
                property_list.append(
                    self.catalog[signature].standard_deviation
                )
        return property_list

    def get_signature_color(self, selected=True) -> list:
        """Gets signature color in the catalog.

        This function gets signature color from the catalog.

        Returns:
            The list of values.

        """
        property_list = []
        for signature in self.catalog:
            if cfg.action is False:
                break
            if selected is True:
                if self.catalog[signature].selected == 1:
                    property_list.append(
                        self.catalog[signature].color
                    )
            else:
                property_list.append(
                    self.catalog[signature].color
                )
        return property_list

    def get_generic_attributes(self, selected=True) -> list:
        """Gets generic attributes in the catalog.

        This function gets generic attributes from the catalog.

        Returns:
            The list of values.

        """
        property_list = []
        for signature in self.catalog:
            if cfg.action is False:
                break
            if selected is True:
                if self.catalog[signature].selected == 1:
                    property_list.append(
                        self.catalog[signature].attributes
                    )
            else:
                property_list.append(
                    self.catalog[signature].attributes
                )
        return property_list


# generate signature id
def generate_signature_id():
    times = dates_times.get_time_string()
    r = str(random.randint(0, 1000))
    uid = 's%s_%s' % (times, r)
    return uid
