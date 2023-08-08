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
from xml.etree import cElementTree
from xml.dom import minidom

import numpy as np

from remotior_sensus.core import (
    configurations as cfg, table_manager as tm
)
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.processor_functions import spectral_signature
from remotior_sensus.util import (
    raster_vector, dates_times, files_directories, read_write_files
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
        self.macroclass_field = macroclass_field
        self.class_field = class_field
        self.geometry_file = geometry_file_path
        # create geometry vector
        if bandset:
            crs = bandset.crs
            raster_vector.create_geometry_vector(
                output_path=self.geometry_file, crs_wkt=crs,
                macroclass_field_name=self.macroclass_field,
                class_field_name=self.class_field
            )

    # TODO add geometry
    # add spectral signature to Spectral Signatures Catalog
    def add_spectral_signature(
            self, value_list, macroclass_id=None, class_id=None,
            macroclass_name=None, class_name=None, wavelength_list=None,
            standard_deviation_list=None, signature_id=None, selected=1,
            min_dist_thr=0, max_like_thr=0, spec_angle_thr=0, geometry=0,
            signature=0, color_string=None
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

        Returns:
            object OutputManger

        """
        cfg.logger.log.debug('start')
        if macroclass_id is None:
            macroclass_id = 1
        if class_id is None:
            class_id = 1
        if color_string is None:
            color_string = '#ffffff'
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
            signature=signature, color_string=color_string
        )
        cfg.logger.log.debug('end')

    # sets macroclass color string
    def set_macroclass_color(self, macroclass_id, color_string):
        self.macroclasses_color_string[macroclass_id] = color_string

    # add spectral signature reference to Spectral Signatures Catalog
    def signature_to_catalog(
            self, signature_id, macroclass_id, class_id, macroclass_name=None,
            class_name=None, selected=1, min_dist_thr=0, max_like_thr=0,
            spec_angle_thr=0, geometry=0, signature=0, color_string=None
    ):
        # add signature to catalog
        self.table = tm.add_spectral_signature_to_catalog_table(
            signature_id=signature_id, macroclass_id=macroclass_id,
            class_id=class_id, class_name=class_name,
            previous_catalog=self.table, selected=selected,
            min_dist_thr=min_dist_thr, max_like_thr=max_like_thr,
            spec_angle_thr=spec_angle_thr, geometry=geometry,
            signature=signature, color_string=color_string
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
                        vector = raster_vector.get_polygon_from_vector(
                            vector_path=self.geometry_file,
                            attribute_filter="%s = '%s'" % (
                                cfg.uid_field_name, signature_id)
                        )
                        (value_list,
                         standard_deviation_list) = self.calculate_signature(
                            vector)
                        mc_value = self.table[
                            self.table['signature_id'] == signature_id
                        ].macroclass_id[0]
                        c_value = self.table[
                            self.table['signature_id'] == signature_id
                        ].class_id[0]
                        c_name = self.table[
                            self.table['signature_id'] == signature_id
                        ].class_name[0]
                        if color_string is None:
                            color_string = self.table[
                                self.table['signature_id'] == signature_id
                            ].color_string[0]
                        mc_name = self.macroclasses[mc_value]
                        self.add_spectral_signature(
                            value_list=value_list, macroclass_id=mc_value,
                            class_id=c_value, macroclass_name=mc_name,
                            class_name=c_name,
                            standard_deviation_list=standard_deviation_list,
                            signature_id=signature_id, geometry=1, signature=1,
                            color_string=color_string
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
                str(err)
        if macroclass_id is not None:
            macroclass_value = macroclass_value
        if macroclass_name is None:
            macroclass_name = self.macroclasses[macroclass_value]
        if class_id is not None:
            class_value = class_id
        if class_name is None:
            class_name = 'merged'
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
                class_name=class_name, calculate_signature=calculate_signature
            )
        # merge signatures if not geometry
        else:
            wavelength = None
            value_arrays = []
            std_arrays = []
            for signature in signature_ids:
                value_arrays.append(self.signatures[signature].value)
                std_arrays.append(
                    self.signatures[signature].standard_deviation
                )
                wavelength = self.signatures[signature].wavelength
            wavelength_list = wavelength.tolist()
            values = np.column_stack(value_arrays)
            stds = np.column_stack(std_arrays)
            values_mean = np.mean(values, axis=1)
            stds_squared = np.square(stds)
            stds_squared_sum = np.sum(stds_squared, axis=1)
            stds_variance = np.divide(stds_squared_sum, stds_squared.shape[1])
            stds_mean = np.sqrt(stds_variance)
            if color_string is None:
                color_string = '#ffffff'
            self.add_spectral_signature(
                value_list=values_mean.tolist(),
                wavelength_list=wavelength_list,
                standard_deviation_list=stds_mean.tolist(),
                macroclass_id=macroclass_value, class_id=class_value,
                macroclass_name=macroclass_name, class_name='merged',
                geometry=0, signature=1, color_string=color_string
            )
        cfg.logger.log.debug('end')

    # import spectral signature csv to Spectral Signatures Catalog
    def import_spectral_signature_csv(
            self, csv_path, macroclass_id=None, class_id=None,
            macroclass_name=None, class_name=None, separator=',',
            color_string='#ffffff'
    ):
        cfg.logger.log.debug('start')
        # import csv as comma separated with fields value, wavelength,
        # standard_deviation (optional)
        if files_directories.is_file(csv_path):
            csv = tm.open_file(file_path=csv_path, separators=separator,
                               field_names=['value', 'wavelength',
                                            'standard_deviation'],
                               progress_message=False, skip_first_line=False
                               )
            bandset_wavelength = self.bandset.bands['wavelength']
            value_list = []
            wavelength_list = []
            standard_deviation_list = []
            if 'wavelength' in tm.columns(csv):
                for b in bandset_wavelength.tolist():
                    arg_min = np.abs(bandset_wavelength - b).argmin()
                    wavelength_list.append(bandset_wavelength[arg_min])
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
            self.add_spectral_signature(
                value_list=value_list, macroclass_id=macroclass_id,
                class_id=class_id,
                macroclass_name=macroclass_name, class_name=class_name,
                wavelength_list=wavelength_list,
                standard_deviation_list=standard_deviation_list,
                geometry=0, signature=1, color_string=color_string
            )
            cfg.logger.log.debug('end; imported: %s' % csv_path)
        else:
            cfg.logger.log.error('error file not found: %s' % csv_path)
            cfg.messages.error('error file not found: %s' % csv_path)

    # import vector to Spectral Signatures Catalog
    def import_vector(
            self, file_path, macroclass_value=None, class_value=None,
            macroclass_name=None, class_name=None, macroclass_field=None,
            class_field=None, macroclass_name_field=None,
            class_name_field=None, calculate_signature=True,
            color_string='#ffffff'
    ):
        cfg.logger.log.debug('start')
        if files_directories.is_file(file_path):
            # check geometry vector
            if not files_directories.is_file(self.geometry_file):
                if self.bandset is None:
                    raise Exception('bandset not found')
                crs = self.bandset.crs
                raster_vector.create_geometry_vector(
                    output_path=self.geometry_file, crs_wkt=crs,
                    macroclass_field_name=self.macroclass_field,
                    class_field_name=self.class_field
                )
            # get vector crs
            vector_crs = raster_vector.get_crs(file_path)
            # check crs
            catalog_sr = osr.SpatialReference()
            catalog_sr.ImportFromWkt(self.bandset.crs)
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
                    o_feature.SetField(cfg.macroclass_field_name,
                                       int(mc_value))
                    catalog_layer.CreateFeature(o_feature)
                    if calculate_signature:
                        temp_path = cfg.temp.temporary_file_path(
                            name_suffix=cfg.gpkg_suffix
                            )
                        raster_vector.create_geometry_vector(
                            output_path=temp_path, crs_wkt=self.bandset.crs,
                            macroclass_field_name=self.macroclass_field,
                            class_field_name=self.class_field
                        )
                        temp_vector = ogr.Open(temp_path, 1)
                        temp_layer = temp_vector.GetLayer()
                        temp_layer.CreateFeature(o_feature)
                        temp_vector.Destroy()
                        (value_list,
                         standard_deviation_list) = self.calculate_signature(
                            temp_path)
                        self.add_spectral_signature(
                            value_list=value_list, macroclass_id=mc_value,
                            class_id=c_value, macroclass_name=mc_name,
                            class_name=c_name,
                            standard_deviation_list=standard_deviation_list,
                            signature_id=signature_id, geometry=1, signature=1,
                            color_string=color_string
                        )
                    else:
                        self.signature_to_catalog(
                            signature_id=signature_id, macroclass_id=mc_value,
                            class_id=c_value, macroclass_name=mc_name,
                            class_name=c_name, geometry=1, signature=0,
                            color_string=color_string
                        )
                    o_feature.Destroy()
                    i_feature.Destroy()
                    i_feature = i_layer.GetNextFeature()
                else:
                    # close files
                    i_vector.Destroy()
                    catalog_vector.Destroy()
                    cfg.logger.log.error('cancel')
            # close files
            i_vector.Destroy()
            catalog_vector.Destroy()
            cfg.logger.log.debug('end; imported: %s' % file_path)
        else:
            cfg.logger.log.error('error file not found: %s' % file_path)
            cfg.messages.error('error file not found: %s' % file_path)

    # import vector to Spectral Signatures Catalog
    def calculate_signature(self, roi_path, n_processes: int = None):
        if n_processes is None:
            n_processes = cfg.n_processes
        _temp_vector = ogr.Open(roi_path)
        _temp_layer = _temp_vector.GetLayer()
        min_x, max_x, min_y, max_y = _temp_layer.GetExtent()
        _temp_layer = None
        _temp_vector = None
        path_list = self.bandset.get_absolute_paths()
        virtual_path_list = []
        for p in path_list:
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
        cfg.multiprocess.multiprocess_spectral_signature()
        value_list, standard_deviation_list = cfg.multiprocess.output
        return value_list, standard_deviation_list

    # save Spectral Signatures Catalog to file
    def save(self, output_path):
        # file list
        file_list = []
        # create temporary directory
        temp_dir = cfg.temp.create_temporary_directory()
        # geometry file
        if files_directories.is_file(self.geometry_file):
            files_directories.copy_file(
                self.geometry_file, '%s/geometry.gpkg' % temp_dir
            )
            file_list.append('%s/geometry.gpkg' % temp_dir)
        # create xml file
        root = cElementTree.Element('macroclasses')
        root.set('version', str(cfg.version))
        root.set('macroclass_field', str(self.macroclass_field))
        root.set('class_field', str(self.class_field))
        if self.macroclasses is not None:
            for macroclass in self.macroclasses:
                macroclass_element = cElementTree.SubElement(
                    root, 'macroclass'
                )
                macroclass_element.set('id', str(macroclass))
                macroclass_element.set('name',
                                       str(self.macroclasses[macroclass]))
                macroclass_element.set(
                    'color', str(self.macroclasses_color_string[macroclass])
                )
        if self.signatures is not None:
            for signature in self.signatures:
                # create file inside temporary directory
                self.signatures[signature].tofile(file='%s/%s' %
                                                       (temp_dir, signature))
                file_list.append('%s/%s' % (temp_dir, signature))
        # save to file
        pretty_xml = minidom.parseString(
            cElementTree.tostring(root)).toprettyxml()
        # create file inside temporary directory
        read_write_files.write_file(pretty_xml,
                                    '%s/macroclasses.xml' % temp_dir)
        file_list.append('%s/macroclasses.xml' % temp_dir)
        # create file inside temporary directory
        if self.table is not None:
            self.table.tofile(file='%s/table' % temp_dir)
            file_list.append('%s/table' % temp_dir)
        # zip files
        files_directories.zip_files(file_list, output_path,
                                    compression=zipfile.ZIP_STORED)
        cfg.logger.log.debug('export signature catalog')
        return output_path

    # load Spectral Signatures Catalog from file
    def load(self, file_path):
        self.table = None
        self.signatures = {}
        # create temporary directory
        temp_dir = cfg.temp.create_temporary_directory()
        file_list = files_directories.unzip_file(file_path, temp_dir)
        for f in file_list:
            f_name = files_directories.file_name(f, suffix=True)
            if f_name == 'geometry.gpkg':
                self.geometry_file = f
            elif f_name == 'table':
                self.table = np.core.records.fromfile(
                    f, dtype=cfg.spectral_dtype_list)
            elif f_name == 'macroclasses.xml':
                tree = cElementTree.parse(f)
                root = tree.getroot()
                version = root.get('version')
                if version is None:
                    cfg.logger.log.error('failed loading signatures: %s'
                                         % file_path)
                    cfg.messages.error('failed loading signatures: %s'
                                       % file_path)
                else:
                    self.macroclass_field = root.get('macroclass_field')
                    self.class_field = root.get('class_field')
                    self.macroclasses = {}
                    self.macroclasses_color_string = {}
                    for child in root:
                        macroclass_id = child.get('id')
                        macroclass_name = child.get('name')
                        macroclass_color = child.get('color')
                        self.macroclasses[int(macroclass_id)] = str(
                            macroclass_name)
                        self.macroclasses_color_string[
                            int(macroclass_id)] = str(macroclass_color)
            else:
                self.signatures[f_name] = np.core.records.fromfile(
                    f, dtype=cfg.signature_dtype_list)


# generate signature id
def generate_signature_id():
    times = dates_times.get_time_string()
    r = str(random.randint(0, 1000))
    uid = 's%s_%s' % (times, r)
    return uid
