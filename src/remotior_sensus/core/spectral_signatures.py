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

import numpy as np

from remotior_sensus.core import (
    configurations as cfg, table_manager as tm
)
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.processor_functions import spectral_signature
from remotior_sensus.util import raster_vector, dates_times, files_directories

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
        # spectral signatures catalog table
        self.table = catalog_table
        # dictionary of spectral signature tables
        self.signatures = {}
        # dictionary of names of macroclasses
        self.macroclasses = {}
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

    # add spectral signature to Spectral Signatures Catalog
    def add_spectral_signature(
            self, value_list, macroclass_id=None, class_id=None,
            macroclass_name=None, class_name=None, wavelength_list=None,
            standard_deviation_list=None, signature_id=None, selected=1,
            min_dist_thr=0, max_like_thr=0, spec_angle_thr=0
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

        Returns:
            object OutputManger

        """
        if macroclass_id is None:
            macroclass_id = 1
        if class_id is None:
            class_id = 1
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
            spec_angle_thr=spec_angle_thr
        )

    # add spectral signature reference to Spectral Signatures Catalog
    def signature_to_catalog(
            self, signature_id, macroclass_id, class_id, macroclass_name=None,
            class_name=None, selected=1, min_dist_thr=0, max_like_thr=0,
            spec_angle_thr=0
    ):
        # add signature to catalog
        self.table = tm.add_spectral_signature_to_catalog_table(
            signature_id=signature_id, macroclass_id=macroclass_id,
            class_id=class_id, class_name=class_name,
            previous_catalog=self.table, selected=selected,
            min_dist_thr=min_dist_thr, max_like_thr=max_like_thr,
            spec_angle_thr=spec_angle_thr
        )
        # add or update macroclass name
        if macroclass_name is not None:
            self.macroclasses[macroclass_id] = str(macroclass_name)
        # check macroclass name
        if macroclass_id not in self.macroclasses:
            self.macroclasses[macroclass_id] = '%s%s' % (
                cfg.macroclass_default, str(len(self.macroclasses) + 1))

    # import spectral signature csv to Spectral Signatures Catalog
    def import_spectral_signature_csv(
            self, csv_path, macroclass_id=None, class_id=None,
            macroclass_name=None,
            class_name=None, separator=','
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
                standard_deviation_list=standard_deviation_list
            )
            cfg.logger.log.debug('end; imported: %s' % csv_path)
        else:
            cfg.logger.log.error('error file not found: %s' % csv_path)
            cfg.messages.error('error file not found: %s' % csv_path)

    # import vector to Spectral Signatures Catalog
    def import_vector(
            self, file_path, macroclass_field, class_field,
            macroclass_name_field, class_name_field,
            calculate_signature=True
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
                    macroclass_value = i_feature.GetField(macroclass_field)
                    class_value = i_feature.GetField(class_field)
                    macroclass_name = i_feature.GetField(macroclass_name_field)
                    class_name = i_feature.GetField(class_name_field)
                    o_feature.SetField(cfg.macroclass_field_name,
                                       macroclass_value)
                    o_feature.SetField(cfg.class_field_name, class_value)
                    o_feature.SetField(cfg.uid_field_name, signature_id)
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
                            value_list=value_list,
                            macroclass_id=macroclass_value,
                            class_id=class_value,
                            macroclass_name=macroclass_name,
                            class_name=class_name,
                            standard_deviation_list=standard_deviation_list,
                            signature_id=signature_id
                        )
                    else:
                        self.signature_to_catalog(
                            signature_id=signature_id,
                            macroclass_id=macroclass_value,
                            class_id=class_value,
                            macroclass_name=macroclass_name,
                            class_name=class_name
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


# generate signature id
def generate_signature_id():
    times = dates_times.get_time_string()
    r = str(random.randint(0, 1000))
    uid = 's%s_%s' % (times, r)
    return uid
