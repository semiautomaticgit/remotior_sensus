# Remotior Sensus , software to process remote sensing and GIS data.
# Copyright (C) 2022-2024 Luca Congedo.
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
Tools to manage raster writing and reading
"""

import os

import numpy as np

from remotior_sensus.core import configurations as cfg
from remotior_sensus.util import files_directories, read_write_files

try:
    if cfg.gdal_path is not None:
        os.add_dll_directory(cfg.gdal_path)
except Exception as error:
    cfg.logger.log.error(str(error))
try:
    from osgeo import ogr
    from osgeo import osr
    from osgeo import gdal_array
except Exception as error:
    cfg.logger.log.error(str(error))
try:
    from osgeo import gdal
    gdal.DontUseExceptions()
except Exception as error:
    try:
        cfg.logger.log.error(str(error))
    except Exception as error:
        str(error)


# get GDAL version
def get_gdal_version():
    v = gdal.VersionInfo('RELEASE_NAME').split('.')
    cfg.logger.log.debug('gdal version: %s' % v)
    return v


# check if file is raster or vector and return crs
def raster_or_vector_input(path):
    cfg.logger.log.debug('start')
    vector = False
    raster = False
    crs = None
    # try vector
    try:
        l_p = ogr.Open(path)
    except Exception as err:
        str(err)
        l_p = None
    # if raster
    if l_p is None:
        raster = True
        l_p = gdal.Open(path, gdal.GA_ReadOnly)
        if l_p is None:
            raster = False
        else:
            try:
                # check projections
                crs = l_p.GetProjection()
                crs = crs.replace(' ', '')
                if len(crs) == 0:
                    crs = None
            except Exception as err:
                crs = None
                raster = False
                cfg.logger.log.error(str(err))
    # if vector
    else:
        vector = True
        layer = l_p.GetLayer()
        # check projection
        proj = layer.GetSpatialRef()
        try:
            crs = proj.ExportToWkt()
            crs = crs.replace(' ', '')
            if len(crs) == 0:
                crs = None
        except Exception as err:
            vector = False
            crs = None
            cfg.logger.log.error(str(err))
    return vector, raster, crs


# number of raster bands
def get_number_bands(path):
    cfg.logger.log.debug('start')
    try:
        r_d = gdal.Open(path, gdal.GA_ReadOnly)
        number = r_d.RasterCount
    except Exception as err:
        number = None
        cfg.logger.log.error(str(err))
    return number


# get CRS of a raster or vector
def get_crs(path):
    # try vector
    try:
        l_p = ogr.Open(path)
    except Exception as err:
        str(err)
        l_p = None
    # if raster
    if l_p is None:
        l_p = gdal.Open(path, gdal.GA_ReadOnly)
        if l_p is None:
            crs = None
        else:
            try:
                # check projections
                crs = l_p.GetProjection()
                crs = crs.replace(' ', '')
                if len(crs) == 0:
                    crs = None
            except Exception as err:
                crs = None
                cfg.logger.log.error(str(err))
    # if vector
    else:
        layer = l_p.GetLayer()
        # get projection
        crs = get_layer_crs(layer)
    cfg.logger.log.debug('path: %s; crs: %s' % (path, crs))
    return crs


# set EPSG of raster
def auto_set_epsg(path):
    (gt, r_p, unit, xy_count, nd, number_of_bands, block_size,
     scale_offset, dt) = raster_info(str(path))
    # check projections
    if r_p is None:
        return None
    else:
        r_epsg = osr.SpatialReference(wkt=r_p)
        r_epsg.AutoIdentifyEPSG()
        return r_epsg


# get CRS of a vector layer
def get_layer_crs(layer):
    # check projection
    proj = layer.GetSpatialRef()
    try:
        crs = proj.ExportToWkt()
        crs = crs.replace(' ', '')
        if len(crs) == 0:
            crs = None
    except Exception as err:
        crs = None
        cfg.logger.log.error(str(err))
    return crs


# get spatial reference from wkt
def get_spatial_reference(input_projection):
    spatial_reference = osr.SpatialReference(wkt=input_projection)
    return spatial_reference


# compare two crs
def compare_crs(first_crs, second_crs):
    if cfg.logger is not None:
        cfg.logger.log.debug(
            'first_crs: %s; second_crs: %s' % (first_crs, second_crs)
        )
    try:
        first_sr = osr.SpatialReference()
        first_sr.ImportFromWkt(first_crs)
        second_sr = osr.SpatialReference()
        second_sr.ImportFromWkt(second_crs)
        if first_sr.IsSame(second_sr) == 1:
            same = True
        else:
            same = False
        if cfg.logger is not None:
            cfg.logger.log.debug('same: %s' % same)
        return same
    except Exception as err:
        if cfg.logger is not None:
            cfg.logger.log.error(str(err))
        return False


# raster information
def raster_info(path):
    if not files_directories.is_file(path):
        if cfg.logger is not None:
            cfg.logger.log.warning('raster: %s' % path)
    _r_d = gdal.Open(path, gdal.GA_ReadOnly)
    if _r_d is None:
        if cfg.logger is not None:
            cfg.logger.log.error('raster: %s' % path)
        return False
    # x pixel count
    x_count = _r_d.RasterXSize
    # y pixel count
    y_count = _r_d.RasterYSize
    # geo transformation
    gt = _r_d.GetGeoTransform()
    _band = _r_d.GetRasterBand(1)
    # nodata value
    nd = _band.GetNoDataValue()
    # offset and scale
    offset = _band.GetOffset()
    scale = _band.GetScale()
    data_type = gdal.GetDataTypeName(_band.DataType)
    unit = None
    # crs
    try:
        crs = _r_d.GetProjection()
        if len(crs) == 0:
            crs = None
        else:
            crs_sr = osr.SpatialReference(wkt=crs)
            if crs_sr.IsProjected:
                unit = crs_sr.GetAttrValue('unit')
    except Exception as err:
        crs = None
        if cfg.logger is not None:
            cfg.logger.log.error(str(err))
    # band number and block size
    number_of_bands = _r_d.RasterCount
    block_size = _band.GetBlockSize()
    info = [gt, crs, unit, [x_count, y_count], nd, number_of_bands, block_size,
            [scale, offset], data_type]
    _band = None
    _r_d = None
    try:
        cfg.logger.log.debug(
            '{} :{}'.format(
                path, [gt, unit, [x_count, y_count], nd, number_of_bands,
                       block_size, [scale, offset], data_type, crs]
            )
        )
    except Exception as err:
        str(err)
    return info


# raster no data value
def raster_nodata_value(path):
    r_d = gdal.Open(path, gdal.GA_ReadOnly)
    band = r_d.GetRasterBand(1)
    nd = band.GetNoDataValue()
    return nd


# get image geotransformation
def image_geotransformation(path):
    cfg.logger.log.debug('path: %s' % path)
    # raster extent and pixel size
    (gt, r_p, unit, xy_count, nd, number_of_bands, block_size, scale_offset,
     data_type) = raster_info(path)
    left = gt[0]
    top = gt[3]
    p_x = gt[1]
    p_y = abs(gt[5])
    right = gt[0] + gt[1] * xy_count[0]
    bottom = gt[3] + gt[5] * xy_count[1]
    c_rsr = osr.SpatialReference(wkt=r_p)
    if c_rsr.IsProjected:
        un = c_rsr.GetAttrValue('unit')
    else:
        un = None
    cfg.logger.log.debug(
        'left: %s; right: %s; top: %s; bottom: %s; p_x: %s; p_y: %s; r_p: '
        '%s; un: %s'
        % (left, top, right, bottom, p_x, p_y, r_p.replace(' ', ''), un)
    )
    info = {
        'left': left, 'top': top, 'right': right, 'bottom': bottom,
        'pixel_size_x': p_x, 'pixel_size_y': p_y,
        'projection': r_p.replace(' ', ''), 'unit': un
    }
    return info


# create raster from a reference raster
def create_raster_from_reference(
        path, band_number, output_raster_list, nodata_value=None,
        driver='GTiff', gdal_format='Float32', compress=False,
        compress_format='DEFLATE21', projection=None, geo_transform=None,
        constant_value=None, x_size=None,
        y_size=None, scale=None, offset=None
):
    if cfg.logger is not None:
        cfg.logger.log.debug('path: %s' % path)
    # list of rasters with same dimensions
    if type(path) is list:
        path = path[0]
    # open input_raster with GDAL
    _gdal_raster_ref = gdal.Open(path, gdal.GA_ReadOnly)
    if gdal_format == 'Float64':
        gdal_format = gdal.GDT_Float64
    elif gdal_format == 'Float32':
        gdal_format = gdal.GDT_Float32
    elif gdal_format == 'Int32':
        gdal_format = gdal.GDT_Int32
    elif gdal_format == 'UInt32':
        gdal_format = gdal.GDT_UInt32
    elif gdal_format == 'Int16':
        gdal_format = gdal.GDT_Int16
    elif gdal_format == 'UInt16':
        gdal_format = gdal.GDT_UInt16
    elif gdal_format == 'Byte':
        gdal_format = gdal.GDT_Byte
    for o in output_raster_list:
        if driver == 'GTiff':
            if not o.lower().endswith(cfg.tif_suffix):
                o += cfg.tif_suffix
        # pixel size and origin from reference
        if projection is None:
            r_p = _gdal_raster_ref.GetProjection()
        else:
            r_p = projection
        if geo_transform is None:
            r_gt = _gdal_raster_ref.GetGeoTransform()
        else:
            r_gt = geo_transform
        t_d = gdal.GetDriverByName(driver)
        if x_size is None:
            c = _gdal_raster_ref.RasterXSize
        else:
            c = x_size
        if y_size is None:
            r = _gdal_raster_ref.RasterYSize
        else:
            r = y_size
        if not compress:
            out_raster = t_d.Create(
                o, c, r, band_number, gdal_format,
                options=['BIGTIFF=YES']
            )
        elif compress_format == 'DEFLATE21':
            out_raster = t_d.Create(
                o, c, r, band_number, gdal_format,
                options=['COMPRESS=DEFLATE', 'PREDICTOR=2',
                         'ZLEVEL=1', 'BIGTIFF=YES']
            )
        else:
            out_raster = t_d.Create(
                o, c, r, band_number, gdal_format,
                options=['COMPRESS=%s' % compress_format, 'BIGTIFF=YES']
            )
        if out_raster is None:
            if cfg.logger is not None:
                cfg.logger.log.error('out_raster None')
            return False
        # set raster projection from reference
        out_raster.SetGeoTransform(r_gt)
        out_raster.SetProjection(r_p)
        if nodata_value is not None:
            for x in range(1, band_number + 1):
                _band = out_raster.GetRasterBand(x)
                _band.SetNoDataValue(nodata_value)
                _band.Fill(nodata_value)
                _band = None
        if constant_value is not None:
            for x in range(1, band_number + 1):
                _band = out_raster.GetRasterBand(x)
                _band.Fill(constant_value)
                _band = None
        if scale is not None:
            for x in range(1, band_number + 1):
                _band = out_raster.GetRasterBand(x)
                _band.SetScale(scale)
                _band = None
        if offset is not None:
            for x in range(1, band_number + 1):
                _band = out_raster.GetRasterBand(x)
                _band.SetOffset(offset)
                _band = None
    _gdal_raster_ref = None
    if cfg.logger is not None:
        cfg.logger.log.debug('output: %s' % str(output_raster_list))
    return output_raster_list


# read a block of band as array
def read_array_block(
        gdal_band, pixel_start_column, pixel_start_row, block_columns,
        block_row, calc_data_type=None
):
    if calc_data_type is None:
        calc_data_type = np.float32
    try:
        offset = gdal_band.GetOffset()
        scale = gdal_band.GetScale()
        if offset is None:
            offset = 0.0
        if scale is None:
            scale = 1.0
    except Exception as err:
        str(err)
        offset = 0.0
        scale = 1.0
    # force calc data type
    if scale != 1:
        calc_data_type = np.float32
    offset = np.asarray(offset).astype(calc_data_type)
    scale = np.asarray(scale).astype(calc_data_type)
    cfg.logger.log.debug(
        'pixel_start_column: %s; pixel_start_row: %s; block_columns: %s; '
        'block_row: %s; scale: %s; offset: %s; calc_data_type: %s'
        % (
            pixel_start_column, pixel_start_row, block_columns, block_row,
            scale, offset, str(calc_data_type))
    )
    try:
        a = np.asarray(
            gdal_band.ReadAsArray(
                pixel_start_column, pixel_start_row, block_columns, block_row
            ) * scale + offset
        ).astype(calc_data_type)
    except Exception as err:
        cfg.logger.log.error(str(err))
        return None
    return a


# read a block of band as array
def band_read_array_block(
        gdal_band, pixel_start_column, pixel_start_row, block_columns,
        block_row, numpy_array=None
):
    cfg.logger.log.debug(
        'pixel_start_column: %s; pixel_start_row: %s; block_columns: %s; '
        'block_row: %s'
        % (pixel_start_column, pixel_start_row, block_columns, block_row)
    )
    try:
        gdal_array.BandReadAsArray(
            gdal_band, pixel_start_column, pixel_start_row, block_columns,
            block_row, buf_obj=numpy_array
        )
    except Exception as err:
        cfg.logger.log.error(str(err))
        return None


# read raster
def read_raster(raster_path, band=1, calc_data_type=None):
    (gt, r_p, unit, xy_count, nd, number_of_bands, block_size, scale_offset,
     data_type) = raster_info(raster_path)
    _r_d = gdal.Open(raster_path, gdal.GA_ReadOnly)
    r_b = _r_d.GetRasterBand(band)
    a = read_array_block(
        gdal_band=r_b, pixel_start_column=0, pixel_start_row=0,
        block_columns=xy_count[0], block_row=xy_count[1],
        calc_data_type=calc_data_type
    )
    return a


# open raster
def open_raster(raster_path, access=None):
    if access is None:
        access = gdal.GA_Update
    raster = gdal.Open(raster_path, access)
    return raster


# open vector
def open_vector(vector_path):
    vector = ogr.Open(vector_path)
    return vector


# get unique values from vector field
def get_vector_values(vector_path, field_name):
    vector = ogr.Open(vector_path)
    i_layer = vector.GetLayer()
    i_layer_name = i_layer.GetName()
    i_layer_definition = i_layer.GetLayerDefn()
    # check if field is fid
    field_index = i_layer_definition.GetFieldIndex(field_name)
    # get unique values
    if field_index == -1:
        # if fid
        values = []
        # copy not dissolved features
        i_feature = i_layer.GetNextFeature()
        while i_feature:
            if cfg.action is True:
                try:
                    i_fid = int(i_feature.GetFID())
                    values.append(i_fid)
                except Exception as err:
                    str(err)
                i_feature = i_layer.GetNextFeature()
    else:
        sql = 'SELECT DISTINCT "%s" FROM "%s"' % (field_name, i_layer_name)
        unique_values = vector.ExecuteSQL(sql, dialect='SQLITE')
        values = []
        for i, f in enumerate(unique_values):
            values.append(f.GetField(0))
        # release sql results
        vector.ReleaseResultSet(unique_values)
    return values


# write raster
def write_raster(
        raster_path, x, y, data_array, nodata_value=None, scale=None,
        offset=None
):
    o_r = gdal.Open(raster_path, gdal.GA_Update)
    x_size = o_r.RasterXSize
    y_size = o_r.RasterYSize
    b_o = o_r.GetRasterBand(1)
    try:
        # it seems a GDAL issue that if scale is float the datatype is
        # converted to Float32
        if scale is not None or offset is not None:
            data_array = np.subtract(
                np.divide(data_array, scale), offset / scale
            )
            b_o.SetScale(scale)
            b_o.SetOffset(offset)
    except Exception as err:
        cfg.logger.log.error(str(err))
    if nodata_value is not None:
        b_o.SetNoDataValue(int(nodata_value))
    try:
        data_array = data_array[::, ::, 0]
    except Exception as err:
        str(err)
    cfg.logger.log.debug(
        'x_size: %s; y_size: %s; data_array.shape: %s; x: %s; y: %s'
        % (str(x_size), str(y_size), str(data_array.shape), str(x), str(y))
    )
    b_o.WriteArray(data_array, x, y)
    b_o.FlushCache()
    return raster_path


# build band overview
def build_band_overview(input_raster_list):
    for input_raster in input_raster_list:
        try:
            _raster = gdal.Open(input_raster, gdal.GA_ReadOnly)
            _raster.BuildOverviews('NEAREST', [8, 16, 32, 64])
            _raster = None
        except Exception as err:
            cfg.logger.log.error(str(err))


# create temporary virtual raster
def create_temporary_virtual_raster(
        input_raster_list, band_number_list=None, nodata_value=None,
        relative_to_vrt=0, intersection=True, box_coordinate_list=None,
        data_type=None, pixel_size=None, override_box_coordinate_list=None,
        grid_reference=None
):
    virtual_raster = cfg.temp.temporary_file_path(name_suffix=cfg.vrt_suffix)
    cfg.logger.log.debug(
        'virtual_raster: %s; data_type: %s'
        % (virtual_raster, data_type)
    )
    create_virtual_raster(
        input_raster_list=input_raster_list, output=virtual_raster,
        band_number_list=band_number_list, nodata_value=nodata_value,
        relative_to_vrt=relative_to_vrt, intersection=intersection,
        box_coordinate_list=box_coordinate_list, data_type=data_type,
        pixel_size=pixel_size,
        override_box_coordinate_list=override_box_coordinate_list,
        grid_reference=grid_reference
    )
    return virtual_raster


# multiband virtual raster creation
def create_virtual_raster(
        output, input_raster_list=None, band_number_list=None,
        nodata_value=None, intersection=False, relative_to_vrt=0,
        pixel_size=None, data_type=None, scale_offset_list=None,
        box_coordinate_list=None, override_box_coordinate_list=False,
        grid_reference=None, relative_extent_list=None, bandset=None,
):
    """
    :param output: output path string
    :param input_raster_list: list of raster paths
    :param band_number_list: list of lists of band numbers for each input
        raster, if None all available bands are added
    :param nodata_value: nodata value
    :param intersection: if True get minimum extent from input intersection,
        if False get maximum extent from union
    :param relative_to_vrt: integer 0 or 1, relative path of input rasters
        to vrt if 1, absolute path if 0
    :param pixel_size: optional list of pixel size x and size y
        for virtual raster output
    :param data_type: virtual raster data type
    :param scale_offset_list: optional list of scale and offset for each raster
    :param box_coordinate_list: list of boundary coordinates left top right
        bottom
    :param override_box_coordinate_list: optional with box_coordinate_list,
        if True use these absolute coordinates
    :param grid_reference: optional path of a raster used as reference grid
    :param relative_extent_list: list of xOff, yOff, xSize, ySize for source
        and destination
    :param bandset: optional, use BandSet for input_raster_list,
        box_coordinate_list, scale_offset_list

    """

    cfg.logger.log.debug('start')
    lefts = []
    rights = []
    tops = []
    bottoms = []
    p_x_sizes = []
    p_y_sizes = []
    all_band_number_list = []
    r_p = r_epsg = x_counts = y_counts = data_types = None
    x_block_sizes = y_block_sizes = None
    offsets = scales = multiplicative_factors = additive_factors = None
    nodata_values = wavelengths = wavelength_units = None
    if bandset is None:
        # iterate input_raster
        for i in input_raster_list:
            # raster extent and pixel size
            (gt, r_p, unit, xy_count, nd, number_of_bands, block_size,
             scale_offset, dt) = raster_info(str(i))
            # check projections
            try:
                if r_p is not None:
                    # check projections
                    if r_epsg is None:
                        r_epsg = osr.SpatialReference()
                        r_epsg.ImportFromWkt(r_p)
                    else:
                        v_epsg = osr.SpatialReference()
                        v_epsg.ImportFromWkt(r_p)
                        if v_epsg.IsSame(r_epsg) != 1:
                            cfg.logger.log.error(
                                'v_epsg.IsSame(r_epsg): %s' % v_epsg.IsSame(
                                    r_epsg
                                )
                            )
                            return False
                else:
                    cfg.logger.log.error('error projection is None')
                    return False
            except Exception as err:
                cfg.logger.log.error(err)
                return False
            # set band number list
            if band_number_list is None:
                all_band_number_list.append(
                    list(range(1, number_of_bands + 1))
                )
            left = gt[0]
            top = gt[3]
            p_x_size = gt[1]
            p_y_size = abs(gt[5])
            right = gt[0] + gt[1] * xy_count[0]
            bottom = gt[3] + gt[5] * xy_count[1]
            # lists
            lefts.append(left)
            rights.append(right)
            tops.append(top)
            bottoms.append(bottom)
            p_x_sizes.append(p_x_size)
            p_y_sizes.append(p_y_size)
    else:
        input_raster_list = bandset.get_absolute_paths()
        lefts = bandset.get_band_attributes('left')
        tops = bandset.get_band_attributes('top')
        rights = bandset.get_band_attributes('right')
        bottoms = bandset.get_band_attributes('bottom')
        crs_s = bandset.get_band_attributes('crs')
        wavelengths = bandset.get_wavelengths()
        wavelength_units = bandset.get_wavelength_units()
        # check projections
        for r_p in crs_s:
            try:
                if r_p is not None:
                    # check projections
                    if r_epsg is None:
                        r_epsg = osr.SpatialReference()
                        r_epsg.ImportFromWkt(r_p)
                    else:
                        v_epsg = osr.SpatialReference()
                        v_epsg.ImportFromWkt(r_p)
                        if v_epsg.IsSame(r_epsg) != 1:
                            cfg.logger.log.error(
                                'v_epsg.IsSame(r_epsg): %s' % str(
                                    v_epsg.IsSame(r_epsg)
                                )
                            )
                            cfg.logger.log.error('error epsg')
                            return False
                else:
                    cfg.logger.log.error('error projection is None')
                    return False
            except Exception as err:
                cfg.logger.log.error(err)
                return False
        p_x_sizes = bandset.get_band_attributes('x_size')
        p_y_sizes = bandset.get_band_attributes('y_size')
        x_counts = bandset.get_band_attributes('x_count')
        y_counts = bandset.get_band_attributes('y_count')
        data_types = bandset.get_band_attributes('data_type')
        nodata_values = bandset.get_band_attributes('nodata')
        x_block_sizes = bandset.get_band_attributes('x_block_size')
        y_block_sizes = bandset.get_band_attributes('y_block_size')
        scales = bandset.get_band_attributes('scale')
        offsets = bandset.get_band_attributes('offset')
        multiplicative_factors = bandset.get_band_attributes(
            'multiplicative_factor'
        )
        additive_factors = bandset.get_band_attributes('additive_factor')
        box_coordinate_list = bandset.box_coordinate_list
        if band_number_list is None:
            band_number_list = bandset.get_raster_band_list()
        else:
            input_raster_list = [input_raster_list[i-1]
                                 for i in band_number_list]
            lefts = [lefts[i-1] for i in band_number_list]
            tops = [tops[i-1] for i in band_number_list]
            rights = [rights[i-1] for i in band_number_list]
            bottoms = [bottoms[i-1] for i in band_number_list]
            wavelengths = [wavelengths[i-1] for i in band_number_list]
            wavelength_units = [wavelength_units[i-1]
                                for i in band_number_list]
            p_x_sizes = [p_x_sizes[i-1] for i in band_number_list]
            p_y_sizes = [p_y_sizes[i-1] for i in band_number_list]
            x_counts = [x_counts[i-1] for i in band_number_list]
            y_counts = [y_counts[i-1] for i in band_number_list]
            data_types = [data_types[i-1] for i in band_number_list]
            nodata_values = [nodata_values[i-1] for i in band_number_list]
            x_block_sizes = [x_block_sizes[i-1] for i in band_number_list]
            y_block_sizes = [y_block_sizes[i-1] for i in band_number_list]
            scales = [scales[i-1] for i in band_number_list]
            offsets = [offsets[i-1] for i in band_number_list]
            multiplicative_factors = [multiplicative_factors[i-1]
                                      for i in band_number_list]
            additive_factors = [additive_factors[i-1]
                                for i in band_number_list]
            raster_band_number_list = bandset.get_raster_band_list()
            new_band_number_list = [raster_band_number_list[i-1]
                                    for i in band_number_list]
            band_number_list = new_band_number_list
    cfg.logger.log.debug(
        'lefts: %s; rights: %s; tops: %s; bottoms: %s; p_x_sizes: %s; '
        'p_y_sizes: %s' % (
            str(lefts), str(rights), str(tops), str(bottoms), str(p_x_sizes),
            str(p_y_sizes))
    )
    cfg.logger.log.debug('box_coordinate_list: %s;' % box_coordinate_list)
    # calculate boundaries
    try:
        # minimum extent of intersection
        if intersection:
            i_left = max(lefts)
            i_top = min(tops)
            i_right = min(rights)
            i_bottom = max(bottoms)
        # maximum extent of union
        else:
            i_left = min(lefts)
            i_top = max(tops)
            i_right = max(rights)
            i_bottom = min(bottoms)
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False
    cfg.logger.log.debug(
        '[i_left, i_top, i_right, i_bottom]: %s' % str(
            [i_left, i_top, i_right, i_bottom]
        )
    )
    if box_coordinate_list is None:
        if pixel_size is None:
            # minimum x and y pixel size
            pixel_x_size = min(p_x_sizes)
            pixel_y_size = min(p_y_sizes)
            input_reference = input_raster_list[p_x_sizes.index(pixel_x_size)]
            # raster extent and pixel size
            (gt, p, unit, xy_count, nd, number_of_bands, block_size,
             scale_offset, dt) = raster_info(input_reference)
            r_left = gt[0]
            r_top = gt[3]
            # find raster box
            diff_left = round((i_left - r_left) / pixel_x_size) * pixel_x_size
            i_left = r_left + diff_left
            diff_top = round((i_top - r_top) / pixel_y_size) * pixel_y_size
            i_top = r_top + diff_top
            diff_right = round(
                (i_right - i_left) / pixel_x_size
            ) * pixel_x_size
            i_right = i_left + diff_right
            diff_bottom = round(
                (i_bottom - i_top) / pixel_y_size
            ) * pixel_y_size
            i_bottom = i_top + diff_bottom
            cfg.logger.log.debug(
                '[pixel_x_size, pixel_y_size]: %s; [r_left, r_top]: %s; ['
                'i_left, i_top, i_right, i_bottom]: %s'
                % (str([pixel_x_size, pixel_y_size]), str([r_left, r_top]),
                   str([i_left, i_top, i_right, i_bottom]))
            )
        else:
            pixel_x_size, pixel_y_size = pixel_size
            if pixel_x_size is None:
                # minimum x and y pixel size
                pixel_x_size = min(p_x_sizes)
            if pixel_y_size is None:
                pixel_y_size = min(p_y_sizes)
    # box coordinates to be adapted
    else:
        # override coordinates with absolute coordinates
        if override_box_coordinate_list:
            i_left = box_coordinate_list[0]
            i_top = box_coordinate_list[1]
            i_right = box_coordinate_list[2]
            i_bottom = box_coordinate_list[3]
            # x and y pixel size
            if pixel_size is None:
                pixel_x_size = min(p_x_sizes)
                pixel_y_size = min(p_y_sizes)
            else:
                pixel_x_size, pixel_y_size = pixel_size
                if pixel_x_size is None:
                    pixel_x_size = min(p_x_sizes)
                if pixel_y_size is None:
                    pixel_y_size = min(p_y_sizes)
        # find minimum extent raster
        elif grid_reference is not None:
            # raster extent and pixel size
            (gt, p, unit, xy_count, nd, number_of_bands, block_size,
             scale_offset, dt) = raster_info(grid_reference)
            r_left = gt[0]
            r_top = gt[3]
            if pixel_size is None:
                pixel_x_size = gt[1]
                pixel_y_size = abs(gt[5])
            else:
                pixel_x_size, pixel_y_size = pixel_size
                if pixel_x_size is None:
                    pixel_x_size = gt[1]
                if pixel_y_size is None:
                    pixel_y_size = abs(gt[5])
            # find raster box
            diff_left = round(
                (box_coordinate_list[0] - r_left) / pixel_x_size
            ) * pixel_x_size
            i_left = r_left + diff_left
            diff_top = round(
                (box_coordinate_list[1] - r_top) / pixel_y_size
            ) * pixel_y_size
            i_top = r_top + diff_top
            diff_right = round(
                (box_coordinate_list[2] - i_left) / pixel_x_size
            ) * pixel_x_size
            i_right = i_left + diff_right
            diff_bottom = round(
                (box_coordinate_list[3] - i_top) / pixel_y_size
            ) * pixel_y_size
            i_bottom = i_top + diff_bottom
            cfg.logger.log.debug(
                '[pixel_x_size, pixel_y_size]: %s; [r_left, r_top]: %s; ['
                'i_left, i_top, i_right, i_bottom]: %s'
                % (str([pixel_x_size, pixel_y_size]), str([r_left, r_top]),
                   str([i_left, i_top, i_right, i_bottom]))
            )
        # intersection extent
        else:
            # x and y pixel size
            if pixel_size is None:
                pixel_x_size = min(p_x_sizes)
                pixel_y_size = min(p_y_sizes)
            else:
                pixel_x_size, pixel_y_size = pixel_size
                if pixel_x_size is None:
                    pixel_x_size = min(p_x_sizes)
                if pixel_y_size is None:
                    pixel_y_size = min(p_y_sizes)
            if pixel_x_size in p_x_sizes:
                input_reference = input_raster_list[
                    p_x_sizes.index(pixel_x_size)]
            else:
                input_reference = input_raster_list[
                    p_x_sizes.index(min(p_x_sizes))]
            # raster extent and pixel size
            (gt, p, unit, xy_count, nd, number_of_bands, block_size,
             scale_offset, dt) = raster_info(input_reference)
            r_left = gt[0]
            r_top = gt[3]
            # find raster box
            diff_left = round(
                (box_coordinate_list[0] - r_left) / pixel_x_size
            ) * pixel_x_size
            i_left = r_left + diff_left
            if i_left < r_left:
                i_left = r_left
            diff_top = round(
                (box_coordinate_list[1] - r_top) / pixel_y_size
            ) * pixel_y_size
            i_top = r_top + diff_top
            if i_top > r_top:
                i_top = r_top
            diff_right = round(
                (box_coordinate_list[2] - i_left) / pixel_x_size
            ) * pixel_x_size
            i_right = i_left + diff_right
            r_right = gt[0] + gt[1] * xy_count[0]
            if i_right > r_right:
                i_right = r_right
            diff_bottom = round(
                (box_coordinate_list[3] - i_top) / pixel_y_size
            ) * pixel_y_size
            i_bottom = i_top + diff_bottom
            r_bottom = gt[3] + gt[5] * xy_count[1]
            if i_bottom < r_bottom:
                i_bottom = r_bottom
    # create virtual raster
    drv = gdal.GetDriverByName('vrt')
    # number of pixels for x and y pixel size
    if i_right == i_left:
        i_right = i_left + pixel_x_size
    if i_top == i_bottom:
        i_top = i_bottom + pixel_y_size
    r_x = abs(int(round((i_right - i_left) / pixel_x_size)))
    r_y = abs(int(round((i_top - i_bottom) / pixel_y_size)))
    # create virtual raster
    files_directories.create_parent_directory(output)
    v_rast = drv.Create(output, r_x, r_y, 0)
    v_rast.SetGeoTransform((i_left, pixel_x_size, 0, i_top, 0, -pixel_y_size))
    v_rast.SetProjection(r_p)
    # set band number list
    if band_number_list is None:
        band_number_list = all_band_number_list
    cfg.logger.log.debug(
        '[r_x, r_y]: %s; band_number_list: %s' % (
            str([r_x, r_y]), str(band_number_list))
    )
    n = 0
    # iterate bands
    for raster in range(len(input_raster_list)):
        for band_number in band_number_list[raster]:
            n += 1
            cfg.logger.log.debug(
                'input_raster: %s' % str(input_raster_list[raster])
            )
            if bandset is None:
                # open input_raster
                input_raster = gdal.Open(
                    input_raster_list[raster], gdal.GA_ReadOnly
                )
                ir_x_size = input_raster.RasterXSize
                ir_y_size = input_raster.RasterYSize
                input_band = input_raster.GetRasterBand(band_number)
                # data type
                try:
                    data_type_s = gdal.GetDataTypeName(input_band.DataType)
                except Exception as err:
                    str(err)
                    data_type_s = cfg.float32_dt
                if data_type is not None:
                    try:
                        gdal_format = eval('gdal.GDT_%s' % data_type)
                    except Exception as err:
                        str(err)
                        gdal_format = input_band.DataType
                else:
                    gdal_format = input_band.DataType
                # add virtual band
                v_rast.AddBand(gdal_format)
                band = v_rast.GetRasterBand(n)
                # block size
                bsize = input_band.GetBlockSize()
                x_block = bsize[0]
                y_block = bsize[1]
                if scale_offset_list is None:
                    offs = input_band.GetOffset()
                    scl = input_band.GetScale()
                else:
                    scl, offs = scale_offset_list[raster]
                # get nodata value
                no_data = input_band.GetNoDataValue()
                # get input_raster geotransformation
                gt = input_raster.GetGeoTransform()
                p_x_size = abs(gt[1])
                p_y_size = abs(gt[5])
                left = gt[0]
                top = gt[3]
                right = gt[0] + gt[1] * ir_x_size
                bottom = gt[3] + gt[5] * ir_y_size
                # set source parameters
                if i_left <= left:
                    s_off_x = 0
                else:
                    s_off_x = round((i_left - left) / p_x_size)
                if i_top >= top:
                    s_off_y = 0
                else:
                    s_off_y = round((top - i_top) / p_y_size)
                if i_right >= right:
                    s_r_x = ir_x_size - s_off_x
                else:
                    s_r_x = ir_x_size - s_off_x - abs(
                        round((right - i_right) / p_x_size)
                    )
                if i_bottom <= bottom:
                    s_r_y = ir_y_size - s_off_y
                else:
                    s_r_y = ir_y_size - s_off_y - abs(
                        round((i_bottom - bottom) / p_y_size)
                    )
                # set destination parameters
                if i_left < left:
                    d_off_x = abs(round((left - i_left) / pixel_x_size))
                else:
                    d_off_x = 0
                if i_top > top:
                    d_off_y = abs(round((i_top - top) / pixel_y_size))
                else:
                    d_off_y = 0
                # number of x pixels
                d_r_x = round((s_r_x * p_x_size) / pixel_x_size)
                # number of y pixels
                d_r_y = round((s_r_y * p_y_size) / pixel_y_size)
                if relative_extent_list is not None:
                    s_off_x, s_off_y, s_r_x, s_r_y = relative_extent_list
                    d_off_x, d_off_y, d_r_x, d_r_y = relative_extent_list
            # BandSet
            else:
                ir_x_size = x_counts[raster]
                ir_y_size = y_counts[raster]
                # data type
                try:
                    data_type_s = data_types[raster]
                except Exception as err:
                    str(err)
                    data_type_s = cfg.float32_dt
                if data_type is not None:
                    try:
                        gdal_format = eval('gdal.GDT_%s' % data_type)
                    except Exception as err:
                        str(err)
                        gdal_format = eval('gdal.GDT_%s' % data_types[raster])
                else:
                    gdal_format = eval('gdal.GDT_%s' % data_types[raster])
                # add virtual band
                v_rast.AddBand(gdal_format)
                band = v_rast.GetRasterBand(n)
                # block size
                x_block = x_block_sizes[raster]
                y_block = y_block_sizes[raster]
                if scale_offset_list is None:
                    if offsets[raster] == cfg.nodata_val_Int64:
                        offs = None
                    else:
                        offs = offsets[raster]
                    if scales[raster] == cfg.nodata_val_Int64:
                        scl = None
                    else:
                        scl = scales[raster]
                else:
                    scl, offs = scale_offset_list[raster]
                if scl is not None:
                    scl = multiplicative_factors[raster] * scl
                if offs is not None:
                    offs = additive_factors[raster] + offs
                # get nodata value
                no_data = nodata_values[raster]
                p_x_size = p_x_sizes[raster]
                p_y_size = p_y_sizes[raster]
                left = lefts[raster]
                top = tops[raster]
                right = rights[raster]
                bottom = bottoms[raster]
                # set source parameters
                if i_left <= left:
                    s_off_x = 0
                else:
                    s_off_x = round((i_left - left) / p_x_size)
                if i_top >= top:
                    s_off_y = 0
                else:
                    s_off_y = round((top - i_top) / p_y_size)
                if i_right >= right:
                    s_r_x = ir_x_size - s_off_x
                else:
                    s_r_x = ir_x_size - s_off_x - abs(
                        round((right - i_right) / p_x_size)
                    )
                if i_bottom <= bottom:
                    s_r_y = ir_y_size - s_off_y
                else:
                    s_r_y = ir_y_size - s_off_y - abs(
                        round((i_bottom - bottom) / p_y_size)
                    )
                # set destination parameters
                if i_left < left:
                    d_off_x = abs(round((left - i_left) / pixel_x_size))
                else:
                    d_off_x = 0
                if i_top > top:
                    d_off_y = abs(round((i_top - top) / pixel_y_size))
                else:
                    d_off_y = 0
                # number of x pixels
                d_r_x = round((s_r_x * p_x_size) / pixel_x_size)
                # number of y pixels
                d_r_y = round((s_r_y * p_y_size) / pixel_y_size)
                if relative_extent_list is not None:
                    s_off_x, s_off_y, s_r_x, s_r_y = relative_extent_list
                    d_off_x, d_off_y, d_r_x, d_r_y = relative_extent_list
            cfg.logger.log.debug(
                '[left, top, right, bottom]: %s; [s_off_x, s_off_y, s_r_x, '
                's_r_y]: %s;[d_off_x, d_off_y, d_r_x, '
                'd_r_y]: %s; data_type_s: %s; nodata_value: %s; no_data: %s'
                % (str([left, top, right, bottom]),
                   str([s_off_x, s_off_y, s_r_x, s_r_y]),
                   str([d_off_x, d_off_y, d_r_x, d_r_y]), str(data_type_s),
                   str(nodata_value), str(no_data))
            )
            try:
                # check path
                if relative_to_vrt == 1:
                    source_path = files_directories.file_name(
                        input_raster_list[raster], True
                    )
                else:
                    source_path = input_raster_list[raster].replace(
                        '//', '/'
                    ).replace('https:/', 'https://').replace(
                        'http:/', 'http://'
                    )
                # set metadata xml
                if no_data is None:
                    xml = '''
                <ComplexSource>
                    <SourceFilename relative_to_vrt="%i">%s</SourceFilename>
                    <SourceBand>%i</SourceBand>
                    <SourceProperties RasterXSize="%i" RasterYSize="%i" 
                    DataType="%s" BlockXSize="%i" BlockYSize="%i" />
                    <SrcRect xOff="%i" yOff="%i" xSize="%i" ySize="%i" />
                    <DstRect xOff="%i" yOff="%i" xSize="%i" ySize="%i" />
                </ComplexSource>
                    '''
                    source = xml % (
                        relative_to_vrt, source_path,
                        band_number, ir_x_size, ir_y_size, data_type_s,
                        x_block, y_block, s_off_x, s_off_y, s_r_x, s_r_y,
                        d_off_x, d_off_y, d_r_x, d_r_y
                    )
                else:
                    xml = '''
                <ComplexSource>
                    <SourceFilename relative_to_vrt="%i">%s</SourceFilename>
                    <SourceBand>%i</SourceBand>
                    <SourceProperties RasterXSize="%i" RasterYSize="%i" 
                    DataType="%s" BlockXSize="%i" BlockYSize="%i" />
                    <SrcRect xOff="%i" yOff="%i" xSize="%i" ySize="%i" />
                    <DstRect xOff="%i" yOff="%i" xSize="%i" ySize="%i" />
                    <NODATA>%s</NODATA>
                </ComplexSource>
                    '''
                    source = xml % (
                        relative_to_vrt, source_path, band_number,
                        ir_x_size, ir_y_size, data_type_s, x_block,
                        y_block, s_off_x, s_off_y, s_r_x, s_r_y,
                        d_off_x, d_off_y, d_r_x, d_r_y, no_data
                    )
                band.SetMetadataItem(
                    'ComplexSource', source, 'new_vrt_sources'
                )
                if wavelengths is not None:
                    if wavelength_units[raster] != cfg.no_unit:
                        band.SetDescription(
                            '%s%s' % (
                                str(wavelengths[raster]).ljust(6),
                                str(wavelength_units[raster]).replace(
                                    ' (1 E-6m)', ''
                                    )
                            )
                        )
                if nodata_value is True:
                    if no_data is not None:
                        try:
                            band.SetNoDataValue(int(nodata_value))
                        except Exception as err:
                            str(err)
                            try:
                                import math
                                if math.isnan(nodata_value):
                                    band.SetNoDataValue(int(nodata_value))
                            except Exception as err:
                                str(err)
                elif not nodata_value:
                    try:
                        band.SetNoDataValue(int(no_data))
                    except Exception as err:
                        str(err)
                else:
                    try:
                        band.SetNoDataValue(int(nodata_value))
                    except Exception as err:
                        str(err)
                        try:
                            import math
                            if math.isnan(nodata_value):
                                band.SetNoDataValue(int(nodata_value))
                        except Exception as err:
                            str(err)
                if offs is not None:
                    band.SetOffset(offs)
                if scl is not None:
                    band.SetScale(scl)
            except Exception as err:
                cfg.logger.log.error(str(err))
    cfg.logger.log.debug('end; output: %s' % str(output))
    return str(output)


# simplified virtual raster creation for mosaic
def create_virtual_raster_2_mosaic(
        input_raster_list, output, band_number_list=None, src_nodata=None,
        dst_nodata=False, relative_to_vrt=0, data_type=None,
        box_coordinate_list=None, override_box_coordinate_list=False,
        pixel_size=None, grid_reference=None, scale_offset_list=None
):
    cfg.logger.log.debug('start')
    lefts = []
    rights = []
    tops = []
    bottoms = []
    p_x_sizes = []
    p_y_sizes = []
    r_p = None
    r_epsg = None
    dt0 = None
    # iterate input_raster
    for i in input_raster_list:
        # raster extent and pixel size
        (gt, r_p, unit, xy_count, nd, number_of_bands, block_size,
         scale_offset, dt) = raster_info(i)
        # data type
        if dt0 is None:
            dt0 = dt
        # check projections
        try:
            if r_p is not None:
                # check projections
                if r_epsg is None:
                    r_epsg = osr.SpatialReference()
                    r_epsg.ImportFromWkt(r_p)
                else:
                    v_epsg = osr.SpatialReference()
                    v_epsg.ImportFromWkt(r_p)
                    if v_epsg.IsSame(r_epsg) != 1:
                        cfg.logger.log.error('error epsg')
                        return False
            else:
                cfg.logger.log.error('error projection')
                return False
        except Exception as err:
            cfg.logger.log.error(err)
            return False
        left = gt[0]
        top = gt[3]
        p_x_size = gt[1]
        p_y_size = abs(gt[5])
        right = gt[0] + gt[1] * xy_count[0]
        bottom = gt[3] + gt[5] * xy_count[1]
        # lists
        lefts.append(left)
        rights.append(right)
        tops.append(top)
        bottoms.append(bottom)
        p_x_sizes.append(p_x_size)
        p_y_sizes.append(p_y_size)
    # calculate boundaries
    try:
        # maximum extent of union
        i_left = min(lefts)
        i_top = max(tops)
        i_right = max(rights)
        i_bottom = min(bottoms)
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False
    pixel_x_size, pixel_y_size = None, None
    if box_coordinate_list is None:
        if pixel_size is None:
            # minimum x and y pixel size
            pixel_x_size = min(p_x_sizes)
            pixel_y_size = min(p_y_sizes)
            input_reference = input_raster_list[p_x_sizes.index(pixel_x_size)]
            # raster extent and pixel size
            (gt, p, unit, xy_count, nd, number_of_bands, block_size,
             scale_offset, dt) = raster_info(input_reference)
            r_left = gt[0]
            r_top = gt[3]
            # find raster box
            diff_left = round((i_left - r_left) / pixel_x_size) * pixel_x_size
            i_left = r_left + diff_left
            diff_top = round((i_top - r_top) / pixel_y_size) * pixel_y_size
            i_top = r_top + diff_top
            diff_right = round(
                (i_right - i_left) / pixel_x_size
            ) * pixel_x_size
            i_right = i_left + diff_right
            diff_bottom = round(
                (i_bottom - i_top) / pixel_y_size
            ) * pixel_y_size
            i_bottom = i_top + diff_bottom
        else:
            pixel_x_size, pixel_y_size = pixel_size
            if pixel_x_size is None:
                pixel_x_size = min(p_x_sizes)
                pixel_y_size = min(p_y_sizes)
    # box coordinates to be adapted
    else:
        # override coordinates with absolute coordinates
        if override_box_coordinate_list:
            i_left = box_coordinate_list[0]
            i_top = box_coordinate_list[1]
            i_right = box_coordinate_list[2]
            i_bottom = box_coordinate_list[3]
        # find minimum extent raster
        elif grid_reference is not None:
            # raster extent and pixel size
            (gt, p, unit, xy_count, nd, number_of_bands, block_size,
             scale_offset, dt) = raster_info(grid_reference)
            r_left = gt[0]
            r_top = gt[3]
            if pixel_size is None:
                pixel_x_size = gt[1]
                pixel_y_size = abs(gt[5])
            else:
                pixel_x_size, pixel_y_size = pixel_size
                if pixel_x_size is None:
                    pixel_x_size = gt[1]
                    pixel_y_size = abs(gt[5])
            # find raster box
            diff_left = round(
                (box_coordinate_list[0] - r_left) / pixel_x_size
            ) * pixel_x_size
            i_left = r_left + diff_left
            diff_top = round(
                (box_coordinate_list[1] - r_top) / pixel_y_size
            ) * pixel_y_size
            i_top = r_top + diff_top
            diff_right = round(
                (box_coordinate_list[2] - i_left) / pixel_x_size
            ) * pixel_x_size
            i_right = i_left + diff_right
            diff_bottom = round(
                (box_coordinate_list[3] - i_top) / pixel_y_size
            ) * pixel_y_size
            i_bottom = i_top + diff_bottom
        # intersection extent
        else:
            if i_left < box_coordinate_list[0]:
                i_left = box_coordinate_list[0]
            if i_top > box_coordinate_list[1]:
                i_top = box_coordinate_list[1]
            if i_right > box_coordinate_list[2]:
                i_right = box_coordinate_list[2]
            if i_bottom < box_coordinate_list[3]:
                i_bottom = box_coordinate_list[3]
    # create virtual raster
    drv = gdal.GetDriverByName('vrt')
    # number of pixels for x and y pixel size
    r_x = abs(int(round((i_right - i_left) / pixel_x_size)))
    r_y = abs(int(round((i_top - i_bottom) / pixel_y_size)))
    # create virtual raster
    files_directories.create_parent_directory(output)
    v_rast = drv.Create(output, r_x, r_y, 0)
    v_rast.SetGeoTransform((i_left, pixel_x_size, 0, i_top, 0, -pixel_y_size))
    v_rast.SetProjection(r_p)
    if data_type is not None:
        try:
            gdal_format = eval('gdal.GDT_%s' % data_type)
        except Exception as err:
            str(err)
            gdal_format = eval('gdal.GDT_%s' % dt0)
    else:
        gdal_format = eval('gdal.GDT_%s' % dt0)
    # add virtual band
    v_rast.AddBand(gdal_format)
    band = v_rast.GetRasterBand(1)
    # iterate bands
    for b in range(len(input_raster_list)):
        # open input_raster
        if band_number_list is None:
            band_number = 1
        else:
            band_number = band_number_list[b]
        # open input_raster
        input_raster = gdal.Open(input_raster_list[b], gdal.GA_ReadOnly)
        input_band = input_raster.GetRasterBand(band_number)
        if scale_offset_list is None:
            offs = input_band.GetOffset()
            scl = input_band.GetScale()
        else:
            scl, offs = scale_offset_list[b]
        # get nodata value
        no_data = input_band.GetNoDataValue()
        i_r_x = input_raster.RasterXSize
        i_r_y = input_raster.RasterYSize
        # data type
        try:
            data_type_s = gdal.GetDataTypeName(input_band.DataType)
        except Exception as err:
            str(err)
            data_type_s = cfg.float32_dt
        if src_nodata is None:
            src_nodata = no_data
        # block size
        bsize = input_band.GetBlockSize()
        x_block = bsize[0]
        y_block = bsize[1]
        # get input_raster geotransformation
        gt = input_raster.GetGeoTransform()
        p_x_size = abs(gt[1])
        p_y_size = abs(gt[5])
        left = gt[0]
        top = gt[3]
        right = gt[0] + gt[1] * i_r_x
        bottom = gt[3] + gt[5] * i_r_y
        # set source parameters
        if i_left <= left:
            s_off_x = 0
        else:
            s_off_x = round((i_left - left) / p_x_size) * p_x_size
        if i_top >= top:
            s_off_y = 0
        else:
            s_off_y = round((i_top - top) / p_y_size) * p_y_size
        if i_right >= right:
            s_r_x = i_r_x - s_off_x
        else:
            s_r_x = i_r_x - s_off_x - abs(
                round((right - i_right) / p_x_size) * p_x_size
            )
        if i_bottom <= bottom:
            s_r_y = i_r_y - s_off_y
        else:
            s_r_y = i_r_y - s_off_y - abs(
                round((i_bottom - bottom) / p_y_size) * p_y_size
            )
        # set destination parameters
        if i_left < left:
            d_off_x = abs(round((left - i_left) / pixel_x_size))
        else:
            d_off_x = 0
        if i_top > top:
            d_off_y = abs(round((i_top - top) / pixel_y_size))
        else:
            d_off_y = 0
        # number of x pixels
        d_r_x = round((s_r_x * p_x_size) / pixel_x_size)
        # number of y pixels
        d_r_y = round((s_r_y * p_y_size) / pixel_y_size)
        try:
            # check path
            if relative_to_vrt == 1:
                source_path = files_directories.file_name(
                    input_raster_list[b], True
                )
            else:
                source_path = input_raster_list[b].replace('//', '/')
            # set metadata xml
            if src_nodata is None:
                xml = '''
                <ComplexSource>
                    <SourceFilename relative_to_vrt="%i">%s</SourceFilename>
                    <SourceBand>%i</SourceBand>
                    <SourceProperties RasterXSize="%i" RasterYSize="%i" 
                    DataType="%s" BlockXSize="%i" BlockYSize="%i" />
                    <SrcRect xOff="%i" yOff="%i" xSize="%i" ySize="%i" />
                    <DstRect xOff="%i" yOff="%i" xSize="%i" ySize="%i" />
                </ComplexSource>
                '''
                source = xml % (relative_to_vrt, source_path, band_number,
                                i_r_x, i_r_y, data_type_s, x_block, y_block,
                                s_off_x, s_off_y, s_r_x, s_r_y,
                                d_off_x, d_off_y, d_r_x, d_r_y)
                band.SetMetadataItem(
                    'ComplexSource', source, 'new_vrt_sources'
                )
            else:
                xml = '''
                <ComplexSource>
                    <SourceFilename relative_to_vrt="%i">%s</SourceFilename>
                    <SourceBand>%i</SourceBand>
                    <SourceProperties RasterXSize="%i" RasterYSize="%i" 
                    DataType="%s" BlockXSize="%i" BlockYSize="%i" />
                    <SrcRect xOff="%i" yOff="%i" xSize="%i" ySize="%i" />
                    <DstRect xOff="%i" yOff="%i" xSize="%i" ySize="%i" />
                    <NODATA>%s</NODATA>
                </ComplexSource>
                '''
                source = xml % (relative_to_vrt, source_path, band_number,
                                i_r_x, i_r_y, data_type_s, x_block, y_block,
                                s_off_x, s_off_y, s_r_x, s_r_y,
                                d_off_x, d_off_y, d_r_x, d_r_y, src_nodata)
                band.SetMetadataItem(
                    'ComplexSource', source, 'new_vrt_sources'
                )
            cfg.logger.log.debug(
                'dst_nodata: %s; src_nodata: %s; no_data: %s'
                % (str(dst_nodata),  str(src_nodata), str(no_data))
            )
            if dst_nodata is True:
                try:
                    band.SetNoDataValue(int(src_nodata))
                except Exception as err:
                    str(err)
                    try:
                        import math
                        if math.isnan(src_nodata):
                            band.SetNoDataValue(int(src_nodata))
                    except Exception as err:
                        str(err)
            elif dst_nodata is False:
                pass
            else:
                try:
                    band.SetNoDataValue(int(dst_nodata))
                except Exception as err:
                    str(err)
                    try:
                        import math
                        if math.isnan(dst_nodata):
                            band.SetNoDataValue(int(dst_nodata))
                    except Exception as err:
                        str(err)
                        try:
                            band.SetNoDataValue(int(no_data))
                        except Exception as err:
                            str(err)
                            try:
                                import math
                                if math.isnan(no_data):
                                    band.SetNoDataValue(int(no_data))
                                else:
                                    cfg.logger.log.error(str(err))
                            except Exception as err:
                                cfg.logger.log.error(str(err))
            if offs is not None:
                band.SetOffset(offs)
            if scl is not None:
                band.SetScale(scl)
        except Exception as err:
            cfg.logger.log.error(str(err))
    cfg.logger.log.debug('end; virtual raster: %s' % str(output))
    return str(output)


# overwrite text to force relative to vrt
def force_relative_to_vrt(file_path):
    text = read_write_files.open_text_file(file_path)
    text = text.replace('relativeToVRT="0"', 'relativeToVRT="1"')
    read_write_files.write_file(text, file_path)
    return file_path


# get spatial reference from EPSG code o wkt
def get_projection_from_epsg_wkt(input_epsg):
    cfg.logger.log.debug('input_epsg: %s' % str(input_epsg))
    input_sr = osr.SpatialReference()
    try:
        input_sr.ImportFromEPSG(input_epsg)
    except Exception as err:
        str(err)
        try:
            input_sr.ImportFromWkt(input_epsg)
        except Exception as err:
            str(err)
            cfg.logger.log.error(str(err))
            input_sr = None
    return input_sr


# project point coordinates
def project_point_coordinates(
        point_x, point_y, input_coordinates, output_coordinates
):
    cfg.logger.log.debug('start')
    try:
        # required by GDAL 3 coordinate order
        try:
            input_coordinates.SetAxisMappingStrategy(
                osr.OAMS_TRADITIONAL_GIS_ORDER
            )
        except Exception as err:
            str(err)
        try:
            output_coordinates.SetAxisMappingStrategy(
                osr.OAMS_TRADITIONAL_GIS_ORDER
            )
        except Exception as err:
            str(err)
        # coordinate transformation
        c_t = osr.CoordinateTransformation(
            input_coordinates, output_coordinates
        )
        point_t = ogr.Geometry(ogr.wkbPoint)
        point_t.AddPoint(point_x, point_y)
        point_t.Transform(c_t)
        cfg.logger.log.debug(
            '[point_t.GetX(), point_t.GetY()]: %s' % str(
                [point_t.GetX(), point_t.GetY()]
            )
        )
        return [point_t.GetX(), point_t.GetY()]
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False


# reproject vector
def reproject_vector(
        input_vector, output, input_epsg=None, output_epsg=None,
        vector_type='wkbMultiPolygon', output_drive=None
):
    if cfg.logger is not None:
        cfg.logger.log.debug('start')
    # input spatial reference
    input_sr = osr.SpatialReference()
    if input_epsg is None:
        l_p = ogr.Open(input_vector)
        layer = l_p.GetLayer()
        proj = layer.GetSpatialRef()
        try:
            crs = proj.ExportToWkt()
            crs_string = crs.replace(' ', '')
            if len(crs_string) == 0:
                if cfg.logger is not None:
                    cfg.logger.log.error('Error input vector')
                return False
            input_sr.ImportFromWkt(crs_string)
        except Exception as err:
            if cfg.logger is not None:
                cfg.logger.log.error(str(err))
            return False
    else:
        # input EPSG or projection
        try:
            input_sr.ImportFromEPSG(input_epsg)
        except Exception as err:
            str(err)
            try:
                input_sr.ImportFromWkt(input_epsg)
            except Exception as err:
                str(err)
                input_sr = input_epsg
    # output spatial reference
    output_sr = osr.SpatialReference()
    try:
        output_sr.ImportFromEPSG(output_epsg)
    except Exception as err:
        str(err)
        try:
            output_sr.ImportFromWkt(output_epsg)
        except Exception as err:
            str(err)
            output_sr = output_epsg
    # required by GDAL 3 coordinate order
    try:
        input_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception as err:
        str(err)
    try:
        output_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception as err:
        str(err)
    # coordinate transformation
    c_t = osr.CoordinateTransformation(input_sr, output_sr)
    # open input vector
    i_vector = ogr.Open(input_vector)
    i_layer = i_vector.GetLayer()
    # create output vector
    if output_drive is None:
        if files_directories.file_extension(
                output, lower=True
        ) == cfg.shp_suffix:
            output_drive = 'ESRI Shapefile'
        else:
            output_drive = 'GPKG'
    i_driver = ogr.GetDriverByName(output_drive)
    o_source = i_driver.CreateDataSource(output)
    name = files_directories.file_name(output, suffix=False)
    if vector_type == 'wkbMultiPolygon':
        o_layer = o_source.CreateLayer(name, output_sr, ogr.wkbMultiPolygon)
    elif vector_type == 'wkbPoint':
        o_layer = o_source.CreateLayer(name, output_sr, ogr.wkbPoint)
    else:
        if cfg.logger is not None:
            cfg.logger.log.error('Error vector type')
        return False
    i_layer_definition = i_layer.GetLayerDefn()
    # copy fields
    for i in range(i_layer_definition.GetFieldCount()):
        f_definition = i_layer_definition.GetFieldDefn(i)
        o_layer.CreateField(f_definition)
    o_layer_definition = o_layer.GetLayerDefn()
    # field_value count
    o_field_count = o_layer_definition.GetFieldCount()
    # iterate input features
    i_feature = i_layer.GetNextFeature()
    while i_feature:
        if cfg.action is True:
            # get geometry
            geom = i_feature.GetGeometryRef()
            # project feature
            geom.Transform(c_t)
            o_feature = ogr.Feature(o_layer_definition)
            o_feature.SetGeometry(geom)
            for i in range(o_field_count):
                field_name = o_layer_definition.GetFieldDefn(i).GetNameRef()
                field_value = i_feature.GetField(i)
                o_feature.SetField(field_name, field_value)
            o_layer.CreateFeature(o_feature)
            o_feature.Destroy()
            i_feature.Destroy()
            i_feature = i_layer.GetNextFeature()
        else:
            cfg.logger.log.error('cancel')
            # close files
            i_vector.Destroy()
            o_source.Destroy()
            return None
    # close files
    i_vector.Destroy()
    o_source.Destroy()
    if cfg.logger is not None:
        cfg.logger.log.debug('output: %s' % output)
    return output


# get layer extent
def get_layer_extent(layer_path):
    _temp_vector = ogr.Open(layer_path)
    _temp_layer = _temp_vector.GetLayer()
    feature_count = _temp_layer.GetFeatureCount()
    if feature_count > 0:
        min_x, max_x, min_y, max_y = _temp_layer.GetExtent()
    else:
        min_x = max_x = min_y = max_y = None
    _temp_layer = None
    _temp_vector = None
    return min_x, max_x, min_y, max_y


# convert reference layer to raster based on the resolution of a raster
def vector_to_raster(
        vector_path=None, output_path=None, field_name=None,
        reference_raster_path=None, all_touched=False,
        area_based=False, output_format='GTiff', burn_values=None,
        attribute_filter=None,
        extent=None, nodata_value=0, x_y_size: list = None,
        background_value=0, compress=False, compress_format='DEFLATE21',
        vector_layer=None, available_ram=None
):
    if cfg.logger is not None:
        cfg.logger.log.debug('start')
    # GDAL config
    try:
        if available_ram is None:
            available_ram = cfg.available_ram
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(int(available_ram)
                                                  * 1000000))
        gdal.SetConfigOption('VSI_CACHE', 'FALSE')
    except Exception as err:
        str(err)
    if vector_path is not None:
        vector_crs = get_crs(vector_path)
    elif vector_layer is not None:
        vector_crs = get_layer_crs(vector_layer)
    else:
        if cfg.logger is not None:
            cfg.logger.log.error('input vector')
        return False
    (gt, reference_crs, unit, xy_count, nd, number_of_bands, block_size,
     scale_offset, data_type) = raster_info(reference_raster_path)
    orig_x = gt[0]
    orig_y = gt[3]
    if x_y_size is not None:
        x_size = x_y_size[0]
        y_size = x_y_size[1]
    else:
        x_size = gt[1]
        y_size = abs(gt[5])
    if area_based is True:
        x_size = x_size / 10
        y_size = y_size / 10
    # number of x pixels
    grid_columns = int(round(xy_count[0] * gt[1] / x_size))
    # number of y pixels
    grid_rows = int(round(xy_count[1] * abs(gt[5]) / y_size))
    # check crs
    same_crs = compare_crs(reference_crs, vector_crs)
    if vector_path is not None:
        if not same_crs:
            input_vector = cfg.temp.temporary_file_path(
                name_suffix=files_directories.file_extension(vector_path)
            )
            reproject_vector(
                vector_path, input_vector, input_epsg=vector_crs,
                output_epsg=reference_crs
            )
        else:
            input_vector = vector_path
        # open input vector
        vector = ogr.Open(input_vector)
        # get layer
        try:
            v_layer = vector.GetLayer()
        except Exception as err:
            if cfg.logger is not None:
                cfg.logger.log.error(err)
            return False
    elif vector_layer is not None:
        if not same_crs:
            if cfg.logger is not None:
                cfg.logger.log.error('different crs')
            return False
        else:
            v_layer = vector_layer
    else:
        v_layer = None
    # attribute filter
    if attribute_filter is not None:
        v_layer.SetAttributeFilter(attribute_filter)
        d = ogr.GetDriverByName('MEMORY')
        d_s = d.CreateDataSource('memData')
        layer_copy = d_s.CopyLayer(v_layer, d_s.GetName(), ['OVERWRITE=YES'])
        min_x, max_x, min_y, max_y = layer_copy.GetExtent()
        v_layer = layer_copy
    else:
        min_x, max_x, min_y, max_y = v_layer.GetExtent()
    if (max_x - min_x) < x_size:
        max_x = min_x + x_size
    if abs(max_y - min_y) < abs(y_size):
        max_y = min_y + abs(y_size)
    # calculate minimum extent
    if not extent:
        orig_x = gt[0] + x_size * int(round((min_x - gt[0]) / x_size))
        orig_y = gt[3] + y_size * int(round((max_y - gt[3]) / y_size))
        grid_columns = abs(int(round((max_x - min_x) / x_size)))
        grid_rows = abs(int(round((max_y - min_y) / y_size)))
    driver = gdal.GetDriverByName(output_format)
    temporary_grid = cfg.temp.temporary_raster_path(extension=cfg.tif_suffix)
    # create raster _grid
    _grid = driver.Create(
        temporary_grid, grid_columns, grid_rows, 1, gdal.GDT_Float32,
        options=['COMPRESS=LZW']
    )
    if _grid is None:
        _grid = driver.Create(
            temporary_grid, grid_columns, grid_rows, 1, gdal.GDT_Int16,
            options=['COMPRESS=LZW']
        )
    if _grid is None:
        if cfg.logger is not None:
            cfg.logger.log.error('error output raster')
        return False
    try:
        _grid.GetRasterBand(1)
    except Exception as err:
        if cfg.logger is not None:
            cfg.logger.log.error(err)
        return False
    # set raster projection from reference
    _grid.SetGeoTransform([orig_x, x_size, 0, orig_y, 0, -y_size])
    _grid.SetProjection(reference_crs)
    _grid = None
    # create output raster
    create_raster_from_reference(
        path=temporary_grid, band_number=1, output_raster_list=[output_path],
        nodata_value=nodata_value, driver='GTiff',
        gdal_format=cfg.raster_data_type, compress=compress,
        compress_format=compress_format, constant_value=background_value
    )
    # convert reference layer to raster
    output_raster = gdal.Open(output_path, gdal.GA_Update)
    if all_touched is False:
        if burn_values is None:
            o_c = gdal.RasterizeLayer(
                output_raster, [1], v_layer, options=[
                    'ATTRIBUTE=%s' % str(field_name)]
            )
        else:
            o_c = gdal.RasterizeLayer(
                output_raster, [1], v_layer, burn_values=[burn_values]
            )
    else:
        if burn_values is None:
            o_c = gdal.RasterizeLayer(
                output_raster, [1], v_layer,
                options=['ATTRIBUTE=%s' % str(field_name),
                         'all_touched=TRUE']
            )
        else:
            o_c = gdal.RasterizeLayer(
                output_raster, [1], v_layer, burn_values=[burn_values],
                options=['all_touched=TRUE']
            )
    if cfg.logger is not None:
        cfg.logger.log.debug('gdal rasterize: %s' % str(o_c))
    return output_path


# extract reference vector to raster values
def extract_vector_to_raster(
        vector_path=None, reference_raster_path=None, all_touched=False,
        output_format='MEM', burn_values=1, calc_data_type=None,
        attribute_filter=None, nodata_value=0, background_value=0,
        vector_layer=None, available_ram=None
):
    if cfg.logger is not None:
        cfg.logger.log.debug('start')
    # GDAL config
    try:
        if available_ram is None:
            available_ram = cfg.available_ram
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(int(available_ram)
                                                  * 1000000))
        gdal.SetConfigOption('VSI_CACHE', 'FALSE')
    except Exception as err:
        str(err)
    if vector_path is not None:
        vector_crs = get_crs(vector_path)
    elif vector_layer is not None:
        vector_crs = get_layer_crs(vector_layer)
    else:
        if cfg.logger is not None:
            cfg.logger.log.error('input vector')
        return False
    (gt, reference_crs, unit, xy_count, nd, number_of_bands, block_size,
     scale_offset, data_type) = raster_info(reference_raster_path)
    # get data type from input
    if data_type == cfg.float64_dt:
        gdal_format = gdal.GDT_Float64
    elif data_type == cfg.float32_dt:
        gdal_format = gdal.GDT_Float32
    elif data_type == cfg.int32_dt:
        gdal_format = gdal.GDT_Int32
    elif data_type == cfg.uint32_dt:
        gdal_format = gdal.GDT_UInt32
    elif data_type == cfg.int16_dt:
        gdal_format = gdal.GDT_Int16
    elif data_type == cfg.uint16_dt:
        gdal_format = gdal.GDT_UInt16
    elif data_type == cfg.byte_dt:
        gdal_format = gdal.GDT_Byte
    else:
        gdal_format = gdal.GDT_Float32
    # optional calc data type, if None use input data type
    if calc_data_type is None:
        if data_type == cfg.float64_dt:
            calc_data_type = np.float64
        elif data_type == cfg.float32_dt:
            calc_data_type = np.float32
        elif data_type == cfg.int32_dt:
            calc_data_type = np.int32
        elif data_type == cfg.uint32_dt:
            calc_data_type = np.uint32
        elif data_type == cfg.int16_dt:
            calc_data_type = np.int16
        elif data_type == cfg.uint16_dt:
            calc_data_type = np.uint16
        elif data_type == cfg.byte_dt:
            calc_data_type = np.uint8
        else:
            calc_data_type = np.float32
    # check crs
    same_crs = compare_crs(reference_crs, vector_crs)
    if vector_path is not None:
        if not same_crs:
            input_vector = cfg.temp.temporary_file_path(
                name_suffix=files_directories.file_extension(vector_path)
            )
            reproject_vector(
                vector_path, input_vector, input_epsg=vector_crs,
                output_epsg=reference_crs
            )
        else:
            input_vector = vector_path
        # open input _vector
        _vector = ogr.Open(input_vector)
        # get layer
        try:
            v_layer = _vector.GetLayer()
        except Exception as err:
            if cfg.logger is not None:
                cfg.logger.log.error(err)
            return False
    elif vector_layer is not None:
        if not same_crs:
            if cfg.logger is not None:
                cfg.logger.log.error('different crs')
            return False
        else:
            v_layer = vector_layer
    else:
        v_layer = None
    if attribute_filter is None:
        if cfg.logger is not None:
            cfg.logger.log.error('attribute filter')
        return False
    # attribute filter
    v_layer.SetAttributeFilter(attribute_filter)
    d = ogr.GetDriverByName('MEMORY')
    d_s = d.CreateDataSource('memData')
    layer_copy = d_s.CopyLayer(v_layer, d_s.GetName(), ['OVERWRITE=YES'])
    v_min_x, v_max_x, v_min_y, v_max_y = layer_copy.GetExtent()
    v_layer = layer_copy
    _vector = None
    # calculate minimum extent
    x_size = gt[1]
    y_size = abs(gt[5])
    if (v_max_x - v_min_x) < x_size:
        v_max_x = v_min_x + x_size
    if abs(v_max_y - v_min_y) < abs(y_size):
        v_max_y = v_min_y + abs(y_size)
    orig_x = gt[0] + x_size * int(round((v_min_x - gt[0]) / x_size))
    orig_y = gt[3] + y_size * int(round((v_max_y - gt[3]) / y_size))
    # increase size by 1
    # TODO check
    grid_columns = abs(int(round((v_max_x - v_min_x) / x_size))) + 1
    grid_rows = abs(int(round((v_max_y - v_min_y) / y_size))) + 1
    pixel_start_column = abs(int(round((gt[0] - orig_x) / x_size)))
    pixel_start_row = abs(int(round((orig_y - gt[3]) / y_size)))
    driver = gdal.GetDriverByName(output_format)
    # create raster _grid
    _grid = driver.Create('', grid_columns, grid_rows, 1, gdal_format)
    if _grid is None:
        if cfg.logger is not None:
            cfg.logger.log.error('error output raster')
        return False
    try:
        _band = _grid.GetRasterBand(1)
    except Exception as err:
        if cfg.logger is not None:
            cfg.logger.log.error(err)
        return False
    # set raster projection from reference
    _grid.SetGeoTransform([orig_x, x_size, 0, orig_y, 0, -y_size])
    _grid.SetProjection(reference_crs)
    _band.SetNoDataValue(nodata_value)
    _band.Fill(background_value)
    _band = None
    # convert reference layer to raster
    if all_touched is False:
        gdal.RasterizeLayer(
            _grid, [1], v_layer, burn_values=[burn_values]
        )
    else:
        gdal.RasterizeLayer(
            _grid, [1], v_layer, burn_values=[burn_values],
            options=['all_touched=TRUE']
        )
    _r_d = gdal.Open(reference_raster_path, gdal.GA_ReadOnly)
    r_b = _r_d.GetRasterBand(1)
    _a = read_array_block(
        gdal_band=r_b, pixel_start_column=pixel_start_column,
        pixel_start_row=pixel_start_row,
        block_columns=grid_columns, block_row=grid_rows,
        calc_data_type=calc_data_type
    )
    _band = _grid.GetRasterBand(1)
    _b = read_array_block(
        gdal_band=_band, pixel_start_column=0,
        pixel_start_row=0, block_columns=grid_columns, block_row=grid_rows
    )
    _grid = _r_d = None
    if cfg.logger is not None:
        cfg.logger.log.debug('extract vector')
    if _a is None:
        return None, None, None
    return _a.ravel(), _b.ravel(), _a.ravel() != nd


# get pixel value
def get_pixel_value(
        point_coordinates, reference_raster_path=None, point_crs=None,
        available_ram=None
):
    if cfg.logger is not None:
        cfg.logger.log.debug('start')
    # GDAL config
    try:
        if available_ram is None:
            available_ram = cfg.available_ram
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(int(available_ram)
                                                  * 1000000))
        gdal.SetConfigOption('VSI_CACHE', 'FALSE')
    except Exception as err:
        str(err)
    (gt, reference_crs, unit, xy_count, nd, number_of_bands, block_size,
     scale_offset, data_type) = raster_info(reference_raster_path)
    orig_x = gt[0]
    orig_y = gt[3]
    p_x_size = gt[1]
    p_y_size = abs(gt[5])
    # TODO implement reprojection
    if point_crs is not None:
        # check crs
        same_crs = compare_crs(reference_crs, point_crs)
        if not same_crs:
            pass
        else:
            pass
    # calculate pixel position
    grid_column = abs(int(round((point_coordinates[0] - orig_x) / p_x_size)))
    grid_row = abs(int(round((orig_y - point_coordinates[1]) / p_y_size)))
    if grid_column < 0 or grid_row < 0:
        return None
    _r_d = gdal.Open(reference_raster_path, gdal.GA_ReadOnly)
    _r_b = _r_d.GetRasterBand(1)
    pixel_value = read_array_block(
        _r_b, pixel_start_column=grid_column, pixel_start_row=grid_row,
        block_columns=1, block_row=1, calc_data_type=None
    )
    pixel_value = float(pixel_value.item())
    _r_b = _r_d = None
    if cfg.logger is not None:
        cfg.logger.log.debug('pixel value: %s' % str(pixel_value))
    return pixel_value


# merge all layers to new layer
# noinspection PyArgumentList
def merge_all_layers(
        input_layers_list, target_layer, min_progress=0, max_progress=100,
        dissolve_output=None
):
    cfg.logger.log.debug('start')
    t_l = create_virtual_layer(input_layers_list)
    # open virtual layer
    input_source = ogr.Open(t_l)
    # get input layer definition
    input_layer = input_source.GetLayer()
    i_sr = input_layer.GetSpatialRef()
    input_layer_def = input_layer.GetLayerDefn()
    input_field_count = input_layer_def.GetFieldCount()
    i_d = ogr.GetDriverByName('GPKG')
    # create output
    output_source = i_d.CreateDataSource(target_layer)
    output_name = files_directories.file_name(target_layer)
    _output_layer = output_source.CreateLayer(
        str(output_name), i_sr, ogr.wkbPolygon
    )
    # fields
    field_names = []
    for f in range(input_field_count):
        f_def = input_layer_def.GetFieldDefn(f)
        _output_layer.CreateField(f_def)
        field_names.append(f_def.GetNameRef())
    if not dissolve_output:
        # add area field
        area_field_def = ogr.FieldDefn(cfg.area_field_name, ogr.OFTReal)
        area_field_def.SetWidth(30)
        area_field_def.SetPrecision(2)
        _output_layer.CreateField(area_field_def)
    output_layer_def = _output_layer.GetLayerDefn()
    # start copy
    layer_count = input_source.GetLayerCount()
    _output_layer.StartTransaction()
    q = 0
    for i in input_source:
        q += 1
        cfg.progress.update(
            step=q, steps=layer_count, minimum=min_progress,
            maximum=max_progress, message='merging vectors',
            percentage=q / layer_count
        )
        i_name = i.GetName()
        i_layer = input_source.GetLayer(i_name)
        i_feature = i_layer.GetNextFeature()
        while i_feature:
            if cfg.action is True:
                geometry = i_feature.GetGeometryRef()
                o_feature = ogr.Feature(output_layer_def)
                o_feature.SetGeometry(geometry)
                if not dissolve_output:
                    # set area
                    area = geometry.GetArea()
                    o_feature.SetField(cfg.area_field_name, area)
                for c in range(input_field_count):
                    try:
                        field_name = field_names[c]
                        field = i_feature.GetField(c)
                        o_feature.SetField(field_name, field)
                    except Exception as err:
                        cfg.logger.log.error(err)
                _output_layer.CreateFeature(o_feature)
                i_feature = i_layer.GetNextFeature()
            else:
                cfg.logger.log.error('cancel')
                _output_layer = None
                return None
    _output_layer.CommitTransaction()
    cfg.logger.log.debug('target_layer: %s' % target_layer)
    return target_layer


# create virtual layer
def create_virtual_layer(input_layer_list, target_layer=None):
    source = '''
    <OGRVRTDataSource>
    '''
    for layer in input_layer_list:
        i = ogr.Open(layer)
        i_l = i.GetLayer()
        i_n = i_l.GetName()
        xml = '''
            <OGRVRTLayer name="%s">
                <SrcDataSource>%s</SrcDataSource>
                <SrcLayer>%s</SrcLayer>
            </OGRVRTLayer>
        '''
        source += xml % (i_n, layer, i_n)
    source += '''
    </OGRVRTDataSource>
    '''
    if target_layer is None:
        target_layer = cfg.temp.temporary_raster_path(extension=cfg.vrt_suffix)
    with open(target_layer, 'w') as file:
        file.write(source)
    return target_layer


# merge dissolve layer to new layer
# noinspection PyArgumentList
def merge_dissolve_layer(
        input_layer, target_layer, column, y_list_coordinates, min_progress=0,
        max_progress=100
):
    cfg.logger.log.debug('start')
    # open virtual layer
    input_source = ogr.Open(input_layer)
    i_layer = input_source.GetLayer()
    i_layer_name = i_layer.GetName()
    i_layer_sr = i_layer.GetSpatialRef()
    i_layer_def = i_layer.GetLayerDefn()
    field_count = i_layer_def.GetFieldCount()
    i_d = ogr.GetDriverByName('GPKG')
    output_source = i_d.CreateDataSource(target_layer)
    output_name = files_directories.file_name(target_layer)
    o_layer = output_source.CreateLayer(
        str(output_name), i_layer_sr, ogr.wkbPolygon
    )
    # fields
    for f_c in range(field_count):
        field_def = i_layer_def.GetFieldDefn(f_c)
        o_layer.CreateField(field_def)
    # add area field
    area_field_def = ogr.FieldDefn(cfg.area_field_name, ogr.OFTReal)
    area_field_def.SetWidth(30)
    area_field_def.SetPrecision(2)
    o_layer.CreateField(area_field_def)
    o_layer_def = o_layer.GetLayerDefn()
    # get unique values
    sql = 'SELECT DISTINCT "%s" FROM "%s"' % (column, i_layer_name)
    unique_values = input_source.ExecuteSQL(sql, dialect='SQLITE')
    values = []
    id_list = []
    for i, f in enumerate(unique_values):
        values.append(f.GetField(0))
    # release sql results
    input_source.ReleaseResultSet(unique_values)
    sql_list = str(y_list_coordinates)[1:-1].replace("'", '')
    # for each value
    n = 0
    min_p = min_progress
    max_p = int((max_progress - min_progress) / (len(values) + 1))
    for v in values:
        n += 1
        cfg.progress.update(
            step=n, steps=len(values), minimum=min_p + max_p * (n - 1),
            maximum=min_p + max_p * n, message='dissolving polygons',
            percentage=n / len(values)
        )
        cfg.logger.log.debug('progress: %s' % str(n / len(values)))
        # get geometries to be dissolved
        sql = 'SELECT DISTINCT(ST_COLLECT(CastToMultiPolygon(geom))), ' \
              'GROUP_CONCAT(DISTINCT id) FROM (SELECT fid as ' \
              'id, geom FROM "%s" WHERE %s = %s) INNER JOIN (SELECT ' \
              'DISTINCT id FROM "rtree_%s_geom" WHERE miny IN (' \
              '%s) OR maxy IN (%s) ) USING (id)' % (
                  i_layer_name, column, str(v), i_layer_name, sql_list,
                  sql_list)
        unique_features = input_source.ExecuteSQL(sql, dialect='SQLITE')
        if unique_features is not None:
            uv_features = unique_features.GetNextFeature()
            field_value = uv_features.GetField(0)
            geometry_ref = uv_features.GetGeometryRef()
            if geometry_ref is not None:
                if not geometry_ref.IsValid():
                    # try to fix invalid geometry with buffer
                    geometry_ref = geometry_ref.Buffer(0.0)
                geometry_count = geometry_ref.GetGeometryCount()
                o_layer.StartTransaction()
                if geometry_count > 1:
                    for g in range(geometry_count):
                        g_geometry_ref = geometry_ref.GetGeometryRef(int(g))
                        try:
                            if g_geometry_ref is not None:
                                # try to fix invalid geometry with buffer
                                if not g_geometry_ref.IsValid():
                                    g_geometry_ref = g_geometry_ref.Buffer(0.0)
                                if g_geometry_ref.IsValid():
                                    o_feature = ogr.Feature(o_layer_def)
                                    o_feature.SetGeometry(g_geometry_ref)
                                    o_feature.SetField(column, v)
                                    # set area
                                    area = g_geometry_ref.GetArea()
                                    o_feature.SetField(
                                        cfg.area_field_name, area
                                    )
                                    o_layer.CreateFeature(o_feature)
                                else:
                                    # rollback
                                    o_layer.RollbackTransaction()
                                    o_layer.CommitTransaction()
                                    o_layer.StartTransaction()
                                    o_feature = ogr.Feature(o_layer_def)
                                    o_feature.SetGeometry(geometry_ref)
                                    o_feature.SetField(column, v)
                                    # set area
                                    area = geometry_ref.GetArea()
                                    o_feature.SetField(
                                        cfg.area_field_name, area
                                    )
                                    o_layer.CreateFeature(o_feature)
                                    break
                        except Exception as err:
                            str(err)
                            # rollback
                            o_layer.RollbackTransaction()
                            o_layer.CommitTransaction()
                            o_layer.StartTransaction()
                            o_feature = ogr.Feature(o_layer_def)
                            o_feature.SetGeometry(geometry_ref)
                            o_feature.SetField(column, v)
                            # set area
                            area = geometry_ref.GetArea()
                            o_feature.SetField(cfg.area_field_name, area)
                            o_layer.CreateFeature(o_feature)
                            break
                    cfg.logger.log.debug('added union geometries')
                # single geometry
                else:
                    o_feature = ogr.Feature(o_layer_def)
                    o_feature.SetGeometry(geometry_ref)
                    o_feature.SetField(column, v)
                    # set area
                    area = geometry_ref.GetArea()
                    o_feature.SetField(cfg.area_field_name, area)
                    o_layer.CreateFeature(o_feature)
                o_layer.CommitTransaction()
            if field_value is not None:
                id_list1 = field_value.split(',')
                id_list.extend(id_list1)
        # release sql results
        input_source.ReleaseResultSet(unique_features)
    cfg.progress.update(step=min_p + max_p * n, message='copying polygons')
    feature_count = i_layer.GetFeatureCount()
    # copy not dissolved features
    i_feature = i_layer.GetNextFeature()
    c = 0
    while i_feature:
        if cfg.action is True:
            c += 1
            cfg.progress.update(
                step=c, steps=feature_count, minimum=min_p + max_p * n,
                maximum=max_progress, percentage=int(c / feature_count * 100)
            )
            i_fid = str(i_feature.GetFID())
            v_field = i_feature.GetField(column)
            if str(i_fid) not in id_list:
                i_geom = i_feature.GetGeometryRef()
                o_feature = ogr.Feature(o_layer_def)
                o_feature.SetGeometry(i_geom)
                o_feature.SetField(column, v_field)
                # set area
                area = i_geom.GetArea()
                o_feature.SetField(cfg.area_field_name, area)
                o_layer.CreateFeature(o_feature)
            i_feature = i_layer.GetNextFeature()
        else:
            cfg.logger.log.error('cancel')
            return None
    cfg.logger.log.debug('end')
    return target_layer


# merge geometries by list of IDs and save them in a new vector
def merge_polygons(
        input_layer, value_list, target_layer
):
    cfg.logger.log.debug('start')
    # open virtual layer
    _input_source = ogr.Open(input_layer)
    _i_layer = _input_source.GetLayer()
    i_layer_name = _i_layer.GetName()
    i_layer_sr = _i_layer.GetSpatialRef()
    i_layer_def = _i_layer.GetLayerDefn()
    field_count = i_layer_def.GetFieldCount()
    i_d = ogr.GetDriverByName('GPKG')
    _output_source = i_d.CreateDataSource(target_layer)
    _output_name = files_directories.file_name(target_layer)
    _o_layer = _output_source.CreateLayer(
        str(_output_name), i_layer_sr, ogr.wkbMultiPolygon
    )
    # fields
    for f_c in range(field_count):
        field_def = i_layer_def.GetFieldDefn(f_c)
        _o_layer.CreateField(field_def)
    o_layer_def = _o_layer.GetLayerDefn()
    # get unique values
    sql = ('SELECT %s, ST_Union(geom) AS merged_geometry FROM %s '
           'WHERE %s IN (%s)') % (
              cfg.uid_field_name, i_layer_name, cfg.uid_field_name,
              str(value_list).replace('[', '').replace(']', '')
          )
    output_values = _input_source.ExecuteSQL(sql, dialect='SQLITE')
    if output_values is not None:
        uv_features = output_values.GetNextFeature()
        geometry_ref = uv_features.GetGeometryRef()
        if geometry_ref is not None:
            if not geometry_ref.IsValid():
                # try to fix invalid geometry with buffer
                geometry_ref = geometry_ref.Buffer(0.0)
            geometry_count = geometry_ref.GetGeometryCount()
            cfg.logger.log.debug('geometry_count: %s' % str(geometry_count))
            _o_layer.StartTransaction()
            _o_feature = ogr.Feature(o_layer_def)
            _o_feature.SetGeometry(geometry_ref)
            _o_layer.CreateFeature(_o_feature)
            _o_layer.CommitTransaction()
            _o_feature = None
    # release sql results
    _input_source.ReleaseResultSet(output_values)
    _i_layer = None
    _o_layer = None
    _input_source = None
    _output_source = None
    cfg.logger.log.debug('end; target_layer: %s' % target_layer)
    return target_layer


# save selected geometries by list of IDs and save them in a new vector
def save_polygons(
        input_layer, value_list, target_layer, vector_format=None
):
    cfg.logger.log.debug('start')
    if vector_format is None:
        vector_format = 'GPKG'
    # open virtual layer
    _input_source = ogr.Open(input_layer)
    _i_layer = _input_source.GetLayer()
    i_layer_name = _i_layer.GetName()
    i_layer_sr = _i_layer.GetSpatialRef()
    i_layer_def = _i_layer.GetLayerDefn()
    field_count = i_layer_def.GetFieldCount()
    i_d = ogr.GetDriverByName(vector_format)
    _output_source = i_d.CreateDataSource(target_layer)
    _output_name = files_directories.file_name(target_layer)
    _o_layer = _output_source.CreateLayer(
        str(_output_name), i_layer_sr, ogr.wkbMultiPolygon
    )
    # fields
    for f_c in range(field_count):
        field_def = i_layer_def.GetFieldDefn(f_c)
        _o_layer.CreateField(field_def)
    o_layer_def = _o_layer.GetLayerDefn()
    # get unique values
    sql = 'SELECT * FROM %s WHERE %s IN (%s)' % (
        i_layer_name, cfg.uid_field_name,
        str(value_list).replace('[', '').replace(']', '')
    )
    output_values = _input_source.ExecuteSQL(sql, dialect='SQLITE')
    if output_values is not None:
        uv_feature = output_values.GetNextFeature()
        while uv_feature:
            if cfg.action is True:
                geometry_ref = uv_feature.GetGeometryRef()
                if geometry_ref is not None:
                    if not geometry_ref.IsValid():
                        # try to fix invalid geometry with buffer
                        geometry_ref = geometry_ref.Buffer(0.0)
                    _o_layer.StartTransaction()
                    _o_feature = ogr.Feature(o_layer_def)
                    _o_feature.SetGeometry(geometry_ref)
                    for i in range(field_count):
                        _o_feature.SetField(i, uv_feature.GetField(i))
                    _o_layer.CreateFeature(_o_feature)
                    _o_layer.CommitTransaction()
                    _o_feature = None
                uv_feature = output_values.GetNextFeature()
    # release sql results
    _input_source.ReleaseResultSet(output_values)
    _i_layer = None
    _o_layer = None
    _input_source = None
    _output_source = None
    cfg.logger.log.debug('end; target_layer: %s' % target_layer)
    return target_layer


# create a geometry vector for Spectral Signatures
def create_geometry_vector(
        output_path, crs_wkt, macroclass_field_name,
        class_field_name, vector_format='GPKG',
):
    # in case crs_wkt is already as crs format
    try:
        crs_wkt = str(crs_wkt.toWkt())
    except Exception as err:
        str(err)
    driver = ogr.GetDriverByName(vector_format)
    data_source = driver.CreateDataSource(output_path)
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(crs_wkt)
    name = files_directories.file_name(output_path, suffix=False)
    files_directories.create_parent_directory(output_path)
    layer = data_source.CreateLayer(name, spatial_ref, ogr.wkbMultiPolygon)
    # spectral signature id
    field_uid = ogr.FieldDefn(cfg.uid_field_name, ogr.OFTString)
    layer.CreateField(field_uid)
    macroclass_field = ogr.FieldDefn(macroclass_field_name, ogr.OFTInteger)
    layer.CreateField(macroclass_field)
    class_field = ogr.FieldDefn(class_field_name, ogr.OFTInteger)
    layer.CreateField(class_field)
    return output_path


# remove polygon from vector and return vector
def remove_polygon_from_vector(vector_path, attribute_field, attribute_value):
    _vector = ogr.Open(vector_path, 1)
    # get layer
    try:
        _v_layer = _vector.GetLayer()
    except Exception as err:
        cfg.logger.log.error(err)
        return False
    v_layer_name = _v_layer.GetName()
    sql = "DELETE FROM %s WHERE %s = '%s'" % (
        v_layer_name, attribute_field, attribute_value)
    _vector.ExecuteSQL(sql)
    _vector = None
    return True


# get polygon from vector and return memory layer
def get_polygon_from_vector(vector_path, attribute_filter=None):
    # open input vector
    vector = ogr.Open(vector_path)
    # get layer
    try:
        _v_layer = vector.GetLayer()
    except Exception as err:
        cfg.logger.log.error(err)
        return False
    # attribute filter
    _v_layer.SetAttributeFilter(attribute_filter)
    d = ogr.GetDriverByName('GPKG')
    temp = cfg.temp.temporary_file_path(name_suffix=cfg.gpkg_suffix)
    _d_s = d.CreateDataSource(temp)
    _d_s.CopyLayer(_v_layer, _v_layer.GetName(), ['OVERWRITE=YES'])
    _v_layer = None
    _d_s = None
    return temp


# gdal copy raster
def gdal_copy_raster(input_raster, output, output_format='GTiff'):
    out_dir = files_directories.parent_directory(output)
    files_directories.create_directory(out_dir)
    raster_driver = gdal.GetDriverByName(output_format)
    _r_d = gdal.Open(input_raster, gdal.GA_ReadOnly)
    # geo transformation
    r_gt = _r_d.GetGeoTransform()
    r_p = _r_d.GetProjection()
    _out_raster = raster_driver.CreateCopy(output, _r_d)
    # set raster projection from reference
    _out_raster.SetGeoTransform(r_gt)
    _out_raster.SetProjection(r_p)
    _r_d = None
    _out_raster = None
    return output


# gdal array to polygon using reference raster
def array_to_polygon(input_array, reference_raster, output):
    out_dir = files_directories.parent_directory(output)
    files_directories.create_directory(out_dir)
    driver = ogr.GetDriverByName('GPKG')
    _r_d = gdal.Open(reference_raster, gdal.GA_ReadOnly)
    # create empty vector
    _data_source = driver.CreateDataSource(output)
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(_r_d.GetProjectionRef())
    _vector_layer = _data_source.CreateLayer(
        'region', spatial_reference, ogr.wkbPolygon
    )
    field_name = 'DN'
    field_definition = ogr.FieldDefn(field_name, ogr.OFTInteger)
    _vector_layer.CreateField(field_definition)
    field = _vector_layer.GetLayerDefn().GetFieldIndex(field_name)
    # create raster from array
    temp_raster = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
    _input_band = _r_d.GetRasterBand(1)
    data_type = _input_band.DataType
    raster_driver = gdal.GetDriverByName('GTiff')
    # x pixel count
    x_count = _r_d.RasterXSize
    # y pixel count
    y_count = _r_d.RasterYSize
    # geo transformation
    gt = _r_d.GetGeoTransform()
    input_array = input_array.reshape(
        input_array.shape[0], input_array.shape[1]
    )
    _raster = raster_driver.Create(temp_raster, x_count, y_count, 1, data_type)
    _raster.SetGeoTransform([gt[0], gt[1], 0, gt[3], 0, gt[5]])
    raster_projection = _r_d.GetProjection()
    _raster.SetProjection(raster_projection)
    _raster_band = _raster.GetRasterBand(1)
    _raster_band.SetNoDataValue(0)
    # write array
    _raster_band.WriteArray(input_array)
    # raster to polygon
    gdal.Polygonize(
        _raster_band, _raster_band.GetMaskBand(), _vector_layer, field
    )
    _raster_band = None
    _raster = None
    _input_band = None
    _r_d = None
    _data_source = None
    _vector_layer = None
    return output


# gdal warp
def gdal_warping(
        input_raster, output, output_format='GTiff', s_srs=None,
        t_srs=None, resample_method=None, raster_data_type=None,
        compression=None, compress_format='DEFLATE', additional_params='',
        n_processes: int = None, available_ram: int = None,
        src_nodata=None, dst_nodata=None, min_progress=0, max_progress=100
):
    cfg.logger.log.debug('start')
    out_dir = files_directories.parent_directory(output)
    files_directories.create_directory(out_dir)
    if resample_method is None:
        resample_method = 'near'
    elif resample_method == 'sum':
        gdal_v = get_gdal_version()
        if float('%s.%s' % (gdal_v[0], gdal_v[1])) < 3.1:
            cfg.logger.log.error('Error GDAL version')
            return False
    if n_processes is None:
        n_processes = cfg.n_processes
    option_string = ' -r %s -co BIGTIFF=YES -multi -wo NUM_THREADS=%s' % (
        resample_method, str(n_processes))
    if compression is None:
        if cfg.raster_compression:
            option_string += ' -co COMPRESS=%s' % compress_format
    elif compression:
        option_string += ' -co COMPRESS=%s' % compress_format
    if s_srs is not None:
        option_string += ' -s_srs %s' % s_srs
    if t_srs is not None:
        option_string += ' -t_srs %s' % t_srs
    if raster_data_type is not None:
        option_string += ' -ot %s' % raster_data_type
    if src_nodata is not None:
        option_string += ' -srcnodata %s' % str(src_nodata)
    if dst_nodata is not None:
        option_string += ' -dstnodata %s' % str(dst_nodata)
    option_string += ' -of %s' % output_format
    if additional_params is not None:
        option_string = ' %s %s' % (additional_params, option_string)
    if available_ram is None:
        available_ram = cfg.available_ram
    # GDAL config
    try:
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(int(available_ram)
                                                  * 1000000))
        gdal.SetConfigOption('VSI_CACHE', 'FALSE')
        gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
    except Exception as err:
        str(err)
    try:
        progress_gdal = (lambda percentage, m, c: cfg.progress.update(
            percentage=100 if int(percentage * 100) > 100
            else int(percentage * 100), steps=100, minimum=min_progress,
            maximum=max_progress, step=100 if int(percentage * 100) > 100
            else int(percentage * 100)
        ))
        to = gdal.WarpOptions(
            gdal.ParseCommandLine(option_string), callback=progress_gdal
        )
        gdal.Warp(output, input_raster, options=to)
    except Exception as err:
        cfg.logger.log.error(str(err))
    cfg.logger.log.debug('end; output: %s' % output)


# create jpg image with gdal
def jpg_gdal(
        input_raster, output, quality=90, available_ram: int = None,
        stretch=True
):
    cfg.logger.log.debug('start')
    if available_ram is None:
        available_ram = cfg.available_ram
    # GDAL config
    try:
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(int(available_ram)
                                                  * 1000000))
        gdal.SetConfigOption('VSI_CACHE', 'FALSE')
        gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
    except Exception as err:
        str(err)
    try:
        _input_dataset = gdal.Open(input_raster)
        if _input_dataset is None:
            cfg.logger.log.error('unable to open the input file')
        else:
            _band_1 = _input_dataset.GetRasterBand(1)
            _band_2 = _input_dataset.GetRasterBand(2)
            _band_3 = _input_dataset.GetRasterBand(3)
            minimum_1 = _band_1.GetMinimum()
            maximum_1 = _band_1.GetMaximum()
            if minimum_1 is None or maximum_1 is None:
                minimum_1, maximum_1 = _band_1.ComputeRasterMinMax(True)
            minimum_2 = _band_1.GetMinimum()
            maximum_2 = _band_1.GetMaximum()
            if minimum_2 is None or maximum_2 is None:
                minimum_2, maximum_2 = _band_2.ComputeRasterMinMax(True)
            minimum_3 = _band_1.GetMinimum()
            maximum_3 = _band_1.GetMaximum()
            if minimum_3 is None or maximum_3 is None:
                minimum_3, maximum_3 = _band_3.ComputeRasterMinMax(True)
            if stretch:
                stretch_min = 1.1
                stretch_max = 0.7
            else:
                stretch_min = 1
                stretch_max = 0
            jpg_options = gdal.TranslateOptions(
                format='JPEG',
                creationOptions=['QUALITY=%s' % quality],
                outputType=gdal.GDT_Byte,
                scaleParams=[
                    [round(minimum_1 * stretch_min),
                     round(maximum_1 * stretch_max), 0, 255],
                    [round(minimum_2 * stretch_min),
                     round(maximum_2 * stretch_max), 0, 255],
                    [round(minimum_3 * stretch_min),
                     round(maximum_3 * stretch_max), 0, 255]
                ],
            )
            gdal.Translate(output, _input_dataset, options=jpg_options)
            _band_1 = None
            _band_2 = None
            _band_3 = None
            _input_dataset = None
    except Exception as err:
        cfg.logger.log.error(str(err))
    cfg.logger.log.debug('end; output: %s' % output)


# create polygon from coordinates
def coordinates_to_polygon_gdal(coordinates, input_coordinates, output=None):
    cfg.logger.log.debug('start')
    if output is None:
        output = cfg.temp.temporary_file_path(name_suffix=cfg.gpkg_suffix)
    # create empty vector
    driver = ogr.GetDriverByName('GPKG')
    _data_source = driver.CreateDataSource(output)
    spatial_reference = get_projection_from_epsg_wkt(input_coordinates)
    _output_layer = _data_source.CreateLayer(
        'region', spatial_reference, ogr.wkbPolygon
    )
    field_name = 'DN'
    field_definition = ogr.FieldDefn(field_name, ogr.OFTInteger)
    _output_layer.CreateField(field_definition)
    feature_defn = _output_layer.GetLayerDefn()
    feature = ogr.Feature(feature_defn)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    geom = ogr.Geometry(ogr.wkbPolygon)
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    for coord in coordinates:
        ring.AddPoint(coord[0], coord[1])
    geom.AddGeometry(ring)
    feature.SetGeometry(geom)
    _output_layer.StartTransaction()
    _output_layer.CreateFeature(feature)
    _output_layer.CommitTransaction()
    _output_layer = None
    _poly = None
    return output


# get vector fields and type
def get_vector_fields(input_layer):
    # open layer
    input_source = ogr.Open(input_layer)
    # get input layer definition
    input_layer = input_source.GetLayer()
    input_layer_def = input_layer.GetLayerDefn()
    input_field_count = input_layer_def.GetFieldCount()
    fields = {}
    for f in range(input_field_count):
        f_def = input_layer_def.GetFieldDefn(f)
        fields[f_def.GetNameRef()] = f_def.GetType()
    return fields


# get raster data
def get_colored_raster(path, value_color_dictionary):
    if not files_directories.is_file(path):
        if cfg.logger is not None:
            cfg.logger.log.warning('raster: %s' % path)
    _r_d = gdal.Open(path, gdal.GA_ReadOnly)
    if _r_d is None:
        if cfg.logger is not None:
            cfg.logger.log.error('raster: %s' % path)
        return False
    _band = _r_d.GetRasterBand(1)
    data = _band.ReadAsArray()
    _band = None
    _r_d = None
    array_3_bands = np.zeros(
        (data.shape[0], data.shape[1], 3), dtype=np.uint8
        )
    for value, color in value_color_dictionary.items():
        mask = data == value
        array_3_bands[mask] = color
    return array_3_bands
