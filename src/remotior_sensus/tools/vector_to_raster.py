# Remotior Sensus , software to process remote sensing and GIS data.
# Copyright (C) 2022-2025 Luca Congedo.
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
Vector to raster.

This tool allows for the conversion from vector polygons to raster.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> vector = rs.vector_to_raster(vector_path='file.gpkg',
    ...     align_raster='reference.tif', output_path='raster.tif')
"""  # noqa: E501

from typing import Union, Optional
import numpy
from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.core.processor_functions import (
    vector_to_raster_iter
)
from remotior_sensus.util import files_directories, shared_tools, raster_vector

try:
    from osgeo import ogr
except Exception as error:
    cfg.logger.log.error(str(error))


def vector_to_raster(
        vector_path: str, align_raster: Union[str, BandSet, int],
        vector_field: Optional[str] = None,
        constant: Optional[int] = None,
        pixel_size: Optional[int] = None,
        output_path: Optional[str] = None,
        method: Optional[str] = None,
        area_precision: Optional[int] = 3, resampling='mode',
        nodata_value: Optional[int] = None,
        minimum_extent: Optional[bool] = True,
        extent_list: Optional[list] = None, output_format='GTiff',
        compress=None, compress_format=None,
        n_processes: Optional[int] = None, available_ram: Optional[int] = None,
        bandset_catalog: Optional[BandSetCatalog] = None,
        progress_message: Optional[bool] = True
) -> OutputManager:
    """Performs the conversion from vector to raster.

    This tool performs the conversion from vector polygons to raster.

    Args:
        vector_path: path of vector used as input.
        align_raster: optional string path of raster used for aligning output 
            pixels and projections; it can also be a BandSet or an integer 
            number of a BandSet in a Catalog.
        output_path: string of output path.
        vector_field: the name of the field used as reference value.
        constant: integer value used as reference for all the polygons.
        pixel_size: size of pixel of output raster.
        minimum_extent: if True, raster has the minimum vector extent; if 
            False, the extent is the same as the align raster.
        extent_list: list of boundary coordinates left top right bottom.
        output_format: output format, default GTiff
        method: method of conversion, default pixel_center, other methods are 
            all_touched for burning all pixels touched or area_based for 
            burning values based on area proportion inside the pixel (warning: 
            using area_based method, a pixel covered by multiple polygons 
            each covering less than 50% of the area may be incorrectly assigned).
        area_precision: for area_based method, the higher the value, the more 
            is the precision in area proportion calculation.
        resampling: type for resampling when method is area_based.
        compress: if True, compress the output raster.
        compress_format: compress format.
        nodata_value: value to be considered as nodata.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        bandset_catalog: BandSetCatalog object.
        progress_message: if True then start progress message, if False does 
            not start the progress message (useful if launched from other tools).

    Returns:
        object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - path = output path

    Examples:
        Perform the conversion to raster of a vector using area_based method
            >>> vector_to_raster(vector_path='file.gpkg', align_raster='reference.tif', method='area_based', output_path='raster.tif')
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=progress_message
    )
    vector_path = files_directories.input_path(vector_path)
    if type(align_raster) is str:
        input_bands = [align_raster]
    else:
        input_bands = align_raster
    cfg.logger.log.debug('input_bands: %s' % str(input_bands))
    # prepare process files
    prepared = shared_tools.prepare_process_files(
        input_bands=input_bands, output_path=output_path,
        n_processes=n_processes, box_coordinate_list=extent_list,
        bandset_catalog=bandset_catalog
    )
    reference_path = prepared['temporary_virtual_raster'][0]
    # prepare output
    temp_path = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
    if n_processes is None:
        n_processes = cfg.n_processes
    if available_ram is None:
        available_ram = cfg.available_ram
    # perform conversion
    if compress is None:
        compress = cfg.raster_compression
    if compress_format is None:
        compress_format = 'DEFLATE21'
    (gt, reference_crs, unit, xy_count, nd, number_of_bands, block_size,
     scale_offset, data_type) = raster_vector.raster_info(align_raster)
    if pixel_size is None:
        x_y_size = (round(gt[1], 3), round(abs(gt[5]), 3))
    else:
        x_y_size = [pixel_size, pixel_size]
    t_pixel_size = x_y_size
    if vector_field is None and constant is None:
        constant = 1
    nodata_value_set = nodata_value
    if nodata_value_set is None:
        nodata_value_set = cfg.nodata_val_Int32
    min_progress = 1
    if method is None or method.lower() == 'pixel_center':
        all_touched = None
        max_progress = 100
    elif method.lower() == 'all_touched':
        all_touched = True
        max_progress = 100
    elif method.lower() == 'area_based':
        all_touched = None
        compress = True
        max_progress = 50
        # calculate pixel size precision
        size_precision = round(x_y_size[0] / area_precision, 2)
        if size_precision == 0:
            size_precision = 0.1
        ratio = size_precision.as_integer_ratio()
        # greatest common divisor
        try:
            area_precision = numpy.gcd(
                ratio[1], x_y_size[0]) * 10 ** (len(str(area_precision)) - 1)
        except Exception as err:
            str(err)
            area_precision = numpy.gcd(
                ratio[1], int(x_y_size[0] * 100)
            ) * 10 ** (len(str(area_precision)) - 1)
        temp_px_size = x_y_size[0] / area_precision
        t_pixel_size = [temp_px_size, temp_px_size]
    else:
        all_touched = None
        max_progress = 100
    if output_path is None:
        output_path = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
    output_path = files_directories.output_path(output_path, cfg.tif_suffix)
    files_directories.create_parent_directory(output_path)
    cfg.logger.log.debug('t_pixel_size: %s' % str(t_pixel_size))
    cfg.progress.update(message='processing', step=1)
    # open input with GDAL
    cfg.logger.log.debug('vector_path: %s' % vector_path)
    vector_crs = raster_vector.get_crs(vector_path)
    reference_crs = raster_vector.get_crs(reference_path)
    # check crs
    same_crs = raster_vector.compare_crs(vector_crs, reference_crs)
    cfg.logger.log.debug('same_crs: %s' % str(same_crs))
    if not same_crs:
        input_vector = cfg.temp.temporary_file_path(
            name_suffix=files_directories.file_extension(vector_path)
        )
        vector_path = raster_vector.reproject_vector(
            vector_path, input_vector, input_epsg=vector_crs,
            output_epsg=reference_crs
        )
    if method is not None and method.lower() == 'area_based':
        # force nodata value (workaround for later gdal copy issue)
        nodata_value_set = cfg.nodata_val_Int32
        temp_vector = cfg.temp.temporary_file_path(name_suffix=cfg.gpkg_suffix)
        # dissolve vector
        cfg.multiprocess.gdal_vector_translate(
            input_file=vector_path, output_file=temp_vector,
            explode_collections=True,
            attribute_field=vector_field, min_progress=1, max_progress=10
        )
        _vector_source = ogr.Open(temp_vector)
        _vector_layer = _vector_source.GetLayer()
        layer_defn = _vector_layer.GetLayerDefn()
        field_definitions = [
            {'name': layer_defn.GetFieldDefn(i).GetName(),
             'type': layer_defn.GetFieldDefn(i).GetType(),
             'width': layer_defn.GetFieldDefn(i).GetWidth(),
             'precision': layer_defn.GetFieldDefn(i).GetPrecision()
             } for i in range(layer_defn.GetFieldCount())
        ]
        # check projection
        proj = _vector_layer.GetSpatialRef()
        crs = proj.ExportToWkt()
        crs = crs.replace(' ', '')
        if len(crs) == 0:
            crs = None
        vector_crs = crs
        feature_list = []
        for idx, feature in enumerate(_vector_layer):
            feature_geom = feature.GetGeometryRef()
            if feature_geom is not None:
                geom = feature_geom.ExportToWkt()
                attrs = [feature.GetField(i) for i in
                         range(layer_defn.GetFieldCount())]
                feature_list.append([field_definitions, geom, attrs])
        _vector_source.Destroy()
        _vector_layer = None
        _vector_source = None
        function_list = []
        argument_list = []
        ram = int(available_ram / n_processes)
        # create virtual raster
        virtual_raster_list = []
        for i, feature in enumerate(feature_list):
            temporary_raster = cfg.temp.temporary_raster_path(
                name_prefix=str(feature[2][0]), extension=cfg.tif_suffix
            )
            argument_list.append(
                {
                    'feature': feature,
                    'vector_crs': vector_crs,
                    'field_name': vector_field,
                    'reference_raster_path': reference_path,
                    'background_value': nodata_value_set,
                    'x_y_size': t_pixel_size,
                    'buffer_size': x_y_size[0],
                    'minimum_extent': minimum_extent,
                    'available_ram': ram,
                    'output': temporary_raster,
                    'src_nodata': None,
                    'dst_nodata': nodata_value_set,
                    'resample_method': resampling,
                    'gdal_path': cfg.gdal_path,
                    'compress': True,
                    'compress_format': 'LZW'
                }
            )
            function_list.append(vector_to_raster_iter)
            virtual_raster_list.append(temporary_raster)
        cfg.multiprocess.run_iterative_process(
            function_list=function_list, argument_list=argument_list,
            min_progress=10, max_progress=75, message='converting to raster'
        )
        results = cfg.multiprocess.output
        output_raster_list = [r[0] for result in results for r in result]
        output_data_type = 'Int32'
        virtual_path = cfg.temp.temporary_file_path(name_suffix=cfg.vrt_suffix)
        raster_vector.create_virtual_raster_2_mosaic(
            input_raster_list=output_raster_list, output=virtual_path,
            src_nodata=nodata_value_set, dst_nodata=nodata_value_set,
            data_type=output_data_type,
            pixel_size=x_y_size, grid_reference=reference_path
        )
        # copy raster
        # (GDAL warns that the Value 2.14748e+09 in the source dataset will
        # be changed to 2.14748e+09 in the destination dataset to avoid being
        # treated as NoData, therefore no value is changed because it considers
        # the float value of 2147483647, while with lower integers it actually
        # changes the value; but the mode resampling doesn't work as expected
        # if input has nodata values, therefore the source nodata value must
        # be the same as destination nodata value)
        cfg.multiprocess.gdal_copy_raster(
            virtual_path, output_path, min_progress=75, max_progress=100
        )
    else:
        # perform conversion
        cfg.multiprocess.multiprocess_vector_to_raster(
            vector_path=vector_path, field_name=vector_field,
            output_path=temp_path, reference_raster_path=reference_path,
            output_format=output_format, nodata_value=nodata_value_set,
            background_value=nodata_value_set, burn_values=constant,
            compress=compress, compress_format=compress_format,
            x_y_size=t_pixel_size, all_touched=all_touched,
            available_ram=available_ram, minimum_extent=minimum_extent,
            min_progress=min_progress, max_progress=max_progress
        )
        cfg.logger.log.debug('temp_path: %s' % temp_path)
        if files_directories.is_file(temp_path):
            files_directories.move_file(
                in_path=temp_path, out_path=output_path
            )
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; output_path: %s' % output_path)
    return OutputManager(path=output_path)
