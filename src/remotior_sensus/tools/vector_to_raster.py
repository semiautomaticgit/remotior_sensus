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
"""
Vector to raster.

This tool allows for the conversion from vector polygons to raster.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> vector = rs.vector_to_raster(vector_path='file.gpkg',
    ...     output_path='vector.tif')
"""  # noqa: E501

from typing import Union, Optional

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.util import files_directories, shared_tools, raster_vector


def vector_to_raster(
        vector_path, align_raster: Union[str, BandSet, int],
        vector_field: Optional[str] = None,
        constant: Optional[int] = None,
        pixel_size: Optional[int] = None,
        output_path: Optional[str] = None,
        method: Optional[str] = None,
        area_precision: Optional[int] = 20, resample='mode',
        nodata_value: Optional[int] = 255,
        minimum_extent: Optional[bool] = True,
        extent_list: Optional[list] = None, output_format='GTiff',
        compress=None, compress_format=None,
        n_processes: Optional[int] = None, available_ram: Optional[int] = None,
        bandset_catalog: Optional[BandSetCatalog] = None,
) -> OutputManager:
    """Performs the conversion from vector to raster.

    This tool performs the conversion from vector polygons to raster.

    Args:
        vector_path: path of vector used as input.
        align_raster: optional string path of raster used for aligning output pixels and projections; it can also be a BandSet or an integer number of a BandSet in a Catalog.
        output_path: string of output path.
        vector_field: the name of the field used as reference value.
        constant: integer value used as reference for all the polygons.
        pixel_size: size of pixel of output raster.
        minimum_extent: if True, raster has the minimum vector extent; if False, the extent is the same as the align raster.
        extent_list: list of boundary coordinates left top right bottom.
        output_format: output format, default GTiff
        method: method of conversion, default pixel_center, other methods are all_touched for burning all pixels touched or area_based for burning values based on area proportion.
        area_precision: for area_based method, the higher the value, the more is the precision in area proportion calculation.
        resample: type for resample when method is area_based.
        compress: if True, compress the output raster.
        compress_format: compress format.
        nodata_value: value to be considered as nodata.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        bandset_catalog: BandSetCatalog object.

    Returns:
        object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - path = output path

    Examples:
        Perform the conversion to raster of a vector
            >>> vector_to_raster(vector_path='file.gpkg',output_path='vector.tif')
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=True
    )
    vector_path = files_directories.input_path(vector_path)
    if type(align_raster) is str:
        input_bands = [align_raster]
    else:
        input_bands = align_raster
    cfg.logger.log.debug('input_bands: %s' % str(input_bands))
    # prepare process files
    (input_raster_list, raster_info, nodata_list, name_list, warped,
     out_path_x, vrt_rx, reference_path, n_processes_x,
     output_list_x, vrt_list_x) = shared_tools.prepare_process_files(
        input_bands=input_bands, output_path=output_path,
        n_processes=n_processes, box_coordinate_list=extent_list,
        bandset_catalog=bandset_catalog
    )
    # prepare output
    temp_path = cfg.temp.temporary_file_path(name_suffix=cfg.vrt_suffix)
    if n_processes is None:
        n_processes = cfg.n_processes
    # perform conversion
    if compress is None:
        compress = cfg.raster_compression
    if compress_format is None:
        compress_format = 'DEFLATE21'
    if pixel_size is None:
        x_y_size = None
    else:
        x_y_size = [pixel_size, pixel_size]
    if vector_field is None and constant is None:
        constant = 1
    nodata_value_set = nodata_value
    min_progress = 1
    if method is None or method.lower() == 'pixel_center':
        all_touched = None
        area_based = None
        max_progress = 100
    elif method.lower() == 'all_touched':
        all_touched = True
        area_based = None
        max_progress = 100
    elif method.lower() == 'area_based':
        all_touched = None
        area_based = True
        compress = True
        minimum_extent = False
        nodata_value_set = None
        max_progress = 50
    else:
        all_touched = None
        area_based = None
        max_progress = 100
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
    # perform conversion
    cfg.multiprocess.multiprocess_vector_to_raster(
        vector_path=vector_path, field_name=vector_field,
        output_path=temp_path, reference_raster_path=reference_path,
        output_format=output_format, nodata_value=nodata_value_set,
        background_value=nodata_value, burn_values=constant,
        compress=compress, compress_format=compress_format,
        x_y_size=x_y_size, all_touched=all_touched, area_based=area_based,
        available_ram=available_ram, minimum_extent=minimum_extent,
        area_precision=area_precision, min_progress=min_progress,
        max_progress=max_progress
    )
    if output_path is None:
        output_path = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
    output_path = files_directories.output_path(output_path, cfg.tif_suffix)
    files_directories.create_parent_directory(output_path)
    # resample raster
    if method is not None and method.lower() == 'area_based':
        (gt, crs, crs_unit, xy_count, nd, number_of_bands, block_size,
         scale_offset, data_type) = raster_vector.raster_info(temp_path)
        # copy raster
        left = gt[0]
        top = gt[3]
        p_x_size = gt[1]
        p_y_size = abs(gt[5])
        right = gt[0] + gt[1] * xy_count[0]
        bottom = gt[3] + gt[5] * xy_count[1]
        if compress_format == 'DEFLATE21':
            compress_format = 'DEFLATE -co PREDICTOR=2 -co ZLEVEL=1'
        extra_params = ' -te %s %s %s %s -tr %s %s' % (
            left, bottom, right, top,
            p_x_size * area_precision, p_y_size * area_precision)
        min_progress = 51
        max_progress = 100
        raster_vector.gdal_warping(
            input_raster=temp_path, output=output_path, output_format='GTiff',
            resample_method=resample, compression=True,
            compress_format=compress_format, additional_params=extra_params,
            n_processes=n_processes, dst_nodata=nodata_value,
            min_progress=min_progress, max_progress=max_progress)
    else:
        if files_directories.is_file(temp_path):
            files_directories.move_file(
                in_path=temp_path, out_path=output_path
            )
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; output_path: %s' % output_path)
    return OutputManager(path=output_path)
