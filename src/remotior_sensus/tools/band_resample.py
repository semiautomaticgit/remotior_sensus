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
Band resample.

This tool allows for resampling and reprojecting bands.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> rs.band_resample(input_bands=['path_1', 'path_2'],
    ... output_path='output_path')
"""  # noqa: E501

from typing import Union, Optional

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.util import shared_tools, raster_vector


def band_resample(
        input_bands: Union[list, int, BandSet],
        output_path: Optional[str] = None,
        epsg_code: Optional[str] = None,
        align_raster: Optional[Union[str, BandSet, int]] = None,
        overwrite: Optional[bool] = False,
        resampling: Optional[str] = None,
        nodata_value: Optional[int] = None,
        x_y_resolution: Optional[Union[list, int]] = None,
        resample_pixel_factor: Optional[float] = None,
        output_data_type: Optional[str] = None,
        same_extent: Optional[bool] = False,
        virtual_output: Optional[bool] = None,
        compress=None, compress_format=None,
        prefix: Optional[str] = '',
        extent_list: Optional[list] = None,
        n_processes: Optional[int] = None,
        available_ram: Optional[int] = None,
        bandset_catalog: Optional[BandSetCatalog] = None
):
    """Performs band resample and reprojection.

    This tool performs the resampling and reprojection of raster bands. 
    Available resampling methods are:

        - nearest_neighbour
        - average
        - sum
        - maximum
        - minimum
        - mode
        - median
        - first_quartile
        - third_quartile

    Args:
        input_bands: input of type BandSet or list of paths or integer
            number of BandSet.
        output_path: string of output path directory or list of paths.
        epsg_code: optional EPSG code for output.
        align_raster: string path of raster used for aligning output pixels and projections.
        overwrite: if True, output overwrites existing files.
        resampling: method of resample such as nearest_neighbour (default), average, sum, maximum, minimum, mode, median, first_quartile, third_quartile.
        nodata_value: value to be considered as nodata.
        x_y_resolution: integer pixel size of output raster or pixel size as list of x, y.
        resample_pixel_factor: define output resolution by multiplying original pixel size to this value.
        output_data_type: optional raster output data type, if None the data type is the same as input raster.
        same_extent: if True, output extent is the same as align_raster.
        virtual_output: if True (and output_path is directory), save output as virtual raster of multiprocess parts.
        prefix: optional string for output name prefix.
        extent_list: list of boundary coordinates left top right bottom.
        compress: if True, compress output.
        compress_format: format of compressions such as LZW or DEFLATE.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        bandset_catalog: optional type BandSetCatalog for BandSet number.

    Returns:
        Object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - paths = output list

    Examples:
        Perform the band resample
            >>> band_resample(input_bands=['path_1', 'path_2'],
            ... output_path='output_path', resampling='mode',
            ... resample_pixel_factor=2)
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=True
    )
    if resampling == 'nearest_neighbour':
        resample = 'near'
    elif resampling == 'average':
        resample = 'average'
    elif resampling == 'sum':
        resample = 'sum'
    elif resampling == 'maximum':
        resample = 'max'
    elif resampling == 'minimum':
        resample = 'min'
    elif resampling == 'mode':
        resample = 'mode'
    elif resampling == 'median':
        resample = 'med'
    elif resampling == 'first_quartile':
        resample = 'q1'
    elif resampling == 'third_quartile':
        resample = 'q3'
    else:
        resample = 'near'
    # prepare process files
    prepared = shared_tools.prepare_process_files(
        input_bands=input_bands, output_path=output_path, overwrite=overwrite,
        n_processes=n_processes, box_coordinate_list=extent_list,
        bandset_catalog=bandset_catalog, prefix=prefix,
        multiple_output=True, multiple_input=True,
        virtual_output=virtual_output
    )
    input_raster_list = prepared['input_raster_list']
    raster_info = prepared['raster_info']
    n_processes = prepared['n_processes']
    nodata_list = prepared['nodata_list']
    output_list = prepared['output_list']
    if type(x_y_resolution) is not list:
        x_y_resolution = [x_y_resolution, x_y_resolution]
    if resample_pixel_factor is None:
        resample_pixel_factor = 1
    try:
        resample_pixel_factor = float(resample_pixel_factor)
    except Exception as err:
        cfg.logger.log.error(str(err))
        cfg.messages.error(str(err))
        return OutputManager(check=False)
    if epsg_code is None:
        epsg = False
    else:
        epsg = None
    resample_parameters = None
    if align_raster is None:
        left = top = right = bottom = p_x = p_y = align_sys_ref = None
    else:
        # raster extent and pixel size
        (gt, crs, crs_unit, xy_count, nd, number_of_bands, block_size,
         scale_offset, data_type) = raster_vector.raster_info(align_raster)
        # copy raster
        left = gt[0]
        top = gt[3]
        right = gt[0] + gt[1] * xy_count[0]
        bottom = gt[3] + gt[5] * xy_count[1]
        p_x = gt[1]
        p_y = abs(gt[5])
        # check projections
        align_sys_ref = raster_vector.auto_set_epsg(align_raster)
        epsg = False
        new_p_x = p_x * resample_pixel_factor
        new_p_y = p_y * resample_pixel_factor
        resample_parameters = '-tr %s %s -te %s %s %s %s' % (
            str(new_p_x), str(new_p_y), str(left), str(bottom), str(right),
            str(top))
    if compress_format == 'DEFLATE21':
        compress_format = 'DEFLATE -co PREDICTOR=2 -co ZLEVEL=1'
    min_progress = 1
    one_progress = int((99 - 1) / len(input_raster_list))
    max_progress = one_progress
    for band in range(0, len(input_raster_list)):
        # raster extent and pixel size
        (left_input, top_input, right_input, bottom_input, p_x_input,
         p_y_input, proj_input,
         unit_input) = raster_vector.image_geotransformation(
            input_raster_list[band]
        )
        if output_data_type is None:
            output_data_type = raster_info[band][8]
        # calculate minimal extent
        if align_raster is not None:
            input_sys_ref = raster_vector.get_spatial_reference(proj_input)
            left_projected, top_projected = (
                raster_vector.project_point_coordinates(
                    left_input, top_input, input_sys_ref, align_sys_ref
                )
            )
            right_projected, bottom_projected = (
                raster_vector.project_point_coordinates(
                    right_input, bottom_input, input_sys_ref, align_sys_ref
                )
            )
            if same_extent is False:
                # minimum extent
                if left_projected < left:
                    left_output = left - int(
                        2 + (left - left_projected) / p_x
                    ) * p_x
                else:
                    left_output = left + int(
                        (left_projected - left) / p_x - 2
                    ) * p_x
                if right_projected > right:
                    right_output = right + int(
                        2 + (right_projected - right) / p_x
                    ) * p_x
                else:
                    right_output = right - int(
                        (right - right_projected) / p_x - 2
                    ) * p_x
                if top_projected > top:
                    top_output = top + int(
                        2 + (top_projected - top) / p_y
                    ) * p_y
                else:
                    top_output = top - int(
                        (top - top_projected) / p_y - 2
                    ) * p_y
                if bottom_projected > bottom:
                    bottom_output = bottom + int(
                        (bottom_projected - bottom) / p_y - 2
                    ) * p_y
                else:
                    bottom_output = bottom - int(
                        2 + (bottom - bottom_projected) / p_y
                    ) * p_y
            else:
                left_output = left
                top_output = top
                right_output = right
                bottom_output = bottom
            resample_parameters = '-tr %s %s -te %s %s %s %s ' % (
                str(p_x), str(p_y), str(left_output), str(bottom_output),
                str(right_output), str(top_output))
        # use epsg
        elif epsg_code is not None:
            # spatial reference
            resample_parameters = None
            try:
                epsg = int(epsg_code)
            except Exception as err:
                cfg.logger.log.error(str(err))
                cfg.messages.error(str(err))
                return OutputManager(check=False)
            if same_extent is False:
                try:
                    resample_parameters = '-tr %s %s' % (
                        str(float(x_y_resolution[0])),
                        str(float(x_y_resolution[1])))
                except Exception as err:
                    str(err)
            else:
                left_output = left
                top_output = top
                right_output = right
                bottom_output = bottom
                try:
                    resample_parameters = '-tr %s %s -te %s %s %s %s ' % (
                        str(float(x_y_resolution[0])),
                        str(float(x_y_resolution[1])),
                        str(left_output), str(bottom_output),
                        str(right_output), str(top_output)
                    )
                except Exception as err:
                    str(err)
        # resample
        else:
            if epsg is False:
                epsg = None
            p_x = p_x_input * resample_pixel_factor
            p_y = p_y_input * resample_pixel_factor
            if same_extent is False:
                try:
                    resample_parameters = '-tr %s %s' % (str(p_x), str(p_y))
                except Exception as err:
                    str(err)
            else:
                left_output = left
                top_output = top
                right_output = right
                bottom_output = bottom
                try:
                    resample_parameters = '-tr %s %s -te %s %s %s %s ' % (
                        str(p_x), str(p_y), str(left_output),
                        str(bottom_output), str(right_output), str(top_output)
                    )
                except Exception as err:
                    str(err)
        if epsg is not None:
            if epsg is False:
                epsg_output = proj_input
            else:
                epsg_output = 'epsg:%s' % str(epsg)
        else:
            epsg_output = None
        if nodata_value is None:
            nodata_value = nodata_list[band]
        cfg.logger.log.debug('resample_parameters: %s' % resample_parameters)
        # calculation
        raster_vector.gdal_warping(
            input_raster=input_raster_list[band],
            output=output_list[band],
            output_format='GTiff',
            resample_method=resample,
            t_srs=epsg_output,
            compression=compress,
            compress_format=compress_format,
            additional_params=resample_parameters,
            raster_data_type=output_data_type,
            dst_nodata=nodata_value,
            available_ram=available_ram,
            n_processes=n_processes,
            min_progress=min_progress, max_progress=max_progress
        )
        min_progress = max_progress
        max_progress += one_progress
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; band resample: %s' % str(output_list))
    return OutputManager(paths=output_list)
