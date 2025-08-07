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
"""Raster zonal stats.

This tool allows for calculating statistics of a raster intersecting a vector.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> statistics = rs.raster_zonal_stats(
    ... raster_path='input_path', reference_path='input_vector_path',
    ... vector_field='field_name', output_path='output_path',
    ... stat_names=['Sum', 'Mean'])
      
"""  # noqa: E501

import io
from typing import Union, Optional

import numpy as np

from remotior_sensus.core import configurations as cfg, table_manager as tm
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.util import (
    files_directories, raster_vector, shared_tools, read_write_files
)
from remotior_sensus.core.processor_functions import (
    get_band_arrays, zonal_rasters
)


def raster_zonal_stats(
        raster_path: str, reference_path: str, vector_field: str = None,
        output_path: str = None, stat_names: Union[list, str] = None,
        stat_percentile: Optional[Union[int, str, list]] = None,
        extent_list: Optional[list] = None,
        n_processes: Optional[int] = None, available_ram: Optional[int] = None,
        progress_message: Optional[bool] = True
) -> OutputManager:
    """Raster zonal stats based on a polygon vector field.

    This tool allows for calculating statistics of a raster intersecting a 
    vector.
    For each unique value in the vector field, one ore more statistics can be
    calculated based on raster pixel values intersecting the vector polygons.
    The output is a csv file and a numpy records table including the statistics 
    for each unique value in the vector field.
    Available functions are:

    - Count
    - Max
    - Mean
    - Median
    - Min
    - Percentile
    - StandardDeviation
    - Sum

    Args:
        raster_path: path of raster used as input.
        reference_path: path of the vector or raster file used as reference 
            input.
        vector_field: the name of the vector field used as reference value.
        stat_names: statistic name as in configurations.statistics_list.
        stat_percentile: integer value for percentile parameter.
        output_path: string of output directory path.
        extent_list: list of boundary coordinates left top right bottom.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        progress_message: if True then start progress message, if False does 
            not start the progress message (useful if launched from other 
            tools).

    Returns:
        object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - path = output csv path
            - extra = {'table': table as numpy records}

    Examples:
        Perform the calculation
            >>> stats = raster_zonal_stats(
            ... raster_path='input_path', reference_path='input_vector_path',
            ... vector_field='field_name',output_path='output_path',
            ... stat_names=['Percentile', 'Max', 'Min'],stat_percentile=[1, 99]
            ... )
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=progress_message
    )
    raster_path = files_directories.input_path(raster_path)
    reference_path = files_directories.input_path(reference_path)
    vector, raster, reference_crs = raster_vector.raster_or_vector_input(
        reference_path
    )
    if extent_list is not None:
        if raster:
            # prepare process files
            prepared = shared_tools.prepare_process_files(
                input_bands=[raster_path, reference_path],
                overwrite=True, n_processes=n_processes,
                box_coordinate_list=extent_list,
                multiple_output=True, multiple_input=True,
                multiple_resolution=False
            )
            input_raster_list = prepared['input_raster_list']
            n_processes = prepared['n_processes']
            raster_path, reference_path = input_raster_list
        else:
            # prepare process files
            prepared = shared_tools.prepare_process_files(
                input_bands=[raster_path],
                overwrite=True, n_processes=n_processes,
                box_coordinate_list=extent_list,
                multiple_output=True, multiple_input=True,
                multiple_resolution=False
            )
            input_raster_list = prepared['input_raster_list']
            n_processes = prepared['n_processes']
            raster_path = input_raster_list[0]
    if n_processes is None:
        n_processes = cfg.n_processes
    # get statistic names
    if type(stat_names) is str:
        stat_names = [stat_names]
    numpy_functions = {}
    for stat_name in stat_names:
        stat_numpy = stat_n = None
        for i in cfg.statistics_list:
            if i[0].lower() == stat_name.lower():
                stat_numpy = i[1]
                stat_n = i[0]
                break
        if stat_numpy is None:
            cfg.logger.log.error(str(stat_name))
            cfg.messages.error(str(stat_name))
            cfg.progress.update(failed=True)
            return OutputManager(check=False)
        cfg.logger.log.debug('stat_numpy: %s' % str(stat_numpy))
        function_numpy = stat_numpy.replace('array', '_a')
        if cfg.stat_percentile in stat_numpy:
            try:
                if type(stat_percentile) is not list:
                    stat_percentile = [stat_percentile]
                for s in stat_percentile:
                    stat_percentile = int(s)
                    function_numpy_2 = function_numpy.replace(
                        cfg.stat_percentile, str(s)
                    )
                    numpy_functions['Percentile_%s'
                                    % stat_percentile] = function_numpy_2
            except Exception as err:
                cfg.logger.log.error(err)
                cfg.messages.error(str(err))
                cfg.progress.update(failed=True)
                return OutputManager(check=False)
        else:
            numpy_functions[stat_n] = function_numpy
    # open input with GDAL
    cfg.logger.log.debug('reference_path: %s' % reference_path)
    raster_crs = raster_vector.get_crs(raster_path)
    reference_crs = raster_vector.get_crs(reference_path)
    output_text = output_table = None
    # check crs
    same_crs = raster_vector.compare_crs(raster_crs, reference_crs)
    cfg.logger.log.debug('same_crs: %s' % str(same_crs))
    if raster:
        if not same_crs:
            t_pmd = cfg.temp.temporary_raster_path(extension=cfg.vrt_suffix)
            raster_path = cfg.multiprocess.create_warped_vrt(
                raster_path=raster_path, output_path=t_pmd,
                output_wkt=str(reference_crs)
            )
        # prepare process files
        prepared = shared_tools.prepare_process_files(
            input_bands=[raster_path, reference_path],
            overwrite=True, n_processes=n_processes,
            box_coordinate_list=extent_list,
            multiple_output=True, multiple_resolution=False
        )
        vrt_path = prepared['temporary_virtual_raster']
        # combination calculation
        cfg.multiprocess.run(
            raster_path=vrt_path, function=zonal_rasters,
            function_argument=numpy_functions,
            any_nodata_mask=True, unique_section=True,
            compress=cfg.raster_compression, n_processes=n_processes,
            available_ram=available_ram, keep_output_argument=True,
            progress_message='zonal raster',
            min_progress=10, max_progress=90
        )
        if cfg.multiprocess.output is False:
            return OutputManager(check=False)
        array_dictionary = cfg.multiprocess.output[0][0][1]
        output_text, output_table = _zonal_stats_table(
            array_dictionary, 'DN', int
        )
    else:
        if not same_crs:
            input_vector = cfg.temp.temporary_file_path(
                name_suffix=files_directories.file_extension(reference_path)
            )
            reference_path = raster_vector.reproject_vector(
                reference_path, input_vector, input_epsg=reference_crs,
                output_epsg=reference_crs
            )
        virtual_path_list = [raster_path]
        # find unique values of vector_field
        try:
            unique_values = raster_vector.get_vector_values(
                vector_path=reference_path, field_name=vector_field
            )
        except Exception as err:
            cfg.logger.log.error(str(err))
            return OutputManager(check=False)
        if len(unique_values) > 0:
            if n_processes > len(unique_values):
                n_processes = len(unique_values)
            unique_values_index_list = list(
                range(
                    0, len(unique_values), round(len(unique_values)
                                                 / n_processes)
                )
            )
            unique_values_index_list.append(len(unique_values))
            # build function argument list of dictionaries
            argument_list = []
            function_list = []
            for val_ids_index in range(1, len(unique_values_index_list)):
                # unique value id list
                signature_id_list = unique_values[
                                    unique_values_index_list[val_ids_index
                                                             - 1]:
                                    unique_values_index_list[val_ids_index]
                                    ]
                argument_list.append(
                    {
                        'signature_id_list': signature_id_list,
                        'roi_path': reference_path,
                        'field_name': vector_field,
                        'numpy_functions': numpy_functions,
                        'virtual_path_list': virtual_path_list,
                        'available_ram': available_ram,
                        # optional calc_data_type
                        'calc_data_type': None
                    }
                )
                function_list.append(get_band_arrays)
            cfg.multiprocess.run_iterative_process(
                function_list=function_list, argument_list=argument_list
            )
            # array for each roi
            cfg.multiprocess.multiprocess_roi_arrays()
            if cfg.multiprocess.output is False:
                return OutputManager(check=False)
            array_dictionary = cfg.multiprocess.output
            output_text, output_table = _zonal_stats_table(
                array_dictionary, vector_field, type(unique_values[0])
            )
    # save output text to file
    tbl_out = shared_tools.join_path(
        files_directories.parent_directory(output_path), '{}{}'.format(
            files_directories.file_name(output_path, suffix=False),
            cfg.csv_suffix
        )
    ).replace('\\', '/')
    read_write_files.write_file(output_text, tbl_out)
    cfg.logger.log.info('end; raster zonal stats: %s' % str(tbl_out))
    return OutputManager(path=tbl_out, extra={'table': output_table})


#
def _zonal_stats_table(stats_dict, field_name, field_type):
    """Creation of zonal stats table.

    Creates text for the output of zonal stats.

    Args:
        stats_dict: output dictionary of zonal stats

    Returns:
        The text of the zonal stats table and the table as numpy records.
    """
    text = []
    cv = cfg.comma_delimiter
    nl = cfg.new_line
    if field_type is str:
        dtype_list = [(field_name, 'U1024')]
    elif field_type is int:
        dtype_list = [(field_name, 'int16')]
    else:
        dtype_list = [(field_name, 'float64')]
    output_field_names = [field_name]
    for val in stats_dict:
        for key in stats_dict[val]:
            dtype_list.append((key, 'float64'))
            output_field_names.append(key)
        break
    table_values = []
    for val in stats_dict:
        value_list = [val]
        for key in stats_dict[val]:
            value_list.append(stats_dict[val][key])
        table_values.append(tuple(value_list))
    table = tm.create_generic_table(table_values, dtype_list)
    # create stream handler
    stream1 = io.StringIO()
    np.savetxt(stream1, table, delimiter=cv, fmt='%s')
    matrix_value = stream1.getvalue()
    for c in output_field_names:
        text.append(c)
        # noinspection PyUnresolvedReferences
        text.append(cv)
    text.pop(-1)
    text.append(nl)
    text.append(matrix_value)
    text.append(nl)
    joined_text = ''.join(text)
    return joined_text, table
