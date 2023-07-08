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
Raster report.

This tool allows for the calculation of a report providing information 
extracted from a raster.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> report = rs.raster_report(raster_path='file.tif',output_path='report.csv')
"""  # noqa: E501

import io
from typing import Optional

import numpy as np

from remotior_sensus.core import configurations as cfg, table_manager as tm
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.core.processor_functions import (
    raster_unique_values_with_sum
)
from remotior_sensus.util import (
    files_directories, raster_vector, read_write_files, shared_tools
)


def raster_report(
        raster_path: str, output_path: Optional[str] = None,
        nodata_value: Optional[int] = None, extent_list: Optional[list] = None,
        n_processes: Optional[int] = None, available_ram: Optional[int] = None
):
    """Calculation of a report providing information extracted from a raster.

    This tool allows for the calculation of a report providing information
    such as pixel count, area per class and percentage of the total area.
    The output is a csv file.
    This tool is intended for integer rasters.

    Args:
        raster_path: path of raster used as input.
        output_path: string of output path.
        nodata_value: value to be considered as nodata.
        extent_list: list of boundary coordinates left top right bottom.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.

    Returns:
        object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - path = output path

    Examples:
        Perform the report of a raster
            >>> raster_report(raster_path='file.tif',output_path='report.csv')
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=True
    )
    raster_path = files_directories.input_path(raster_path)
    if extent_list is not None:
        # prepare process files
        prepared = shared_tools.prepare_process_files(
            input_bands=[raster_path], output_path=output_path,
            n_processes=n_processes, box_coordinate_list=extent_list
        )
        n_processes = prepared['n_processes']
        raster_path = prepared['temporary_virtual_raster']
    if output_path is None:
        output_path = cfg.temp.temporary_file_path(name_suffix=cfg.csv_suffix)
    output_path = files_directories.output_path(output_path, cfg.csv_suffix)
    files_directories.create_parent_directory(output_path)
    (gt, crs, crs_unit, xy_count, nd, number_of_bands, block_size,
     scale_offset, data_type) = raster_vector.raster_info(raster_path)
    pixel_size_x = abs(gt[1])
    pixel_size_y = abs(gt[5])
    if n_processes is None:
        n_processes = cfg.n_processes
    # dummy bands for memory calculation
    dummy_bands = 2
    # multiprocess calculate unique values and sum
    cfg.multiprocess.run(
        raster_path=raster_path, function=raster_unique_values_with_sum,
        use_value_as_nodata=nodata_value, n_processes=n_processes,
        available_ram=available_ram, keep_output_argument=True,
        dummy_bands=dummy_bands,
        progress_message='unique values', min_progress=2, max_progress=99
    )
    cfg.progress.update(message='output table', step=99)
    # calculate sum of values
    cfg.multiprocess.multiprocess_sum_array(nodata_value)
    unique_val = cfg.multiprocess.output
    # create table
    table = _report_table(
        table=unique_val, crs_unit=crs_unit, pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y
    )
    # save combination to table
    read_write_files.write_file(table, output_path)
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; raster report: %s' % output_path)
    return OutputManager(path=output_path)


def _report_table(table, crs_unit, pixel_size_x, pixel_size_y):
    """Create text for tables."""
    cfg.logger.log.debug('start')
    total_sum = table['sum'].sum()
    text = []
    cv = cfg.comma_delimiter
    nl = cfg.new_line
    # table
    if 'degree' not in crs_unit:
        output_field_names = ['RasterValue', 'PixelSum', 'Percentage %',
                              'Area [%s^2]' % crs_unit]
        input_field_names = ['new_val', 'sum', 'percentage', 'area']
        cross_class = tm.calculate_multi(
            matrix=table, expression_string_list=[
                '"sum" * %s * %s' % (pixel_size_x, pixel_size_y),
                '100 * "sum" / %s' % total_sum],
            output_field_name_list=['area', 'percentage'],
            progress_message=False
        )
    else:
        output_field_names = ['RasterValue', 'PixelSum', 'Percentage %',
                              'Area not available']
        input_field_names = ['new_val', 'sum', 'percentage', 'area']
        # area is set to nan
        cross_class = tm.calculate_multi(
            matrix=table,
            expression_string_list=['100 * "sum" / %s' % total_sum,
                                    'np.nan * "sum"'],
            output_field_name_list=['percentage', 'area'],
            progress_message=False
        )
    redefined = tm.redefine_matrix_columns(
        matrix=cross_class, input_field_names=input_field_names,
        output_field_names=output_field_names, progress_message=False
    )
    # create stream handler
    stream1 = io.StringIO()
    np.savetxt(stream1, redefined, delimiter=cv, fmt='%1.2f')
    matrix_value = stream1.getvalue()
    for c in output_field_names:
        text.append(c)
        text.append(cv)
    text.pop(-1)
    text.append(nl)
    text.append(matrix_value.replace('.00', ''))
    text.append(nl)
    joined_text = ''.join(text)
    return joined_text
