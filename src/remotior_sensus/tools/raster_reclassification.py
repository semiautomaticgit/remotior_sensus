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
Raster reclassification.

This tool allows for the reclassification of a raster based on
a reclassification table.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> reclassification = rs.raster_reclassification(raster_path='file1.tif',
    ... output_path='output.tif',
    ... reclassification_table=[[1, -10], ['nan', 6000]])
"""

from typing import Union, Optional

import numpy as np

from remotior_sensus.core import configurations as cfg, table_manager as tm
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.core.processor_functions import (
    raster_unique_values_with_sum, reclassify_raster
)
from remotior_sensus.util import files_directories, raster_vector, shared_tools


def raster_reclassification(
        raster_path: str, output_path: Optional[str] = None,
        overwrite: Optional[bool] = False,
        reclassification_table: Optional[Union[list, np.array]] = None,
        csv_path: Optional[str] = None, separator: Optional[str] = ',',
        output_data_type: Optional[str] = None,
        extent_list: Optional[list] = None, n_processes: Optional[int] = None,
        available_ram: Optional[int] = None,
        virtual_output: Optional[bool] = None
) -> OutputManager:
    """Performs raster reclassification.

    This tool reclassifies a raster based on a reclassification table.
    The reclassification table is defined by two columns: old values and new
    values.
    Old values define the values of the raster to be reclassified to new
    values.
    Old values can be integer numbers or conditions selecting ranges of values.

    Args:
        raster_path: path of raster used as input.
        output_path: string of output path.
        overwrite: if True, output overwrites existing files.
        reclassification_table: table of values for reclassification or list
            of values [[old_value, new_value], ...]; if None, csv_path is used.
        csv_path: path to a csv file containing the reclassification table; 
            used if reclassification_table is None.
        separator: separator character for csv file; default is comma separated.
        output_data_type: set output data type such as 'Float32' or 'Int32'; 
            if None, input data type is used.
        extent_list: list of boundary coordinates left top right bottom.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        virtual_output: if True (and output_path is directory), save output
            as virtual raster of multiprocess parts.

    Returns:
        Object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - path = output path

    Examples:
        Perform the reclassification using a csv file containing the reclassification values
            >>> # import Remotior Sensus and start the session
            >>> import remotior_sensus
            >>> rs = remotior_sensus.Session()
            >>> # start the process
            >>> reclassification = rs.raster_reclassification(raster_path='file1.tif',output_path='output.tif',csv_path='file.csv')
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=True
    )
    cfg.logger.log.debug('raster_path: %s' % str(raster_path))
    if n_processes is None:
        n_processes = cfg.n_processes
    raster_path = files_directories.input_path(raster_path)
    if extent_list is not None:
        # prepare process files
        prepared = shared_tools.prepare_process_files(
            input_bands=[raster_path], output_path=output_path,
            overwrite=overwrite, n_processes=n_processes,
            box_coordinate_list=extent_list
        )
        n_processes = prepared['n_processes']
        raster_path = prepared['temporary_virtual_raster']
    (gt, crs, crs_unit, xy_count, nd, number_of_bands, block_size,
     scale_offset, data_type) = raster_vector.raster_info(raster_path)
    if output_data_type is None:
        output_data_type = data_type
    # check output path
    out_path, vrt_r = files_directories.raster_output_path(
        output_path, virtual_output, overwrite=overwrite
    )
    # reclassification table
    if reclassification_table is None:
        if csv_path is None:
            cfg.logger.log.error('unable to reclassify')
            cfg.messages.error('unable to reclassify')
            return OutputManager(check=False)
        else:
            table = _import_reclassification_table(
                csv_path=csv_path, separator=separator
                )
            if table.check:
                reclassification_table = table.extra['table']
            else:
                cfg.logger.log.error('unable to reclassify')
                cfg.messages.error('unable to reclassify')
                return OutputManager(check=False)
    # if reclassification list
    elif type(reclassification_table) is list:
        table = _list_to_reclassification_table(reclassification_table)
        if table.check:
            reclassification_table = table.extra['table']
        else:
            cfg.logger.log.error('unable to reclassify')
            cfg.messages.error('unable to reclassify')
            return OutputManager(check=False)
    # check output data type
    cfg.logger.log.debug('output_data_type: %s' % str(output_data_type))
    if (output_data_type.lower() == cfg.uint32_dt.lower()
            or output_data_type.lower() == cfg.uint16_dt.lower()):
        new_values = reclassification_table.new_value
        for i in new_values:
            try:
                if int(i) < 0:
                    output_data_type = cfg.int32_dt
                    cfg.logger.log.debug(
                        'output_data_type: %s' % str(output_data_type)
                    )
                    break
            except Exception as err:
                str(err)
    # dummy bands for memory calculation
    dummy_bands = 4
    # process calculation
    cfg.multiprocess.run(
        raster_path=raster_path, function=reclassify_raster,
        function_argument=reclassification_table,
        function_variable=cfg.variable_raster_name,
        output_raster_path=out_path, n_processes=n_processes,
        available_ram=available_ram, output_data_type=output_data_type,
        output_nodata_value=nd, compress=cfg.raster_compression,
        virtual_raster=vrt_r, progress_message='processing raster',
        dummy_bands=dummy_bands, min_progress=1, max_progress=99
    )
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; band reclassification: %s' % str(out_path))
    return OutputManager(path=out_path)


def unique_values_table(
        raster_path: str, n_processes: Optional[int] = None,
        available_ram: Optional[int] = None,
        incremental: Optional[bool] = False,
        progress_message: Optional[bool] = True
):
    """Calculate unique values from raster."""
    cfg.logger.log.debug('start')
    if progress_message:
        cfg.progress.update(
            process=__name__.split('.')[-1].replace('_', ' '),
            message='starting', start=True
        )
    cfg.logger.log.debug('raster_path: %s' % str(raster_path))
    if n_processes is None:
        n_processes = cfg.n_processes
    # dummy bands for memory calculation
    dummy_bands = 2
    cfg.multiprocess.run(
        raster_path=raster_path, function=raster_unique_values_with_sum,
        keep_output_argument=True, n_processes=n_processes,
        available_ram=available_ram, dummy_bands=dummy_bands,
        progress_message='unique values', min_progress=1, max_progress=99
    )
    # calculate sum of values
    cfg.multiprocess.multiprocess_sum_array()
    values = cfg.multiprocess.output
    # add old value field
    if not incremental:
        table = tm.append_field(
            values['new_val'], cfg.old_value, values['new_val'], 'U1024'
        )
    else:
        table = tm.append_field(
            values['new_val'], cfg.old_value,
            np.arange(1, values['new_val'].shape[0] + 1, 1),
            'U1024'
        )
    # add new value field
    table = tm.append_field(table, cfg.new_value, values['new_val'], 'U1024')
    table = tm.redefine_matrix_columns(
        matrix=table, input_field_names=[cfg.old_value, cfg.new_value],
        output_field_names=[cfg.old_value, cfg.new_value]
    )
    cfg.progress.update(end=True)
    cfg.logger.log.debug('end')
    return table


def _list_to_reclassification_table(input_list):
    """# create reclassification table from list of values
    [(old value, new value),  (old value, new value)].

    configurations.variable_raster_name can be used as variable in old value
    e.g.
    raster > 0.3
    """
    # table of values
    table = None
    dtype_list = [(cfg.old_value, 'U1024'), (cfg.new_value, 'U1024')]
    # test array
    _x = np.ones(3)
    for i in range(len(input_list)):
        # check new value
        try:
            new_value = str(input_list[i][1]).lower().replace('null', 'nan')
            eval(new_value.replace('nan', 'np.nan'))
        except Exception as err:
            cfg.logger.log.error(str(err))
            cfg.messages.error(str(err))
            return OutputManager(check=False)
        # check old value
        try:
            old_value = int(input_list[i][0])
        except Exception as err:
            str(err)
            try:
                old_value = input_list[i][0].lower().replace('null', 'nan')
                test_old_value = old_value.replace(
                    cfg.variable_raster_name, '_x'
                ).replace('nan', 'np.nan')
                eval(test_old_value)
            except Exception as err:
                str(err)
                cfg.logger.log.error(
                    'unable to process value %s' % str(input_list[i][0])
                )
                cfg.messages.error(
                    'unable to process value %s' % str(input_list[i][0])
                )
                return OutputManager(check=False)
        # add to table
        if table is None:
            table = np.array(
                (str(old_value), str(new_value)), dtype=dtype_list
            )
        else:
            table = tm.append_values_to_table(
                table, [str(old_value), str(new_value)]
            )
    return OutputManager(extra={'table': table})


def _import_reclassification_table(csv_path, separator=','):
    """Imports reclassification table from csv with two columns of values
    (old value, new value)."""
    # open csv
    csv = tm.open_file(
        file_path=csv_path, separators=separator,
        field_names=[cfg.old_value, cfg.new_value], progress_message=False,
        skip_first_line=False
        )
    # table of values
    table = None
    dtype_list = [(cfg.old_value, 'U1024'), (cfg.new_value, 'U1024')]
    # test array
    _x = np.ones(3)
    for i in range(len(csv)):
        # check new value
        try:
            new_value = str(csv[cfg.new_value][i]).lower().replace(
                'null', 'nan'
            )
            eval(new_value.replace('nan', 'np.nan'))
        except Exception as err:
            cfg.logger.log.error(str(err))
            cfg.messages.error(str(err))
            return OutputManager(check=False)
        # check old value
        try:
            old_value = int(csv[cfg.old_value][i])
        except Exception as err:
            str(err)
            try:
                old_value = csv[cfg.old_value][i].lower().replace(
                    'null', 'nan'
                )
                test_old_value = old_value.replace(
                    cfg.variable_raster_name, '_x'
                )
                eval(test_old_value)
            except Exception as err:
                str(err)
                cfg.logger.log.error(
                    'unable to process value %s' % str(csv[cfg.old_value][i])
                )
                cfg.messages.error(
                    'unable to process value %s' % str(csv[cfg.old_value][i])
                )
                return OutputManager(check=False)
        # add to table
        if table is None:
            table = np.array(
                (str(old_value), str(new_value)), dtype=dtype_list
            )
        else:
            table = tm.append_values_to_table(
                table, [str(old_value), str(new_value)]
            )
    return OutputManager(extra={'table': table})
