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
Raster edit.

This tool allows for the direct editing of a raster band.
Warning, this tool edits the original input raster. 

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> edit = rs.raster_edit(raster_path='input_path', 
    ... vector_path='vector_path')
"""  # noqa: E501

from typing import Union, Optional

import numpy as np

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.util import (
    files_directories, raster_vector, shared_tools
)
from remotior_sensus.core.processor_functions import edit_raster


def raster_edit(
        raster_path: str, vector_path: Optional[str] = None,
        field_name: Optional[str] = None,
        expression: Optional[str] = None,
        constant_value: Optional[Union[int, float]] = None,
        old_array: Optional[np.ndarray] = None,
        column_start: Optional[int] = None, row_start: Optional[int] = None,
        available_ram: Optional[int] = None,
        progress_message: Optional[bool] = True
) -> OutputManager:
    """Edit a raster band based on a vector file.

    This tool allows for editing the values of a raster using a vector file.
    All pixel intersecting the vector will be edited with a new value.
    New pixel values can be a constant value or defined by a custom expression defining a condition.
    The input raster is directly edited, without producing a new file.

    Args:
        raster_path: path of raster used as input.
        vector_path: string of vector path (optional for undo operation).
        field_name: string of vector field name.
        expression: optional conditional expression; pixels are edited if 
            intersecting the vector and the expression is True; the variable 
            "raster" is used to refer to the raster pixel values;
            e.g. where("raster" > 500, 10, "raster").
        constant_value: optional new value for pixels.
        old_array: optional previous array for undo operation.
        column_start: optional starting column for undo operation.
        row_start: optional starting row for undo operation.
        available_ram: number of megabytes of RAM available to processes.
        progress_message: if True then start progress message, if False does 
            not start the progress message (useful if launched from other tools).

    Returns:
        object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - extra = {'old_array': previous array for undo operation, 
              'column_start': starting column for undo operation, 
              'row_start': starting row for undo operation}

    Examples:
        Perform the edit of a raster
            >>> edit_1 = raster_edit(raster_path='input_path', 
            ... vector_path='output_path')
            
        Perform the undo of edit of a raster
            >>> edit_2 = raster_edit(raster_path='input_path', 
            ... column_start=edit_1.extra['column_start'],
            ... row_start=edit_1.extra['row_start'],
            ... old_array=edit_1.extra['old_array']
            ... )
            
        Perform the edit of a raster with condition
            >>> edit_3 = raster_edit(raster_path='input_path', 
            ... vector_path='output_path', 
            ... expression='where("raster" > 500, 10, "raster")'
            ... )
            
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=progress_message
    )
    raster_path = files_directories.input_path(raster_path)
    # prepare process files
    prepared = shared_tools.prepare_process_files(
        input_bands=[raster_path], output_path=None
    )
    raster_info = prepared['raster_info']
    input_raster_list = prepared['input_raster_list']
    if vector_path is not None:
        # get vector info
        vector, raster, mask_crs = raster_vector.raster_or_vector_input(
            vector_path
        )
        # check crs
        same_crs = raster_vector.compare_crs(raster_info[0][1], mask_crs)
        if not same_crs:
            # project vector to raster crs
            t_vector = cfg.temp.temporary_file_path(
                name_suffix=cfg.gpkg_suffix
            )
            try:
                raster_vector.reproject_vector(
                    vector_path, t_vector, raster_info[0][1], mask_crs
                )
            except Exception as err:
                cfg.logger.log.error(str(err))
                cfg.messages.error(str(err))
                cfg.progress.update(failed=True)
                return OutputManager(check=False)
            vector_path = t_vector
    # convert vector to raster
    vector_raster = cfg.temp.temporary_raster_path(
        extension=cfg.tif_suffix
    )
    if field_name is None:
        burn_values = 1
    else:
        burn_values = None
    # perform conversion
    cfg.multiprocess.multiprocess_vector_to_raster(
        vector_path=vector_path, output_path=vector_raster,
        field_name=field_name, burn_values=burn_values,
        nodata_value=0, background_value=0,
        reference_raster_path=input_raster_list[0], minimum_extent=True,
        available_ram=available_ram
    )
    if expression is not None:
        cfg.logger.log.debug('expression: %s' % expression)
        check, expression = check_expression(expression, constant_value)
        if check is False:
            cfg.logger.log.error('invalid expression')
            cfg.messages.error('invalid expression')
            cfg.progress.update(failed=True)
            return OutputManager(check=False)
        cfg.logger.log.debug('expression: %s' % expression)
    # run edit raster
    argument_list = [{
        'input_raster': input_raster_list[0], 'vector_raster': vector_raster,
        'constant_value': constant_value, 'expression': expression,
        'old_array': old_array, 'column_start': column_start,
        'row_start': row_start, 'gdal_path': cfg.gdal_path
    }]
    function_list = [edit_raster]
    cfg.multiprocess.run_iterative_process(
        function_list=function_list, argument_list=argument_list
    )
    output = cfg.multiprocess.output
    if output is not False:
        old_array, column_start, row_start = output[0]
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; raster edit: %s' % str(raster_path))
    return OutputManager(extra={
        'old_array': old_array, 'column_start': column_start,
        'row_start': row_start,
    })


# check the expression
# noinspection PyShadowingBuiltins
def check_expression(expression, constant_value):
    # expose numpy functions
    log = np.log
    _log = log
    log10 = np.log10
    _log10 = log10
    sqrt = np.sqrt
    _sqrt = sqrt
    cos = np.cos
    _cos = cos
    arccos = np.arccos
    _arccos = arccos
    sin = np.sin
    _sin = sin
    arcsin = np.arcsin
    _arcsin = arcsin
    tan = np.tan
    _tan = tan
    arctan = np.arctan
    _arctan = arctan
    exp = np.exp
    _exp = exp
    min = np.nanmin
    _min = min
    max = np.nanmax
    _max = max
    sum = np.nansum
    _sum = sum
    percentile = np.nanpercentile
    _percentile = percentile
    median = np.nanmedian
    _median = median
    mean = np.nanmean
    _mean = mean
    std = np.nanstd
    _std = std
    where = np.where
    _where = where
    nan = np.nan
    _nan = nan
    # replace variables
    expression = expression.replace(
        '%s%s%s' % (cfg.variable_band_quotes, cfg.variable_vector_name,
                    cfg.variable_band_quotes), str(constant_value)
    )
    expression = expression.replace(
        '%s%s%s' % (cfg.variable_band_quotes, cfg.variable_raster_name,
                    cfg.variable_band_quotes), cfg.variable_raster_name
    )
    conditional_exp = 'np.where(%s> 0 , %s, %s)' % (
        cfg.variable_vector_name, expression, cfg.variable_raster_name
    )
    # test array
    _test_array = np.arange(9).reshape(3, 3)
    expression_copy = conditional_exp
    expression_copy = expression_copy.replace(cfg.variable_vector_name,
                                              '_test_array')
    expression_copy = expression_copy.replace(cfg.variable_raster_name,
                                              '_test_array')
    try:
        eval(expression_copy)
        return True, conditional_exp
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False, None
