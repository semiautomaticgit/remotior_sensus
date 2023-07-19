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
"""Table manager.

This tool allows for managing table data as NumPy structured arrays.
It includes functions for field calculation, join and pivot tables.
Also, functions to manage tables used in other tools are included.
Tables can be exported to csv files.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # open a file
    >>> file1 = 'file1.csv'
    >>> table1 = rs.table_manager.open_file('file1.csv', field_names=['field1', 'field2'])
    >>> # perform a calculation
    >>> calculation = rs.table_manager.calculate(
    ... matrix=table1, expression_string='"field1" * 1.5',
    ... output_field_name='calc'
    ... )
    >>> # export the table to csv
    >>> rs.table_manager.export_table(
    ... matrix=calculation, output_path='output.csv',
    ... fields=['field1', 'calc'], separator=';', decimal_separator='.'
    ... )
"""  # noqa: E501

import io
import itertools
import os
from typing import Union, Optional

import numpy as np
from numpy.lib import recfunctions as rfn

from remotior_sensus.core import configurations as cfg
from remotior_sensus.util import files_directories

# test issues
try:
    if cfg.gdal_path is not None:
        os.add_dll_directory(cfg.gdal_path)
except Exception as error:
    cfg.logger.log.error(str(error))

try:
    from osgeo import ogr
except Exception as error:
    cfg.logger.log.error(str(error))

try:
    assert rfn.assign_fields_by_name
    assert rfn.require_fields
except Exception as error:
    cfg.logger.log.error(str(error))


def open_file(
        file_path: str, separators: Optional[Union[str, list]] = None,
        field_names: Optional[list] = None,
        skip_first_line: Optional[bool] = True,
        progress_message: Optional[bool] = True
) -> np.recarray:
    """Opens a file.

    Opens a file by reading the content and creating a table (which is a 
    NumPy structured array).
    File formats csv and dbf are supported.

    Args:
        file_path: path of output file.
        separators: list of characters used as separator in csv files; default is tab and comma.
        field_names: list of strings to be used as field names.
        skip_first_line: skip the first line in csv files, in case the first line contains field names.
        progress_message: if True then start progress message; 
            if False does not start the progress message (useful if launched from other tools).

    Returns:
        Table as NumPy structured array.

    Examples:
        Open a file
            >>> table_1 = open_file('file.csv')
    """  # noqa: E501
    if files_directories.file_extension(file_path) == cfg.dbf_suffix:
        table = _open_dbf(
            file_path=file_path, field_name_list=field_names,
            progress_message=progress_message
            )
    else:
        table = _open_csv(
            file_path=file_path, separators=separators,
            field_name_list=field_names, progress_message=progress_message,
            skip_first_line=skip_first_line
            )
    return table


# open dbf file
def _open_dbf(file_path, field_name_list=None, progress_message=True):
    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(process='open_dbf', message='starting', start=True)
    cfg.logger.log.debug('file_path: %s' % file_path)
    input_file = ogr.Open(file_path)
    i_layer = input_file.GetLayer()
    i_layer_def = i_layer.GetLayerDefn()
    field_count = i_layer_def.GetFieldCount()
    # fields
    dtype_list = []
    for c in range(field_count):
        tp = i_layer_def.GetFieldDefn(c).GetTypeName()
        width = i_layer_def.GetFieldDefn(c).GetWidth()
        if tp == 'Integer':
            dtype = 'int64'
        elif tp == 'Integer64':
            dtype = 'int64'
        elif tp == 'Real':
            dtype = 'float64'
        else:
            dtype = 'U%s' % width
        try:
            dtype_list.append((field_name_list[c], dtype))
        except Exception as err:
            str(err)
            dtype_list.append(
                (i_layer_def.GetFieldDefn(c).GetNameRef(), dtype)
            )
    mat_list = []
    i = 0
    feature_count = i_layer.GetFeatureCount()
    i_feature = i_layer.GetNextFeature()
    while i_feature:
        if cfg.action is True:
            i += 1
            progress = int(100 * i / feature_count)
            if progress_message:
                cfg.progress.update(message='opening file', step=progress)
            else:
                cfg.progress.update(message='opening file',
                                    percentage=progress)
            row = []
            for c in range(field_count):
                f = i_feature.GetField(c)
                row.append(f)
            mat_list.append(row)
            i_feature = i_layer.GetNextFeature()
    array_ml = np.array(mat_list)
    rec_array = np.rec.fromarrays(array_ml.T, dtype=dtype_list)
    cfg.logger.log.info('end')
    if progress_message:
        cfg.progress.update(end=True)
    return rec_array


# open csv file
def _open_csv(
        file_path, separators=None, field_name_list=None,
        progress_message=True, skip_first_line=True
):
    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(process='open_csv', message='starting', start=True)
    if separators is None:
        separators = [cfg.tab_delimiter, cfg.comma_delimiter]
    cfg.logger.log.debug('file_path: %s' % file_path)
    with open(file_path, 'r') as file:
        # read file
        text = file.read()
        split_new_line = text.split(cfg.new_line)
        field_list = []
        data_list = []
        line_count = len(split_new_line)
        min_progress = 1
        max_progress = 80
        i = 0
        for line in split_new_line:
            i += 1
            if progress_message:
                cfg.progress.update(
                    message='opening file', step=i, steps=line_count,
                    minimum=min_progress,
                    maximum=max_progress, percentage=int(100 * i / line_count)
                )
            else:
                cfg.progress.update(
                    message='opening file',
                    percentage=int(100 * i / line_count)
                )
            if len(line) > 0:
                feature_list = [line]
                for s in separators:
                    features = []
                    for line_x in feature_list:
                        features.extend(line_x.split(s))
                    feature_list = features
                if i == 1:
                    if skip_first_line:
                        field_list = feature_list
                    else:
                        data_list.append(feature_list)
                        for f in range(1, len(feature_list) + 1):
                            field_list.append('field%s' % f)
                else:
                    data_list.append(feature_list)
    # create array
    array_data = np.array(data_list)
    dtype_list = []
    min_progress = 80
    max_progress = 99
    field_count = len(field_list)
    for t in range(len(field_list)):
        if progress_message:
            cfg.progress.update(
                message='processing data', step=t + 1, steps=field_count,
                minimum=min_progress, maximum=max_progress,
                percentage=int(100 * (t + 1) / line_count)
            )
        else:
            cfg.progress.update(
                message='processing data',
                percentage=int(100 * (t + 1) / line_count)
            )
        # check integer or float
        try:
            a = array_data[::, t].astype(np.float64)
            try:
                b = array_data[::, t].astype(np.int64)
                if np.nansum(a != b) == 0:
                    cfg.logger.log.debug('integer field: %s' % t)
                try:
                    dtype_list.append((field_name_list[t], 'int64'))
                except Exception as err:
                    str(err)
                    dtype_list.append((field_list[t], 'int64'))
            except Exception as err:
                str(err)
                cfg.logger.log.debug('float field: %s' % t)
                try:
                    dtype_list.append((field_name_list[t], 'float64'))
                except Exception as err:
                    str(err)
                    dtype_list.append((field_list[t], 'float64'))
        except Exception as err:
            str(err)
            cfg.logger.log.debug('string field: %s' % t)
            try:
                dtype_list.append((field_name_list[t], 'U64'))
            except Exception as err:
                str(err)
                dtype_list.append((field_list[t], 'U64'))
    rec_array = np.rec.fromarrays(array_data.T, dtype=dtype_list)
    if progress_message:
        cfg.progress.update(end=True)
    cfg.logger.log.info('end')
    return rec_array


# join matrices without duplicates
def join_matrices(
        matrix1, matrix2, field1_name, field2_name, join_type='leftouter',
        matrix1_postfix='_m1',
        matrix2_postfix='_m2', use_mask=False, progress_message=True
):
    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(
            process='join_matrices', message='starting', start=True
        )
    renamed_matrix1 = rfn.rename_fields(matrix1, {field1_name: field2_name})
    new_matrix = rfn.join_by(
        field2_name, renamed_matrix1, matrix2, jointype=join_type,
        r1postfix=matrix1_postfix,
        r2postfix=matrix2_postfix, asrecarray=True, usemask=use_mask
    )
    if progress_message:
        cfg.progress.update(end=True)
    cfg.logger.log.info('end')
    return new_matrix


# join tables
def join_tables(
        table1, table2, field1_name, field2_name, postfix='2',
        nodata_value=None, join_type='left', n_processes: int = None,
        progress_message=True, min_progress=None, max_progress=None
):
    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(
            process='join_tables', message='starting', start=True
        )
    if n_processes is None:
        n_processes = cfg.n_processes
    join_type = join_type.lower()
    if nodata_value is None:
        nodata_value = -9999
    cfg.multiprocess.join_tables_multiprocess(
        table1=table1, table2=table2, field1_name=field1_name,
        field2_name=field2_name, nodata_value=nodata_value,
        join_type=join_type, postfix=postfix, n_processes=n_processes,
        min_progress=min_progress, max_progress=max_progress
    )
    if progress_message:
        cfg.progress.update(end=True)
    cfg.logger.log.info('end')
    return cfg.multiprocess.output


# calculate pivot_60
def pivot_matrix(
        matrix, row_field, column_function_list, secondary_row_field_list=None,
        filter_string=None, nodata_value=-999, cross_matrix=None,
        field_names=False, progress_message=True
):
    """
    :param matrix: matrix array
    :param row_field: field out_column_name of values used as rows
    :param column_function_list: list of lists of column out_column_name and
        function on values and optional out dtype
    :param secondary_row_field_list: optional list of fields to be used as
        secondary rows
    :param filter_string: optional string for filtering input matrix values
    :param nodata_value: optional value for nodata
    :param cross_matrix: optional, if True the output table field names are
        only combination values
    :param field_names: optional, if True returns field names without
        performing the pivot matrix
    :param progress_message: optional, if True display process message

    """

    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(
            process='pivot_matrix', message='starting', start=True
        )
    # matrix filter
    if filter_string is None:
        matrix_1 = matrix
    else:
        matrix_1 = eval('matrix[%s]' % filter_string)
    # rows
    row_values = np.unique(matrix_1[row_field])
    row_value_list: list = row_values.tolist()
    cfg.logger.log.debug(
        'column_function_list: %s; len(row_value_list): %s' % (
            str(column_function_list), len(row_value_list))
    )
    # primary rows
    if secondary_row_field_list is None:
        pivot = np.zeros([len(column_function_list) + 1, len(row_value_list)])
        secondary_row_list = None
    # secondary rows
    else:
        unique_value_list = []
        for field in secondary_row_field_list:
            unique_value_list.append(np.unique(matrix_1[field]).tolist())
        if len(secondary_row_field_list) == 1:
            secondary_row_list = unique_value_list[0]
        else:
            assert itertools
            secondary_row_list = eval(
                'list(itertools.product%s)' % str((tuple(unique_value_list)))
            )
        pivot = np.zeros(
            [len(column_function_list) * len(secondary_row_list) + 1,
             len(row_value_list)]
        )
        cfg.logger.log.debug(
            'secondary_row_list: %s' % str(secondary_row_list)
        )
    output_column_list = [(row_field, matrix_1[row_field].dtype)]
    # filters for secondary rows
    secondary_row_filter_list = []
    # primary rows
    if secondary_row_field_list is None:
        for column_function in column_function_list:
            out_column_name = '%s_%s' % (
                column_function[0], column_function[1])
            # output columns
            output_column_list.append(
                (out_column_name, matrix_1[column_function[0]].dtype)
            )
    # secondary rows
    else:
        for column_function in column_function_list:
            for combination in secondary_row_list:
                out_column_name = '%s_%s' % (
                    column_function[0], column_function[1].replace('.', ''))
                function_string = ['[ ']
                for n in range(len(secondary_row_field_list)):
                    try:
                        try:
                            float(combination[n])
                            comb_string = str(combination[n])
                        except Exception as err:
                            str(err)
                            comb_string = '"%s"' % combination[n]
                        if cross_matrix:
                            out_column_name = str(combination[n])
                        else:
                            out_column_name = '%s_%s%s' % (
                                out_column_name, secondary_row_field_list[n],
                                combination[n])
                        function_string.append(
                            '(r["%s"] == %s)' % (
                                secondary_row_field_list[n], comb_string)
                        )
                        function_string.append(' &')
                    except Exception as err:
                        str(err)
                        try:
                            float(combination)
                            comb_string = str(combination)
                        except Exception as err:
                            str(err)
                            comb_string = '"%s"' % combination
                        if cross_matrix:
                            out_column_name = str(combination)
                        else:
                            out_column_name = '%s_%s%s' % (
                                out_column_name, secondary_row_field_list[n],
                                combination)
                        function_string.append(
                            '(r["%s"] == %s)' % (
                                secondary_row_field_list[n], comb_string)
                        )
                        function_string.append(' &')
                if function_string[-1] == ' &':
                    function_string.pop(-1)
                function_string.append(']')
                secondary_row_filter_list.append(''.join(function_string))
                # output columns
                try:
                    output_column_list.append(
                        (out_column_name, column_function[2])
                    )
                except Exception as err:
                    str(err)
                    output_column_list.append(
                        (out_column_name, matrix_1[column_function[0]].dtype)
                    )
    cfg.logger.log.debug(
        'secondary_row_filter_list: %s; output_column_list: %s'
        % (str(secondary_row_filter_list), str(output_column_list))
    )
    if field_names:
        return output_column_list
    pivot = np.rec.fromarrays(pivot, dtype=output_column_list)
    pivot[row_field] = row_values
    # populate table
    row_value_count = len(row_value_list)
    for v in range(row_value_count):
        if progress_message:
            cfg.progress.update(
                message='processing data',
                step=int(100 * (v + 1) / row_value_count),
                percentage=int(100 * (v + 1) / row_value_count)
            )
        else:
            cfg.progress.update(
                message='processing data',
                percentage=int(100 * (v + 1) / row_value_count)
            )
        r = matrix_1[matrix_1[row_field] == row_value_list[v]]
        assert r.shape
        d = 0
        for column in range(len(column_function_list)):
            operator = replace_numpy_operators(column_function_list[column][1])
            try:
                datatype = column_function_list[column][2]
            except Exception as err:
                str(err)
                datatype = matrix_1[column_function_list[column][0]].dtype
            if secondary_row_field_list is None:
                try:
                    s = eval(
                        '%s(r["%s"].astype("%s"))' % (
                            operator, column_function_list[column][0],
                            datatype)
                    )
                except Exception as err:
                    str(err)
                    s = nodata_value
                pivot[output_column_list[column + 1][0]][v] = s
            else:
                for _ in secondary_row_list:
                    try:
                        s = eval(
                            '%s((r["%s"]%s).astype("%s"))'
                            % (operator, column_function_list[column][0],
                               secondary_row_filter_list[d], datatype)
                        )
                    except Exception as err:
                        str(err)
                        s = nodata_value
                    pivot[output_column_list[d + 1][0]][v] = s
                    d += 1
    if progress_message:
        cfg.progress.update(end=True)
    cfg.logger.log.info('end')
    return pivot


# get values from matrix
def get_values(
        matrix, value_field, conditional_string=None, progress_message=True
):
    """
    :param matrix: input matrix
    :param value_field: string of value field name
    :param conditional_string: optional string used for condition where
        field must be referred to using 'field.' such as field.field_name
        or 'matrix.' such as matrix.field_name
    :param progress_message: optional, if True display process message
    """
    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(
            process='get_values', message='starting', start=True
        )
    # alias for fields
    field = matrix
    matrix_values = field[value_field]
    if conditional_string is None:
        values = matrix_values
    else:
        try:
            a = eval(conditional_string)
            values = matrix_values[a]
        except Exception as err:
            cfg.logger.log.error(str(err))
            return
    if progress_message:
        cfg.progress.update(end=True)
    cfg.logger.log.info('end')
    return values


# replace variables in expression for calculation
def replace_variables(matrix, expression_string):
    for field in columns(matrix):
        expression_string = expression_string.replace(
            '%s%s%s' % (
                cfg.variable_band_quotes, field, cfg.variable_band_quotes),
            'field[%s%s%s]' % (
                cfg.variable_band_quotes, field, cfg.variable_band_quotes)
        )
    cfg.logger.log.debug('expression_string: %s' % expression_string)
    return expression_string


# calculate single expression
def calculate(
        matrix, expression_string, output_field_name, progress_message=True
):
    """
    :param matrix: input matrix
    :param expression_string: string used for calculation where fields must
        be named using double quotes
        e.g."field_name" or using 'field.' such as field.field_name or
        'matrix.' such as matrix.field_name
    :param output_field_name: string of output field name
    :param progress_message: optional, if True display process message
    """
    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(
            process='calculate', message='starting', start=True
        )
    # alias for field in expression
    field = matrix
    expression_string = replace_variables(matrix, expression_string)
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
    matrix1 = None
    try:
        a = eval(expression_string)
        try:
            # existing field
            assert field[output_field_name]
            matrix1 = np.rec.array(np.copy(matrix))
            matrix1[output_field_name] = a
        except Exception as err:
            str(err)
            # new field
            matrix1 = append_field(matrix, output_field_name, a, a.dtype)
    except Exception as err:
        cfg.logger.log.error(str(err))
    if progress_message:
        cfg.progress.update(end=True)
    cfg.logger.log.info('end')
    return matrix1


# calculate multiple expressions
def calculate_multi(
        matrix, expression_string_list, output_field_name_list,
        progress_message=True
):
    """
    :param matrix: input matrix
    :param expression_string_list: list of strings used for calculation
        where fields must be named using double quotes
        e.g."field_name" or using 'field.' such as field.field_name or
        'matrix.' such as matrix.field_name
    :param output_field_name_list: list of strings of output field names
    :param progress_message: optional, if True display process message
    """
    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(
            process='calculate_multi', message='starting', start=True
        )
    # alias for field in expression
    field = matrix
    matrix1 = np.rec.array(np.copy(field))
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
    expression_count = len(expression_string_list)
    for e in range(expression_count):
        if progress_message:
            cfg.progress.update(
                message='processing data',
                step=int(100 * (e + 1) / expression_count),
                percentage=int(100 * (e + 1) / expression_count)
            )
        else:
            cfg.progress.update(
                message='processing data', percentage=int(
                    100 * (e + 1) / expression_count
                )
            )
        expression = replace_variables(matrix, expression_string_list[e])
        try:
            a = eval(expression)
            try:
                matrix1[output_field_name_list[e]] = a
            except Exception as err:
                str(err)
                matrix1 = append_field(
                    matrix1, output_field_name_list[e], a, a.dtype
                )
        except Exception as err:
            cfg.logger.log.error(str(err))
    if progress_message:
        cfg.progress.update(end=True)
    cfg.logger.log.info('end')
    return matrix1


# create new matrix selecting columns by name and optionally rename output
# columns
def redefine_matrix_columns(
        matrix, input_field_names, output_field_names=None,
        progress_message=True
):
    """
    :param matrix: matrix array
    :param input_field_names: list of field names to be included in output
    :param output_field_names: optional list of output field names with the
        same length as input_field_names
    :param progress_message: optional, if True display process message

    """

    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(
            process='redefine_matrix_columns', message='starting', start=True
        )
    field_list = []
    c = 0
    for field in input_field_names:
        if field in matrix.dtype.names:
            data_type = matrix[field].dtype
            field_list.append((field, data_type))
            c += 1
        else:
            cfg.logger.log.error('field %s not found' % field)
    matrix_f = define_fields(matrix, field_list)
    if output_field_names is not None:
        for f in range(len(output_field_names)):
            try:
                matrix_f = rename_field(
                    matrix_f, input_field_names[f], output_field_names[f]
                )
            except Exception as err:
                cfg.logger.log.error(str(err))
    if progress_message:
        cfg.progress.update(end=True)
    return matrix_f


# export matrix to csv, with field name list, and field decimals as integer
# or list
def matrix_to_csv(
        matrix, output_path, fields=None, field_decimals: Union[int, list] = 2,
        separator=None, decimal_separator=None, nodata_value=None,
        nodata_value_output='nan', progress_message=True
):
    """
    :param matrix: input matrix
    :param output_path: output path
    :param fields: optional list of fields to export
    :param field_decimals: integer number of decimals for decimal fields or
        list of integer values for decimal fields
    :param separator: separator of fields
    :param decimal_separator: optional decimal separator for float values
        replacing . character
    :param nodata_value: optional nodata value to be replaced
    :param nodata_value_output: optional string to replace nodata value in
        output
    :param progress_message:

    """

    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(
            process='matrix_to_csv', message='starting', start=True
        )
    if fields is None:
        fields = matrix.dtype.names
    if separator is None:
        separator = cfg.tab_delimiter
    dtypes = []
    for name in matrix.dtype.names:
        data_type = matrix[name].dtype
        if 'int' in str(data_type).lower() or '<i' in str(data_type).lower():
            data_type = np.dtype('int64')
        dtypes.append((name, str(data_type)))
    cfg.logger.log.debug('dtypes: %s' % str(dtypes))
    matrix_c = np.rec.array(np.copy(matrix))
    cfg.logger.log.debug('matrix_c.shape: %s' % str(matrix_c.shape))
    matrix_c = define_fields(matrix_c, dtypes)
    # list of field formats
    format_list = []
    # list of fields
    field_list = []
    c = 0
    header = []
    # iterate fields
    for field in fields:
        cfg.logger.log.debug('field: %s' % str(field))
        # header
        header.append(field)
        header.append(separator)
        if field in matrix.dtype.names:
            data_type = matrix[field].dtype
            field_list.append((field, data_type))
            # integer fields
            if ('int' in str(data_type).lower()
                    or '<i' in str(data_type).lower()):
                matrix_c[field][
                    matrix[field] == nodata_value] = cfg.nodata_val_Int64
                format_list.append('%s')
            # float fields
            elif (str(data_type)[0].lower() == 'f'
                  or '<f' in str(data_type)[0].lower()
                  or ('>f' in str(data_type)[0].lower())):
                matrix_c[field][
                    matrix[field] == nodata_value] = cfg.nodata_val_Int64
                try:
                    format_list.append('%1.{}f'.format(field_decimals[c]))
                except Exception as err:
                    str(err)
                    try:
                        format_list.append('%1.{}f'.format(field_decimals))
                    except Exception as err:
                        str(err)
                        format_list.append('%1.2f')
            # string fields
            else:
                format_list.append('%s')
                matrix_c[field][matrix[field] == str(
                    nodata_value)] = str(cfg.nodata_val_Int64)
        else:
            cfg.logger.log.error('field %s not found' % field)
        c += 1
    header.pop(-1)
    header.append(cfg.new_line)
    joined_header = ''.join(header)
    matrix_format = define_fields(matrix_c, field_list)
    # export numpy matrix to text
    stream = io.StringIO()
    np.savetxt(stream, matrix_format, delimiter=separator, fmt=format_list)
    csv = '{}{}'.format(joined_header, stream.getvalue())
    if decimal_separator is not None:
        csv = csv.replace('.', decimal_separator)
    # replace nodata values
    if nodata_value_output is not None:
        if decimal_separator is None:
            s = '.'
        else:
            s = decimal_separator
        if type(field_decimals) is list:
            for fd in field_decimals[c]:
                csv = csv.replace(
                    '%s%s%s' % (cfg.nodata_val_Int64, s, '0' * fd),
                    str(nodata_value_output)
                )
        else:
            csv = csv.replace(
                '%s%s%s' % (cfg.nodata_val_Int64, s, '0' * field_decimals),
                str(nodata_value_output)
            )
        csv = csv.replace(str(cfg.nodata_val_Int64), str(nodata_value_output))
    files_directories.create_parent_directory(output_path)
    with open(output_path, 'w') as file:
        file.write(csv)
    if progress_message:
        cfg.progress.update(end=True)
    cfg.logger.log.info('end')


"""Alias for matrix_to_csv."""
export_table = matrix_to_csv


# replace operators
def replace_numpy_operators(expression):
    if expression == 'sum':
        f = 'np.nansum'
    elif expression == 'min':
        f = 'np.nanmin'
    elif expression == 'max':
        f = 'np.nanmax'
    elif expression == 'percentile':
        f = 'np.nanpercentile'
    elif expression == 'median':
        f = 'np.nanmedian'
    elif expression == 'mean':
        f = 'np.nanmean'
    elif expression == 'std':
        f = 'np.nanstd'
    elif expression == 'log':
        f = 'np.log'
    elif expression == 'log10':
        f = 'np.log10'
    elif expression == 'sqrt':
        f = 'np.sqrt'
    elif expression == 'cos':
        f = 'np.cos'
    elif expression == 'sin':
        f = 'np.sin'
    elif expression == 'tan':
        f = 'np.tan'
    elif expression == 'exp':
        f = 'np.exp'
    elif expression == 'where':
        f = 'np.where'
    else:
        f = expression
    return f


# list column names
def columns(matrix):
    cfg.logger.log.debug('matrix.dtype.names: %s' % str(matrix.dtype.names))
    return matrix.dtype.names


# rename field matrices
def rename_field(matrix, old_field, new_field):
    new_matrix = rfn.rename_fields(matrix, {old_field: new_field})
    cfg.logger.log.debug(
        'old_field: %s; new_field: %s' % (old_field, new_field)
    )
    return new_matrix


# append field matrix
def append_field(matrix, field_name, data, data_type):
    new_matrix = rfn.append_fields(
        matrix, field_name, data, data_type, usemask=False, asrecarray=True
    )
    cfg.logger.log.debug(
        'field_name: %s; data_type: %s' % (field_name, data_type)
    )
    return new_matrix


# define fields matrix by list of tuples (field, dataType)
def define_fields(matrix, field_list):
    matrix_with_fields = rfn.require_fields(matrix, field_list)
    new_matrix = np.rec.array(matrix_with_fields)
    cfg.logger.log.debug('field_list: %s' % field_list)
    return new_matrix


# append values to table
def append_values_to_table(matrix, value_list):
    array_1 = np.array(value_list)
    rec_array = np.rec.fromarrays(array_1, dtype=matrix.dtype)
    a = append_tables(matrix, rec_array)
    cfg.logger.log.debug('a.shape: %s' % a.shape)
    return a


# append two tables
def append_tables(matrix1, matrix2):
    a = rfn.stack_arrays((matrix1, matrix2), asrecarray=True, usemask=False)
    cfg.logger.log.debug('a.shape: %s' % a.shape)
    return a


# sort table
def sort_table_by_field(matrix, field_name):
    a = np.rec.array(np.copy(matrix))
    a.sort(order=field_name)
    cfg.logger.log.debug('a.shape: %s' % a.shape)
    return a


# add product to preprocess table
def add_product_to_preprocess(
        product_list, spacecraft_list, processing_level, band_name_list,
        product_path_list,
        scale_list, offset_list, nodata_list, date_list, k1_list, k2_list,
        band_number_list, e_sun_list, sun_elevation_list,
        earth_sun_distance_list
):
    dtype_list = [
        ('product', 'U64'), ('spacecraft', 'U64'), ('processing_level', 'U64'),
        ('band_name', 'U128'), ('product_path', 'U1024'), ('scale', 'float64'),
        ('offset', 'float64'), ('nodata', 'float64'),
        ('date', 'datetime64[D]'), ('k1', 'float64'), ('k2', 'float64'),
        ('band_number', 'U64'), ('e_sun', 'float64'),
        ('sun_elevation', 'float64'), ('earth_sun_distance', 'float64')
    ]
    rec_array = np.rec.fromarrays(
        np.array(
            [product_list, spacecraft_list, processing_level, band_name_list,
             product_path_list, scale_list,
             offset_list, nodata_list, date_list, k1_list, k2_list,
             band_number_list, e_sun_list, sun_elevation_list,
             earth_sun_distance_list]
        ), dtype=dtype_list
    )
    cfg.logger.log.debug('rec_array.shape: %s' % rec_array.shape)
    return rec_array


# create product table
def create_product_table(
        product=None, product_id=None, acquisition_date=None, cloud_cover=None,
        zone_path=None,
        row=None, min_lat=None, min_lon=None, max_lat=None, max_lon=None,
        collection=None, size=None,
        preview=None, uid=None, image=None
):
    dtype_list = [
        ('product', 'U512'), ('image', 'U1024'), ('product_id', 'U512'),
        ('acquisition_date', 'datetime64[D]'), ('cloud_cover', 'int8'),
        ('zone_path', 'U8'), ('row', 'U8'), ('min_lat', 'float64'),
        ('min_lon', 'float64'), ('max_lat', 'float64'), ('max_lon', 'float64'),
        ('collection', 'U1024'), ('size', 'U512'), ('preview', 'U1024'),
        ('uid', 'U1024')
    ]
    rec_array = np.rec.fromrecords(
        [(
            product, image, product_id, acquisition_date, cloud_cover,
            zone_path, row, min_lat, min_lon, max_lat, max_lon,
            collection, size, preview, uid)], dtype=dtype_list
    )
    cfg.logger.log.debug('rec_array.shape: %s' % rec_array.shape)
    return rec_array


# create table of products
def stack_product_table(product_list):
    if product_list is None or len(product_list) == 0:
        return None
    table = rfn.stack_arrays(product_list, asrecarray=True, usemask=False)
    table.sort(order='product')
    cfg.logger.log.debug('table.shape: %s' % table.shape)
    return table


# create bandset table
def create_bandset_table(band_list):
    if band_list is None:
        return None
    table = rfn.stack_arrays(band_list, asrecarray=True, usemask=False)
    table.sort(order='band_number')
    cfg.logger.log.debug('table.shape: %s' % table.shape)
    return table


# add spectral signature to catalog table
def add_spectral_signature_to_catalog_table(
        signature_id=None, macroclass_id=0, class_id=0, class_name=None,
        previous_catalog=None, selected=1, min_dist_thr=0, max_like_thr=0,
        spec_angle_thr=0
):
    dtype_list = [('signature_id', 'U64'), ('macroclass_id', 'int16'),
                  ('class_id', 'int16'), ('class_name', 'U512'),
                  ('selected', 'byte'), ('min_dist_thr', 'float64'),
                  ('max_like_thr', 'float64'), ('spec_angle_thr', 'float64')]
    rec_array = np.rec.fromrecords(
        [(signature_id, int(macroclass_id), int(class_id), str(class_name),
          selected,
          min_dist_thr, max_like_thr, spec_angle_thr)],
        dtype=dtype_list
    )
    # add to previous bandset catalog table
    if previous_catalog is not None:
        # replace signature_id
        catalog = previous_catalog[
            previous_catalog['signature_id'] != signature_id]
        table = rfn.stack_arrays(
            [catalog, rec_array], asrecarray=True, usemask=False
        )
    else:
        table = rec_array
    return table


# create spectral signature table
def create_spectral_signature_table(
        value_list, wavelength_list, standard_deviation_list=None
):
    dtype_list = [('value', 'float64'), ('wavelength', 'float64'),
                  ('standard_deviation', 'float64')]
    if standard_deviation_list is None:
        standard_deviation_list = [0] * len(value_list)
    data_list = [value_list, wavelength_list, standard_deviation_list]
    rec_array = np.rec.fromarrays(np.array(data_list), dtype=dtype_list)
    return rec_array


# create band table
def create_band_table(
        band_number=0, raster_band=None, path=None,
        name=None, wavelength=0, wavelength_unit=None, additive_factor=0,
        multiplicative_factor=1, date='1900-01-01', x_size=0,
        y_size=0, top=0, left=0, bottom=0, right=0, x_count=0, y_count=0,
        nodata=None, data_type=None, crs=None, root_directory=None,
        number_of_bands=0, x_block_size=0, y_block_size=0, scale=None,
        offset=None
):
    dtype_list = [
        ('band_number', 'int16'), ('raster_band', 'int16'),
        ('path', 'U1024'), ('root_directory', 'U1024'), ('name', 'U256'),
        ('wavelength', 'float64'), ('wavelength_unit', 'U32'),
        ('additive_factor', 'float32'), ('multiplicative_factor', 'float32'),
        ('date', 'datetime64[D]'),
        ('x_size', 'float32'), ('y_size', 'float32'),
        ('top', 'float32'), ('left', 'float32'),
        ('bottom', 'float32'), ('right', 'float32'),
        ('x_count', 'int64'), ('y_count', 'int64'),
        ('nodata', 'int64'), ('data_type', 'U16'),
        ('number_of_bands', 'int16'),
        ('x_block_size', 'int64'), ('y_block_size', 'int64'),
        ('scale', 'int64'), ('offset', 'int64'), ('crs', 'U1024')
    ]
    if nodata is None:
        nodata = cfg.nodata_val_Int64
    else:
        try:
            _array = np.array([nodata], dtype=int)
        except Exception as err:
            cfg.logger.log.error('%s; nodata: %s' % (err, nodata))
            nodata = cfg.nodata_val_Int64
    if raster_band is None:
        raster_band = 1
    if scale is None:
        scale = 1
    if offset is None:
        offset = 0
    try:
        rec_array = np.rec.fromrecords(
            [(band_number, raster_band, path, root_directory, name,
              wavelength, wavelength_unit, additive_factor,
              multiplicative_factor, date, x_size, y_size, top, left, bottom,
              right, x_count, y_count, nodata, data_type,
              number_of_bands, x_block_size, y_block_size, scale, offset,
              crs)],
            dtype=dtype_list
        )
        cfg.logger.log.debug('rec_array.shape: %s' % rec_array.shape)
    except Exception as err:
        if 'empty' not in str(err):
            cfg.logger.log.error(str(err))
        # create empty table
        rec_array = np.rec.fromrecords(np.zeros((0,)), dtype=dtype_list)
    return rec_array


# create bandset catalog table
def create_bandset_catalog_table(
        bandset_number=0, root_directory=None, date='NaT', bandset_uid=0,
        bandset_name=None, previous_catalog=None, crs=None,
        box_coordinate_left=None, box_coordinate_top=None,
        box_coordinate_right=None, box_coordinate_bottom=None
):
    dtype_list = [
        ('bandset_number', 'int64'), ('bandset_name', 'U512'),
        ('date', 'datetime64[D]'), ('root_directory', 'U1024'),
        ('crs', 'U1024'), ('box_coordinate_left', 'float64'),
        ('box_coordinate_top', 'float64'), ('box_coordinate_right', 'float64'),
        ('box_coordinate_bottom', 'float64'), ('uid', 'U64')
    ]
    rec_array = np.rec.fromrecords(
        [(bandset_number, bandset_name, date, root_directory, crs,
          box_coordinate_left, box_coordinate_top, box_coordinate_right,
          box_coordinate_bottom, bandset_uid)], dtype=dtype_list
    )
    # add to previous bandset catalog table
    if previous_catalog is not None:
        table = rfn.stack_arrays(
            [previous_catalog, rec_array], asrecarray=True, usemask=False
        )
        table.sort(order='bandset_number')
    else:
        table = rec_array
    return table


# find nearest value
def find_nearest_value(array, field_name, value, threshold=None):
    difference = np.abs(array[field_name] - value)
    if threshold is None:
        index = np.argmin(difference)
        output = array[index]
    else:
        if np.nanmin(difference) < threshold:
            index = np.argmin(difference)
            output = array[index]
        else:
            output = None
    return output
