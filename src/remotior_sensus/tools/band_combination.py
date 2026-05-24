# Remotior Sensus , software to process remote sensing and GIS data.
# Copyright (C) 2022-2026 Luca Congedo.
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
"""Band combination.

This tool is intended for combining classifications in order to get a
raster where each value corresponds to a combination of class values.
A unique value is assigned to each combination of values.
The output is a raster made of unique values corresponding to combinations
of values.
An output text file describes the correspondence between unique values
and combinations, as well as the statistics of each combination.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> combination = rs.band_combination(input_bands=['path_1', 'path_2'],
    ... output_path='output_path')
"""

import io
from typing import Union, Optional

import numpy as np

from remotior_sensus.core import configurations as cfg, table_manager as tm
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.core.processor_functions import (
    cross_rasters, raster_unique_values
)
from remotior_sensus.util import (
    shared_tools, files_directories, raster_vector, read_write_files
)


def band_combination(
        input_bands: Union[list, int, BandSet],
        output_path: Optional[str] = None, nodata_value: Optional[int] = None,
        no_raster_output: Optional[bool] = False,
        overwrite: Optional[bool] = False,
        n_processes: Optional[int] = None, available_ram: Optional[int] = None,
        bandset_catalog: Optional[BandSetCatalog] = None,
        extent_list: Optional[list] = None,
        column_name_list: Optional[list] = None,
        output_table: Optional[bool] = True,
        separator: Optional[str] = None,
        progress_message: Optional[bool] = True
) -> OutputManager:
    """Calculation of band combination.

    This tool allows for the combination of rasters or bands loaded in a
    BandSet.
    This tool is intended for combining classifications in order to get a
    raster where each value corresponds to a combination of class values.
    Input raster values must be integer type.
    The output is a combination raster and a text file reporting the
    statistics of each combination.

    Args:
        input_bands: list of paths of input rasters, or number of BandSet, or 
            BandSet object.
        output_path: path of the output raster.
        no_raster_output: if True, no output raster is written to file.
        overwrite: if True, output overwrites existing files.
        nodata_value: input value to be considered as nodata.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        bandset_catalog: BandSetCatalog object required if input_bands is a 
            BandSet number.
        extent_list: list of boundary coordinates left top right bottom.
        column_name_list: list of strings corresponding to input bands used 
            as column names in output table, if None then column names are 
            extracted for input band names.
        output_table: if True then calculate output table; 
            if False then calculate only array of combinations and sum.
        separator: separator of fields of output table (default tab)
        progress_message: if True then start progress message, if False does 
            not start the progress message (useful if launched from other tools).

    Returns:
        If output_table is True returns the :func:`~remotior_sensus.core.output_manager.OutputManager` object with
            - paths = [output raster path, output table path]

        If output_table is False returns the :func:`~remotior_sensus.core.output_manager.OutputManager` object with
            - paths = [virtual raster path]
            - extra = {'combinations': array of combinations, 'sums': array of the sums of values}

    Examples:
        Combination using two rasters having paths path_1 and path_2
            >>> combination = band_combination(input_bands=['path_1', 'path_2'],output_path='output_path')

        The combination raster and the table are finally created; the paths can be retrieved from the output that is an :func:`~remotior_sensus.core.output_manager.OutputManager` object
            >>> raster_path, table_path = combination.paths
            >>> print(raster_path)
            output_path

        Combination using a virtual raster as output file
            >>> combination = band_combination(input_bands=['path_1', 'path_2'],output_path='output_path.vrt')
            >>> raster_path, table_path = combination.paths
            >>> print(raster_path)
            output_path.vrt

        Using input BandSet number
            >>> catalog = BandSetCatalog()
            >>> combination = band_combination(input_bands=1,output_path='output_path',bandset_catalog=catalog)

        Using input BandSet
            >>> catalog = BandSetCatalog()
            >>> combination = band_combination(input_bands=catalog.get_bandset(1),output_path='output_path')
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '),
        message='starting', start=progress_message
    )
    separator = separator or cfg.tab_delimiter
    # prepare process files
    prepared = shared_tools.prepare_process_files(
        input_bands=input_bands, output_path=output_path, overwrite=overwrite,
        n_processes=n_processes, bandset_catalog=bandset_catalog,
        box_coordinate_list=extent_list
    )
    input_raster_list = prepared['input_raster_list']
    name_list = prepared['name_list']
    out_path = prepared['output_path']
    vrt_r = prepared['virtual_output']
    vrt_path = prepared['temporary_virtual_raster']
    n_processes = prepared['n_processes']
    raster_info = prepared['raster_info']
    d_types_list = []
    numpy_type_list = []
    for i, info in enumerate(raster_info):
        if cfg.int64_dt in info[8]:
            numpy_data_type = np.int64
        elif cfg.uint64_dt in info[8]:
            numpy_data_type = np.uint64
        elif cfg.float64_dt in info[8]:
            numpy_data_type = np.float64
        elif cfg.int32_dt in info[8]:
            numpy_data_type = np.int32
        elif cfg.uint32_dt in info[8]:
            numpy_data_type = np.uint32
        elif cfg.float32_dt in info[8]:
            numpy_data_type = np.float32
        elif cfg.int16_dt in info[8]:
            numpy_data_type = np.int16
        elif cfg.uint16_dt in info[8]:
            numpy_data_type = np.uint16
        elif cfg.byte_dt in info[8]:
            numpy_data_type = np.int8
        else:
            numpy_data_type = np.uint64
        d_types_list.append(numpy_data_type)
        numpy_type_list.append((f'f{i}', numpy_data_type))
    same_type = all(d == d_types_list[0] for d in d_types_list)
    # calculation data type for unique values
    if same_type:
        calculation_datatype = d_types_list[0]
    else:
        if (cfg.int64_dt in d_types_list or cfg.uint64_dt in d_types_list
                or cfg.float64_dt in d_types_list
                or cfg.int32_dt in d_types_list
                or cfg.uint32_dt in d_types_list
                or cfg.float32_dt in d_types_list):
            calculation_datatype = np.int64
        else:
            calculation_datatype = np.int32
    # dummy bands for memory calculation as the number of input raster
    dummy_bands = round(len(input_raster_list) * 0.5)
    cfg.multiprocess.run(
        raster_path=vrt_path, function=raster_unique_values,
        keep_output_argument=True, n_processes=n_processes,
        calculation_datatype=calculation_datatype,
        available_ram=available_ram, dummy_bands=dummy_bands,
        progress_message='unique values', min_progress=2, max_progress=50
    )
    cfg.multiprocess.multiprocess_unique_values()
    if cfg.multiprocess.output is False:
        cfg.logger.log.error('unable to calculate')
        cfg.messages.error('unable to calculate')
        cfg.progress.update(failed=True)
        # synch bcast
        _output = shared_tools.mpi_bcast(None)
        return OutputManager(check=False)
    cmb = cfg.multiprocess.output

    cfg.logger.log.debug('len(cmb): %s; cmb[0]: %s' % (len(cmb), str(cmb[0])))
    # random variable list
    rnd_var_list = None
    # array of combinations
    cmb = np.asarray(cmb).T
    cmb_arr = np.rec.fromarrays(cmb, dtype=numpy_type_list)
    cmb_arr_size = cmb_arr.size
    # expression builder
    max_dig = max(len(str(cmb_arr_size)) - 2, 1)
    rec_combinations_array = reclassification_list = None
    # maximum value during calculation
    calc_max_32 = 2 ** 32 // 2 - 1
    calc_max_64 = 2 ** 64 // 2 - 1
    t = 0
    maximum_value_type = np.int64
    max_v = np.nanmax(cmb, axis=0)
    min_v = np.nanmin(cmb, axis=0)
    while t < max_dig * 100:
        if cfg.action:
            t += 1
            rnd_var_list = []
            expression_comb = []
            maximum_value = np.array(0)
            max_digit = max(max_dig, t // 10 + 1)
            calc_max = calc_max_32
            # first deterministic variable list
            first_var_list = [1] * len(input_raster_list)
            for y in range(len(input_raster_list)):
                if t == 1:
                    if y + 1 < len(input_raster_list):
                        first_var_list[y + 1] = (
                                (first_var_list[y] * (max_v[y + 1] + 1))
                                % calc_max
                        )
                    if min_v[y] < 0:
                        add_c = -1 * min_v[y] + 1
                    else:
                        add_c = 0
                    # expression combination
                    expression_comb.append(
                        f'("f{y}".astype("datatype") + {add_c}) '
                        f'* {first_var_list[y]}'
                    )
                    expression_comb.append(' + ')
                    maximum_value += np.array((first_var_list[y] + add_c)
                                              * max_v[y])
                    rnd_var_list.append([first_var_list[y], add_c])
                elif t == 2:
                    calc_max = calc_max_64
                    if y + 1 < len(input_raster_list):
                        first_var_list[y + 1] = (
                                (first_var_list[y] * (max_v[y + 1] + 1))
                                % calc_max
                        )
                    if min_v[y] < 0:
                        add_c = -1 * min_v[y] + 1
                    else:
                        add_c = 0
                    # expression combination
                    expression_comb.append(
                        f'("f{y}".astype("datatype") + {add_c}) '
                        f'* {first_var_list[y]}'
                    )
                    expression_comb.append(' + ')
                    maximum_value += np.array((first_var_list[y] + add_c)
                                              * max_v[y])
                    rnd_var_list.append([first_var_list[y], add_c])
                else:
                    if t > 10:
                        calc_max = calc_max_64
                    rnd_var = np.random.randint(10 ** max_digit)
                    if min_v[y] < 0:
                        add_c = -1 * min_v[y] + 1
                        if t > 10:
                            digit_len = np.random.randint(1, max_digit + 1)
                            add_c += np.random.randint(10 ** (digit_len - 1),
                                                       10 ** digit_len)
                    else:
                        add_c = 0
                        if t > 10:
                            digit_len = np.random.randint(1, max_digit + 1)
                            add_c = np.random.randint(10 ** (digit_len - 1),
                                                      10 ** digit_len)
                    # avoid too large numbers
                    while (rnd_var * (np.array(max_v[y]) + add_c)
                           > calc_max):
                        rnd_var = rnd_var // 2
                    rnd_var_list.append([rnd_var, add_c])
                    # expression combination
                    expression_comb.append(
                        f'("f{y}".astype("datatype") + {add_c}) * {rnd_var}'
                    )
                    expression_comb.append(' + ')
                    maximum_value += rnd_var * (np.array(max_v[y]) + add_c)
            expression_comb.pop(-1)
            if maximum_value < calc_max:
                # noinspection PyUnresolvedReferences
                maximum_value_type = maximum_value.dtype
                joined_expression_comb = ''.join(expression_comb).replace(
                    'datatype', str(maximum_value_type))
                rec_combinations_array = tm.calculate(
                    matrix=cmb_arr,
                    expression_string=joined_expression_comb,
                    output_field_name='id', progress_message=False,
                    calculation_type='int64'
                )
                # check if unique new values are as many as combinations
                uni = np.unique(rec_combinations_array.id)
                if uni.size == cmb_arr_size:
                    new_val = np.arange(1, rec_combinations_array.shape[0] + 1,
                                        dtype=np.int64)
                    rec_combinations_array = tm.sort_table_by_field(
                        rec_combinations_array, 'id'
                    )
                    rec_combinations_array = tm.append_field(
                        rec_combinations_array, 'new_val', new_val, 'int64'
                    )
                    reclassification_list = uni
                    break
        else:
            cfg.logger.log.error('cancel')
            cfg.progress.update(failed=True)
            # synch bcast
            _output = shared_tools.mpi_bcast(None)
            return OutputManager(check=False)
    if (maximum_value_type == np.int64 or maximum_value_type == np.uint64
            or maximum_value_type == np.float64):
        output_nodata = cfg.nodata_val_Int64
        output_data_type = cfg.int64_dt
        calc_data_type = np.int64
    else:
        output_nodata = cfg.nodata_val_Int32
        output_data_type = cfg.int32_dt
        calc_data_type = np.int32
    expression = []
    for r, val in enumerate(rnd_var_list):
        if cfg.action:
            expression.append(
                f'({cfg.array_function_placeholder}[::, ::, {r}] + {val[1]}) '
                f'* {val[0]}'
            )
            expression.append(' + ')
    expression.pop(-1)
    joined_expression = ''.join(expression)
    cfg.logger.log.debug('joined_expression: %s' % joined_expression)
    if no_raster_output:
        output_raster_path = False
    else:
        output_raster_path = out_path
    # dummy bands for memory calculation
    dummy_bands = 3
    (vrt_path, reclassification_list,
     joined_expression) = shared_tools.mpi_bcast([
        vrt_path, reclassification_list,  joined_expression])
    # combination calculation
    cfg.multiprocess.run(
        raster_path=vrt_path, function=cross_rasters,
        function_argument=reclassification_list,
        function_variable=joined_expression,
        dummy_bands=dummy_bands, calculation_datatype=calc_data_type,
        use_value_as_nodata=nodata_value,
        any_nodata_mask=True, output_raster_path=output_raster_path,
        output_data_type=output_data_type, output_nodata_value=output_nodata,
        compress=cfg.raster_compression, n_processes=n_processes,
        available_ram=available_ram, keep_output_argument=True,
        virtual_raster=vrt_r, progress_message='cross rasters',
        min_progress=50, max_progress=90
    )
    cfg.progress.update(message='output table', step=90)
    # calculate sum of values
    cfg.multiprocess.multiprocess_sum_array(nodata_value)
    if cfg.multiprocess.output is False:
        cfg.logger.log.error('unable to calculate')
        cfg.messages.error('unable to calculate')
        cfg.progress.update(failed=True)
        # synch bcast
        _output = shared_tools.mpi_bcast(None)
        return OutputManager(check=False)
    sum_val = cfg.multiprocess.output
    tbl_out = None
    if not output_table:
        r_out = OutputManager(
            paths=[vrt_path], extra={
                'combinations': rec_combinations_array, 'sums': sum_val
            }
        )
        # synch bcast
        _output = shared_tools.mpi_bcast(r_out)
        return r_out
    else:
        # get pixel unit from input raster
        if type(vrt_path) is list:
            raster_i = vrt_path[0]
        else:
            raster_i = vrt_path
        (gt, crs, un, xy_count, nd, number_of_bands, block_size,
         scale_offset, data_type) = raster_vector.raster_info(raster_i)
        p_x = gt[1]
        p_y = abs(gt[5])
        joined_table = tm.join_tables(
            table1=rec_combinations_array, table2=sum_val,
            field1_name='new_val', field2_name='new_val',
            nodata_value=cfg.nodata_val_UInt64, join_type='left',
            progress_message=False, min_progress=90, max_progress=100
        )
        joined_table = shared_tools.mpi_bcast(joined_table)
        # name of combination field in output table
        combination_name_list = []
        p = 0
        for r_name in name_list:
            if column_name_list is None:
                combination_name_list.append(r_name[0:16])
            else:
                try:
                    combination_name_list.append(column_name_list[p])
                except Exception as err:
                    str(err)
                    combination_name_list.append(r_name[0:16])
                p += 1
        # create table
        table = _combination_table(
            joined_table[joined_table['sum'] != cfg.nodata_val_UInt64],
            combination_name_list, un, p_x, p_y, separator
        )
        if (not cfg.mpi_comm) or (cfg.mpi_rank == 0):
            # save combination to table
            tbl_out = shared_tools.join_path(
                files_directories.parent_directory(out_path), '{}{}'.format(
                    files_directories.file_name(
                        out_path, suffix=False
                    ), cfg.csv_suffix
                )
            ).replace('\\', '/')
            read_write_files.write_file(table, tbl_out)
            cfg.progress.update(end=True)
            cfg.logger.log.info(
                'end; band combination: %s; table: %s'
                % (str(out_path), str(tbl_out))
            )
            r_out = OutputManager(
                paths=[out_path, tbl_out], extra={
                    'combinations': rec_combinations_array, 'sums': sum_val
                }
            )
        else:
            r_out = None
        r_out = shared_tools.mpi_bcast(r_out)
        return r_out


def _combination_table(
        table: np.ndarray, combination_names: list, crs_unit: str,
        pixel_size_x: int, pixel_size_y: int, separator: str
) -> str:
    """Creation of combination table.

    Creates text for table where each value corresponds to a combination
    of class values, and the area statistics of each combination,
    if the input bands are in cartographic coordinates.

    Args:
        table: table of band combination.
        combination_names: list of names of combinations.
        crs_unit: unit of crs used for area calculation.
        pixel_size_x: pixel size along x.
        pixel_size_y: pixel size along y.
        separator: separator for csv file.

    Returns:
        The text of the combination table.
    """
    separator = separator or cfg.comma_delimiter
    output_field_names = ['RasterValue', *combination_names, 'PixelSum']
    input_field_names = [
        'new_val', *(f'f{c}' for c in range(len(combination_names))), 'sum'
    ]
    # table
    if 'degree' not in crs_unit:
        output_field_names.append('Area [%s^2]' % crs_unit)
        input_field_names.append('area')
        cross_class = tm.calculate(
            matrix=table, expression_string='"sum" * %s * %s' % (
                str(pixel_size_x), str(pixel_size_y)),
            output_field_name='area', progress_message=False
        )
    else:
        cross_class = table
    redefined = tm.redefine_matrix_columns(
        matrix=cross_class, input_field_names=input_field_names,
        output_field_names=output_field_names, progress_message=False
    )
    if (not cfg.mpi_comm) or (cfg.mpi_rank == 0):
        # create stream handler
        stream1 = io.StringIO()
        np.savetxt(stream1, redefined, delimiter=separator, fmt='%1.2f')
        matrix_value = stream1.getvalue().replace('.00', '')
        header = separator.join(output_field_names)
        output = f'{header}{cfg.new_line}{matrix_value}{cfg.new_line}'
    else:
        output = None
    return output
