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
        overwrite: Optional[bool] = False,
        n_processes: Optional[int] = None, available_ram: Optional[int] = None,
        bandset_catalog: Optional[BandSetCatalog] = None,
        extent_list: Optional[list] = None,
        column_name_list: Optional[list] = None,
        output_table: Optional[bool] = True,
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
        input_bands: list of paths of input rasters, or number of BandSet, or BandSet object.
        output_path: path of the output raster.
        overwrite: if True, output overwrites existing files.
        nodata_value: input value to be considered as nodata.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        bandset_catalog: BandSetCatalog object required if input_bands is a BandSet number.
        extent_list: list of boundary coordinates left top right bottom.
        column_name_list: list of strings corresponding to input bands used 
            as column names in output table, if None then column names are extracted for input band names.
        output_table: if True then calculate output table; 
            if False then calculate only array of combinations and sum.
        progress_message: if True then start progress message; 
            if False does not start the progress message (useful if launched from other tools).

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
    if progress_message:
        cfg.logger.log.info('start')
        cfg.progress.update(
            process=__name__.split('.')[-1].replace('_', ' '),
            message='starting', start=True
        )
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
    # dummy bands for memory calculation as the number of input raster
    dummy_bands = len(input_raster_list) + 1
    cfg.multiprocess.run(
        raster_path=vrt_path, function=raster_unique_values,
        keep_output_argument=True, n_processes=n_processes,
        available_ram=available_ram, dummy_bands=dummy_bands,
        progress_message='unique values', min_progress=2, max_progress=50
    )
    cfg.multiprocess.multiprocess_unique_values()
    cmb = cfg.multiprocess.output
    cfg.logger.log.debug('len(cmb): %s; cmb[0]: %s' % (len(cmb), str(cmb[0])))
    # random variable list
    rnd_var_list = None
    # array of combinations
    dtype_list = []
    for c in range(len(input_raster_list)):
        dtype_list.append(('f%s' % str(c), 'int64'))
    cmb_arr = np.rec.fromarrays(np.asarray(cmb).T, dtype=dtype_list)
    cmb_arr_ravel = np.asarray(cmb).ravel()
    max_v = np.nanmax(cmb_arr_ravel)
    if np.sum(cmb_arr_ravel <= 0) < 1:
        # calculation data type
        calc_data_type = np.uint32
        output_data_type = cfg.uint32_dt
        calc_nodata = cfg.nodata_val_UInt32
        add_c = 1
    else:
        add_c = 1 - np.nanmin(cmb_arr_ravel)
        calc_data_type = np.int32
        output_data_type = cfg.int32_dt
        calc_nodata = cfg.nodata_val_Int32
    # expression builder
    max_dig = 10 - len(str(np.sum(max_v)))
    if max_dig < 0:
        max_dig = 2
    rec_combinations_array = None
    reclassification_list = None
    t = 0
    method = 1
    while t < 5000:
        if cfg.action is True:
            t += 1
            rnd_var_list = []
            expression_comb = []
            for y in range(len(input_raster_list)):
                if method == 1:
                    exp_r = int(np.random.random() * 10) + 1
                    if exp_r > max_dig:
                        exp_r = max_dig
                    const_v = int(10 ** exp_r)
                    method += 1
                elif method == 2:
                    const_v = int(10 ** max_dig)
                    method += 1
                else:
                    exp_r = int(np.random.random() * 10) + 1
                    if exp_r > 8:
                        exp_r = 8
                    const_v = int(10 ** exp_r)
                    method = 1
                if const_v < 1:
                    const_v = 3
                rnd_var = int(const_v * np.random.random())
                if rnd_var == 0:
                    rnd_var = 1
                # avoid too large numbers
                while np.sum(
                        rnd_var * (np.array(max_v, dtype=np.float32) + add_c)
                ) > calc_nodata:
                    rnd_var = int(rnd_var / 2)
                rnd_var_list.append(rnd_var)
                # expression combination
                expression_comb.append(
                    '("f%s" + %s) * %s' % (str(y), str(add_c), str(rnd_var))
                )
                expression_comb.append(' + ')
            expression_comb.pop(-1)
            joined_expression_comb = ''.join(expression_comb)
            rec_combinations_array = tm.calculate(
                matrix=cmb_arr, expression_string=joined_expression_comb,
                output_field_name='id', progress_message=False
            )
            new_val = list(range(1, rec_combinations_array.shape[0] + 1))
            rec_combinations_array = tm.sort_table_by_field(
                rec_combinations_array, 'id'
            )
            rec_combinations_array = tm.append_field(
                rec_combinations_array, 'new_val', new_val, 'int64'
            )
            # check if unique new values are the same number as combinations
            uni = np.unique(rec_combinations_array.id)
            if uni.shape == cmb_arr.shape:
                reclassification_list = sorted(list(uni))
                break
        else:
            cfg.logger.log.error('cancel')
            return OutputManager(check=False)
    expression = []
    for r in range(len(rnd_var_list)):
        expression.append(
            '(%s[::, ::, %s] + %s) * %s' % (
                cfg.array_function_placeholder, str(r), str(add_c),
                str(rnd_var_list[r]))
        )
        expression.append(' + ')
    expression.pop(-1)
    joined_expression = ''.join(expression)
    cfg.logger.log.debug('joined_expression: %s' % joined_expression)
    # dummy bands for memory calculation
    dummy_bands = 3
    # combination calculation
    cfg.multiprocess.run(
        raster_path=vrt_path, function=cross_rasters,
        function_argument=reclassification_list,
        function_variable=joined_expression, dummy_bands=dummy_bands,
        calculation_datatype=calc_data_type, use_value_as_nodata=nodata_value,
        any_nodata_mask=True, output_raster_path=out_path,
        output_data_type=output_data_type, output_nodata_value=calc_nodata,
        compress=cfg.raster_compression, n_processes=n_processes,
        available_ram=available_ram, keep_output_argument=True,
        virtual_raster=vrt_r, progress_message='cross rasters',
        min_progress=50, max_progress=90
    )
    cfg.progress.update(message='output table', step=90)
    # calculate sum of values
    cfg.multiprocess.multiprocess_sum_array(nodata_value)
    sum_val = cfg.multiprocess.output
    if not output_table:
        return OutputManager(
            paths=[vrt_path], extra={
                'combinations': rec_combinations_array, 'sums': sum_val
            }
        )
    else:
        # get pixel unit
        (gt, crs, un, xy_count, nd, number_of_bands, block_size, scale_offset,
         data_type) = raster_vector.raster_info(out_path)
        p_x = gt[1]
        p_y = abs(gt[5])
        joined_table = tm.join_tables(
            table1=rec_combinations_array, table2=sum_val,
            field1_name='new_val', field2_name='new_val',
            nodata_value=cfg.nodata_val_Int64, join_type='left',
            progress_message=False, min_progress=90, max_progress=100
        )
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
            joined_table[joined_table['sum'] != cfg.nodata_val_Int64],
            combination_name_list, un, p_x, p_y
        )
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
        return OutputManager(
            paths=[out_path, tbl_out], extra={
                'combinations': rec_combinations_array,
                'sums': sum_val
            }
        )


#
def _combination_table(
        table: np.array, combination_names: list, crs_unit: str,
        pixel_size_x: int, pixel_size_y: int
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
        
    Returns:
        The text of the combination table.
    """
    text = []
    cv = cfg.comma_delimiter
    nl = cfg.new_line
    # table
    if 'degree' not in crs_unit:
        output_field_names = ['RasterValue']
        input_field_names = ['new_val']
        for c in range(len(combination_names)):
            output_field_names.append(combination_names[c])
            input_field_names.append('f%s' % str(c))
        output_field_names.append('PixelSum')
        output_field_names.append('Area [%s^2]' % crs_unit)
        input_field_names.append('sum')
        input_field_names.append('area')
        cross_class = tm.calculate(
            matrix=table, expression_string='"sum" * %s * %s' % (
                str(pixel_size_x), str(pixel_size_y)),
            output_field_name='area', progress_message=False
        )
    else:
        output_field_names = ['RasterValue']
        input_field_names = ['new_val']
        for c in range(len(combination_names)):
            output_field_names.append(combination_names[c])
            input_field_names.append('f%s' % str(c))
        output_field_names.append('PixelSum')
        input_field_names.append('sum')
        cross_class = table
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
