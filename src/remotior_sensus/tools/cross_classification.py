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
"""Cros classification.

This tool performs the cross classification which is similar 
to band combination, but it is executed between two files only.
The reference file can also be of vector type.
A unique value is assigned to each combination of values.
The output is a raster made of unique values corresponding to combinations
of values.
An output text file describes the correspondance between unique values
and combinations, as well as the statistics of each combination,
with the option to calculate an error matrix or linear regression.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> cross = rs.cross_classification(classification_path='file1.tif',reference_path='file2.tif',output_path='output.tif')
"""  # noqa: E501

import io
from typing import Optional

import numpy as np

from remotior_sensus.core import configurations as cfg, table_manager as tm
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.tools.band_combination import band_combination
from remotior_sensus.util import (
    files_directories, raster_vector, read_write_files, shared_tools
)


def cross_classification(
        classification_path: str, reference_path: str,
        output_path: Optional[str] = None, overwrite: Optional[bool] = False,
        vector_field: Optional[str] = None, nodata_value: Optional[int] = None,
        cross_matrix: Optional[bool] = False,
        regression_raster: Optional[bool] = False,
        error_matrix: Optional[bool] = False,
        extent_list: Optional[list] = None,
        n_processes: Optional[int] = None, available_ram: Optional[int] = None
) -> OutputManager:
    """Calculation of cross classification.

    This tool allows for the cross classification of two files (a
    classification raster and a reference vector or raster) in order to get
    a raster where each value corresponds to a combination of class values.
    Input raster values must be integer type.
    The output is a cross raster and, depending on the parameters, a text
    file reporting the statistics of each combination, error matrix,
    or linear regression statistics.

    Args:
        classification_path: path of raster used as classification input.
        reference_path: path of the vector or raster file used as reference input.
        vector_field: in case of vector reference, the name of the field used as reference value.
        output_path: path of the output raster.
        overwrite: if True, output overwrites existing files.
        nodata_value: value to be considered as nodata.
        cross_matrix: if True then calculate the cross matrix.
        regression_raster: if True then calculate linear regression statistics.
        error_matrix: if True then calculate error matrix.
        extent_list: list of boundary coordinates left top right bottom.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.

    Returns:
        Object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - paths = [output raster path, output table path]

    Examples:
        Perform the cross classification between two files
            >>> cross = cross_classification(classification_path='file1.tif',reference_path='file2.tif',output_path='output.tif')

        Perform the cross classification between two files and calculate the error matrix
            >>> cross = cross_classification(classification_path='file1.tif',reference_path='file2.tif',output_path='output.tif',error_matrix=True)
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=True
    )
    # check output path
    out_path, vrt_r = files_directories.raster_output_path(
        output_path,
        overwrite=overwrite
        )
    vector, raster, reference_crs = raster_vector.raster_or_vector_input(
        reference_path
    )
    if extent_list is not None:
        if raster:
            # prepare process files
            prepared = shared_tools.prepare_process_files(
                input_bands=[classification_path, reference_path],
                output_path=output_path,
                overwrite=overwrite, n_processes=n_processes,
                box_coordinate_list=extent_list,
                multiple_output=True, multiple_input=True
            )
            input_raster_list = prepared['input_raster_list']
            n_processes = prepared['n_processes']
            classification_path, reference_path = input_raster_list
        else:
            # prepare process files
            prepared = shared_tools.prepare_process_files(
                input_bands=[classification_path], output_path=output_path,
                overwrite=overwrite, n_processes=n_processes,
                box_coordinate_list=extent_list,
                multiple_output=True, multiple_input=True
            )
            input_raster_list = prepared['input_raster_list']
            n_processes = prepared['n_processes']
            classification_path = input_raster_list[0]
    classification_crs = raster_vector.get_crs(classification_path)
    # check crs
    same_crs = raster_vector.compare_crs(reference_crs, classification_crs)
    # if reference is raster
    if raster:
        if not same_crs:
            t_pmd = cfg.temp.temporary_raster_path(extension=cfg.vrt_suffix)
            reference_raster = cfg.multiprocess.create_warped_vrt(
                raster_path=reference_path, output_path=t_pmd,
                output_wkt=str(classification_crs)
            )
        else:
            reference_raster = reference_path
    # if reference is vector
    else:
        if vector_field is None:
            cfg.logger.log.error('vector field missing')
            cfg.messages.error('vector field missing')
            return OutputManager(check=False)
        if not same_crs:
            # project vector to raster crs
            t_vector = cfg.temp.temporary_file_path(
                name_suffix=cfg.gpkg_suffix
            )
            try:
                raster_vector.reproject_vector(
                    reference_path, t_vector, reference_crs, classification_crs
                )
            except Exception as err:
                cfg.logger.log.error(str(err))
                cfg.messages.error(str(err))
                return OutputManager(check=False)
            reference_path = t_vector
        # convert vector to raster
        reference_raster = cfg.temp.temporary_raster_path(
            extension=cfg.tif_suffix
        )
        # perform conversion
        cfg.multiprocess.multiprocess_vector_to_raster(
            vector_path=reference_path, field_name=vector_field,
            output_path=reference_raster,
            reference_raster_path=classification_path, nodata_value=0,
            background_value=0, available_ram=available_ram,
            minimum_extent=False
        )

    combination = band_combination(
        input_bands=[reference_raster, classification_path],
        output_path=out_path, nodata_value=nodata_value,
        n_processes=n_processes, available_ram=available_ram,
        output_table=False, progress_message=False
    )
    vrt_check = combination.paths[0]
    rec_combinations_array = combination.extra['combinations']
    sum_val = combination.extra['sums']
    cfg.progress.update(message='output table', step=90)
    # get pixel unit
    (gt, crs, un, xy_count, nd, number_of_bands, block_size, scale_offset,
     data_type) = raster_vector.raster_info(out_path)
    p_x = gt[1]
    p_y = abs(gt[5])
    joined_table = tm.join_tables(
        table1=rec_combinations_array, table2=sum_val, field1_name='new_val',
        field2_name='new_val', nodata_value=cfg.nodata_val_Int64,
        join_type='left', progress_message=False
    )
    # create table
    table, slope, intercept, unique_values = _cross_table(
        table=joined_table[joined_table['sum'] != cfg.nodata_val_Int64],
        crs_unit=un, pixel_size_x=p_x, pixel_size_y=p_y,
        regression_raster=regression_raster, cross_matrix=cross_matrix,
        error_matrix=error_matrix
    )
    # save combination to table
    tbl_out = shared_tools.join_path(
        files_directories.parent_directory(out_path), '{}{}'.format(
            files_directories.file_name(out_path, suffix=False), cfg.csv_suffix
        )
    ).replace('\\', '/')
    read_write_files.write_file(table, tbl_out)
    out_raster_b0 = None
    out_raster_b1 = None
    # regression raster
    if regression_raster:
        cfg.progress.update(message='regression raster', step=98)
        # output rasters
        out_raster_b0 = shared_tools.join_path(
            files_directories.parent_directory(out_path), '{}_b0{}'.format(
                files_directories.file_name(out_path, suffix=False),
                cfg.tif_suffix
            )
        ).replace('\\', '/')
        out_raster_b1 = shared_tools.join_path(
            files_directories.parent_directory(out_path), '{}_b1{}'.format(
                files_directories.file_name(out_path, suffix=False),
                cfg.tif_suffix
            )
        ).replace('\\', '/')
        raster_vector.create_raster_from_reference(
            vrt_check, 1, [out_raster_b0], nodata_value=cfg.nodata_val_Float32,
            driver='GTiff', gdal_format='Float32',
            compress=cfg.raster_compression, compress_format='LZW',
            constant_value=intercept
        )
        cfg.progress.update(message='regression raster', step=99)
        raster_vector.create_raster_from_reference(
            vrt_check, 1, [out_raster_b1], nodata_value=cfg.nodata_val_Float32,
            driver='GTiff', gdal_format='Float32',
            compress=cfg.raster_compression, compress_format='LZW',
            constant_value=slope
        )
    cfg.progress.update(end=True)
    cfg.logger.log.info(
        'end; cross classification: %s; table: %s'
        % (str(out_path), str(tbl_out))
    )
    return OutputManager(
        paths=[out_path, tbl_out],
        extra={
            'unique_values': unique_values,
            'regression_raster_b0': out_raster_b0,
            'regression_raster_b1': out_raster_b1
        }
    )


# create text for table
# noinspection PyTypeChecker
def _cross_table(
        table, crs_unit, pixel_size_x, pixel_size_y, regression_raster=False,
        cross_matrix=False,
        error_matrix=False
):
    slope = ''
    intercept = ''
    text = []
    cv = cfg.comma_delimiter
    nl = cfg.new_line
    # table
    if 'degree' not in crs_unit:
        output_field_names = ['RasterValue', 'Reference', 'Classification',
                              'PixelSum', 'Area [%s^2]' % crs_unit]
        input_field_names = ['new_val', 'f0', 'f1', 'sum', 'area']
        cross_class = tm.calculate(
            matrix=table, expression_string='"sum" * %s * %s' % (
                str(pixel_size_x), str(pixel_size_y)),
            output_field_name='area', progress_message=False
        )
    else:
        output_field_names = ['RasterValue', 'Reference', 'Classification',
                              'PixelSum']
        input_field_names = ['new_val', 'f0', 'f1', 'sum']
        cross_class = table
    redefined = tm.redefine_matrix_columns(
        matrix=cross_class, input_field_names=input_field_names,
        output_field_names=output_field_names, progress_message=False
    )
    # export matrix creating stream handler
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
    columns = np.unique(
        np.append(cross_class['f0'], cross_class['f1'])
    ).tolist()
    # cross matrix
    if cross_matrix:
        if 'degree' not in crs_unit:
            text.append('CROSS MATRIX [%s^2]' % (str(crs_unit)))
            text.append(nl)
            cross_matrix = tm.pivot_matrix(
                cross_class, row_field='f0', secondary_row_field_list=['f1'],
                column_function_list=[['area', 'sum']], cross_matrix=True,
                progress_message=False
            )
        else:
            text.append('CROSS MATRIX [PixelSum]')
            text.append(nl)
            cross_matrix = tm.pivot_matrix(
                cross_class, row_field='f0', secondary_row_field_list=['f1'],
                column_function_list=[['sum', 'sum']], cross_matrix=True,
                progress_message=False
            )
        text.append(cv)
        text.append('Classification >')
        text.append(nl)
        text.append('Reference V')
        text.append(cv)
        totals = []
        for c in tm.columns(cross_matrix):
            if c != 'f0':
                totals.append(str(int(cross_matrix[c].sum())))
                totals.append(cv)
                text.append(c)
                text.append(cv)
        totals.pop(-1)
        joined_totals = ''.join(totals)
        text.pop(-1)
        text.append(nl)
        # export matrix creating stream handler
        stream2 = io.StringIO()
        np.savetxt(stream2, cross_matrix, delimiter=cv, fmt='%1.2f')
        matrix2_value = stream2.getvalue()
        text.append(matrix2_value.replace('.00', ''))
        text.append('Total')
        text.append(cv)
        text.append(joined_totals)
        text.append(nl)
    elif error_matrix:
        # array matrix
        text.append('ERROR MATRIX [pixel count]')
        text.append(nl)
        text.append(cv)
        text.append('> Reference')
        text.append(nl)
        cross_matrix = tm.pivot_matrix(
            cross_class, row_field='f0', secondary_row_field_list=['f1'],
            column_function_list=[['sum', 'sum']], cross_matrix=True,
            progress_message=False
        )
        # regular error matrix
        err_matrix = np.zeros((len(columns) + 1, len(columns) + 2))
        # area based error matrix
        err_matrix_unbiased = np.zeros((len(columns) + 1, len(columns) + 3))
        text.append('V_Classified')
        text.append(cv)
        text_matrix_unbiased = ['V_Classified', cv]
        # copy values from cross matrix to error matrix
        c = 1
        for column in columns:
            text.append(str(column))
            text.append(cv)
            text_matrix_unbiased.append(str(column))
            text_matrix_unbiased.append(cv)
            if str(column) in tm.columns(cross_matrix):
                err_matrix[:len(columns), c] = cross_matrix[str(column)]
                # copy for accuracy metrics in case area based values are
                # not calculated
                err_matrix_unbiased[:len(columns), c] = cross_matrix[
                    str(column)]
            c += 1
        err_matrix[:len(columns), 0] = np.array(columns)
        text.append('Total')
        text.append(nl)
        total_pixels = err_matrix.sum() - err_matrix[::, 0].sum()
        # add column as total rows N_i
        err_matrix[::, -1] = err_matrix.sum(axis=1) - err_matrix[::, 0]
        # add row of total columns
        err_matrix[-1, ::] = err_matrix.sum(axis=0)
        # nodata value to be masked
        nd = -99999
        err_matrix[-1, 0] = nd
        # area based matrix as proportion P_ij of class over total pixels
        err_matrix_unbiased[::, 0:err_matrix_unbiased.shape[1] - 2] = \
            err_matrix[::, 0:err_matrix.shape[1] - 1] / total_pixels
        err_matrix_unbiased[::, err_matrix_unbiased.shape[1] - 2] = \
            err_matrix[::, err_matrix.shape[1] - 1] * (
                    pixel_size_x * pixel_size_y)
        err_matrix_unbiased[::, err_matrix_unbiased.shape[1] - 1] = \
            err_matrix[::, err_matrix.shape[1] - 1] / total_pixels
        # replace first columns
        err_matrix_unbiased[:-1, 0] = err_matrix[:-1, 0]
        err_matrix_unbiased[-1, 0] = nd
        # export matrix creating stream handler
        stream = io.StringIO()
        # error matrix
        np.savetxt(stream, err_matrix, delimiter=cv, fmt='%i')
        err_matrix_value = stream.getvalue()
        text.append(
            err_matrix_value.replace('.00', '').replace(str(nd), 'Total')
        )
        if 'degree' not in crs_unit:
            text.append(nl)
            # area based error matrix (see Olofsson, et al., 2014,
            # Good practices for estimating area and assessing
            # accuracy of land change, Remote Sensing of Environment, 148,
            # 42-57)
            text.append(
                'AREA BASED ERROR MATRIX%s%s> Reference%s' % (nl, cv, nl)
            )
            # added column Wi is area proportion for class i
            joined_text_matrix_unbiased = ''.join(text_matrix_unbiased)
            text.append(joined_text_matrix_unbiased)
            text.append('Area%sWi%s' % (cv, nl))
            # export matrix creating stream handler
            stream_i = io.StringIO()
            # area based matrix
            np.savetxt(stream_i, err_matrix_unbiased, delimiter=cv, fmt='%.4f')
            err_matrix_unbiased_value = stream_i.getvalue()
            text.append(
                err_matrix_unbiased_value.replace('.0000', '').replace(
                    str(nd), 'Total'
                )
            )
            # estimated area as population proportion P_ij * total_area
            total_area = err_matrix_unbiased[
                         :-1, err_matrix_unbiased.shape[1] - 2].sum()
            estimated_area_matrix = err_matrix_unbiased[-1, 1:-2] * total_area
            estimated_area_matrix = estimated_area_matrix.reshape(
                1, estimated_area_matrix.shape[0]
            )
            # export matrix creating stream handler
            stream_estimated_area = io.StringIO()
            np.savetxt(
                stream_estimated_area, estimated_area_matrix, delimiter=cv,
                fmt='%.2f'
            )
            estimated_area_text = stream_estimated_area.getvalue()
            text.append(
                'Estimated area%s%s%s%s%s' % (
                    cv, estimated_area_text.replace('.00', '').rstrip(nl), cv,
                    str('%1.2f' % total_area).replace('.00', ''), nl)
            )
            # standard error SE_j as summation for each class i of (Wi *
            # P_ij - P_ij^2) / (N_i - 1)
            # create W_i and N_i -1 matrices for calculation
            w_i = np.zeros((len(columns), len(columns)))
            n_i = np.zeros((len(columns), len(columns)))
            for c in range(len(columns)):
                w_i[::, c] = err_matrix_unbiased[:-1, -1]
                n_i[::, c] = err_matrix[:-1, -1] - 1
            std_err_matrix = (err_matrix_unbiased[
                              :-1, 1:-2] * w_i - err_matrix_unbiased[
                                                 :-1, 1:-2] ** 2) / n_i
            std_err = np.sqrt(np.nansum(std_err_matrix, axis=0))
            std_err = std_err.reshape(1, std_err.shape[0])
            # export matrix creating stream handler
            stream_std_err = io.StringIO()
            np.savetxt(stream_std_err, std_err, delimiter=cv, fmt='%.4f')
            std_err_text = stream_std_err.getvalue()
            text.append('SE%s%s' % (cv, std_err_text.replace('.0000', '')))
            # standard error area
            std_err_area = std_err * total_pixels * pixel_size_x * pixel_size_y
            stream_std_err_area = io.StringIO()
            np.savetxt(
                stream_std_err_area, std_err_area, delimiter=cv, fmt='%.2f'
            )
            std_err_area_text = stream_std_err_area.getvalue()
            text.append(
                'SE area%s%s' % (cv, std_err_area_text.replace('.00', ''))
            )
            # confidence interval
            confidence_interval = std_err * total_pixels * (
                    pixel_size_x * pixel_size_y) * 1.96
            stream_confidence_interval = io.StringIO()
            np.savetxt(
                stream_confidence_interval, confidence_interval, delimiter=cv,
                fmt='%.2f'
            )
            confidence_interval_text = stream_confidence_interval.getvalue()
            text.append(
                '95% CI area{}{}'.format(
                    cv, confidence_interval_text.replace('.00', '')
                )
            )
        # accuracy metrics
        nii_tot = 0
        text_p = ['PA [%]']
        text_u = ['UA [%]']
        try:
            for g in range(len(columns)):
                nii = err_matrix_unbiased[g, g + 1]
                nii_tot = nii_tot + nii
                nip = err_matrix_unbiased[g, 1:(len(columns) + 1)].sum()
                npi = err_matrix_unbiased[0:len(columns), g + 1].sum()
                if npi != 0:
                    p = 100 * nii / npi
                else:
                    p = np.nan
                if nip != 0:
                    u = 100 * nii / nip
                else:
                    u = np.nan
                text_p.append(cv)
                text_p.append(str('%1.4f' % p))
                text_u.append(cv)
                text_u.append(str('%1.4f' % u))
        except Exception as err:
            cfg.logger.log.error(str(err))
        joined_text_p = ''.join(text_p)
        joined_text_u = ''.join(text_u)
        text.append(joined_text_p)
        text.append(nl)
        text.append(joined_text_u)
        text.append(nl)
        text.append(nl)
        text.append(
            'Overall accuracy [%] = {}'.format('%1.4f' % (nii_tot * 100))
        )
        text.append(nl)
        text.append(nl)
        text.append('Area unit = %s^2' % crs_unit)
        text.append(nl)
        text.append('SE = standard error')
        text.append(nl)
        text.append('CI = confidence interval')
        text.append(nl)
        text.append("PA = producer's accuracy")
        text.append(nl)
        text.append("UA = user's accuracy")
        text.append(nl)
    # calculate regression
    if regression_raster:
        # linear regression y = b0 + b1 x + E
        r_coefficient = ''
        r_coefficient2 = ''
        var_y = ''
        var_slope = ''
        conf_slope = ''
        var_intercept = ''
        conf_intercept = ''
        # sum_x and sum_y, total, x_mean and y_mean for calculation of sums
        # S_xx, S_yy and S_xy
        r_sum_x = np.nansum(cross_class['f0'] * cross_class['sum'])
        r_sum_y = np.nansum(cross_class['f1'] * cross_class['sum'])
        r_sum_tot = cross_class['sum'].sum()
        r_x_mean = r_sum_x / r_sum_tot
        r_y_mean = r_sum_y / r_sum_tot
        # S_xy = summation from 1 to n of (x - x_mean) * (y - y_mean)
        s_xy = np.nansum(
            (cross_class['f0'] - r_x_mean) * (cross_class['f1'] - r_y_mean) *
            cross_class['sum']
        )
        # S_xx = summation from 1 to n of (x - x_mean)^2
        s_xx = np.nansum(
            ((cross_class['f0'] - r_x_mean) ** 2) * cross_class['sum']
        )
        # S_yy = summation from 1 to n of (y - y_mean)^2
        s_yy = np.nansum(
            ((cross_class['f1'] - r_y_mean) ** 2) * cross_class['sum']
        )
        try:
            r_coefficient = s_xy / (s_xx * s_yy) ** 0.5
            r_coefficient2 = r_coefficient ** 2
            slope = s_xy / s_xx
            intercept = r_y_mean - slope * r_x_mean
            var_y = (s_yy - slope * s_xy) / (r_sum_tot - 2)
            var_slope = var_y / s_xx
            var_intercept = var_y * (1 / r_sum_tot + r_x_mean ** 2 / s_xx)
            conf_slope = 2 * var_slope ** 0.5
            conf_intercept = 2 * var_intercept ** 0.5
        except Exception as err:
            cfg.logger.log.error(str(err))
        text.append(nl)
        text.append('Linear regression Y = B0 + B1*X')
        text.append(nl)
        text.append(
            'Coeff. det. R^2%s%s' % (cfg.tab_delimiter, str(r_coefficient2))
        )
        text.append(nl)
        text.append(
            'Coeff. correlation r%s%s' % (
                cfg.tab_delimiter, str(r_coefficient))
        )
        text.append(nl)
        text.append(
            'B0%s%s ± %s' % (
                cfg.tab_delimiter, str(intercept), str(conf_intercept))
        )
        text.append(nl)
        text.append(
            'B1%s%s ± %s' % (cfg.tab_delimiter, str(slope), str(conf_slope))
        )
        text.append(nl)
        text.append('Variance Y%s%s' % (cfg.tab_delimiter, str(var_y)))
        text.append(nl)
        text.append(
            'Variance B0%s%s' % (cfg.tab_delimiter, str(var_intercept))
        )
        text.append(nl)
        text.append('Variance B1%s%s' % (cfg.tab_delimiter, str(var_slope)))
        text.append(nl)
    joined_text = ''.join(text)
    return joined_text, slope, intercept, columns
