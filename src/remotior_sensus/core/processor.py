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

import datetime
# garbage collector for memory issue
import gc
import os

import numpy as np
from numpy.lib import recfunctions as rfn

from remotior_sensus.core import configurations as cfg, table_manager as tm
from remotior_sensus.core.log import Log
from remotior_sensus.util import files_directories, raster_vector


def function_initiator(
        process_parameters=None, input_parameters=None, output_parameters=None,
        function=None, function_argument=None, function_variable=None,
        run_separate_process=False, classification=False,
        classification_confidence=False, signature_raster=False
):
    # get process parameters
    process_id = str(process_parameters[0])
    cfg.temp = process_parameters[1]
    memory = process_parameters[2]
    gdal_path = process_parameters[3]
    progress_queue = process_parameters[4]
    refresh_time = process_parameters[5]
    # get input raster parameters
    raster_list = input_parameters[0]
    calc_data_type = input_parameters[1]
    boundary_size = input_parameters[2]
    if input_parameters[3] is None:
        x_min_piece = 0
        y_min_piece = 0
        x_size_piece = y_size_piece = sections = None
    else:
        x_min_piece = input_parameters[3].x_min
        x_size_piece = input_parameters[3].x_size
        sections = input_parameters[3].sections
        y_min_piece_no_boundary = input_parameters[3].y_min_no_boundary
        if y_min_piece_no_boundary is not None:
            y_min_piece = y_min_piece_no_boundary
        else:
            y_min_piece = input_parameters[3].y_min
        y_size_piece_no_boundary = input_parameters[3].y_size_no_boundary
        if y_size_piece_no_boundary is not None:
            y_size_piece = y_size_piece_no_boundary
        else:
            y_size_piece = input_parameters[3].y_size
    scale = input_parameters[4]
    offset = input_parameters[5]
    use_value_as_nodata = input_parameters[6]
    single_band_number = input_parameters[7]
    input_nodata_as_value = input_parameters[8]
    multi_add_factors = input_parameters[9]
    dummy_bands = input_parameters[10]
    specific_output = input_parameters[11]
    if specific_output is not None:
        specific_output_piece = specific_output['pieces'][int(process_id)]
    else:
        specific_output_piece = None
    # get output parameters
    (output_raster_list, output_data_type, compress, compress_format,
     any_nodata_mask, output_no_data, output_band_number, keep_output_array,
     keep_output_argument) = output_parameters
    # start logger
    cfg.logger = Log(directory=cfg.temp.dir, multiprocess=str(process_id))
    # set gdal path
    if gdal_path is not None:
        for d in gdal_path.split(';'):
            try:
                os.add_dll_directory(d)
                cfg.gdal_path = d
            except Exception as err:
                str(err)
    # import gdal
    from osgeo import gdal
    # GDAL config
    try:
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(memory))
        gdal.SetConfigOption('VSI_CACHE', 'FALSE')
        gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
    except Exception as err:
        str(err)
    cfg.logger.log.debug('start')
    cfg.logger.log.debug(
        'keep_output_array: %s; keep_output_argument: %s; any_nodata_mask: %s;'
        ' input_nodata_as_value: %s'
        % (keep_output_array, keep_output_argument, any_nodata_mask,
           input_nodata_as_value)
    )
    # list of output arrays
    output_array_list = []
    # output files
    out_files = []
    # output classification
    output_signature_raster = {}
    output_signature_raster_dic = {}
    out_class = out_alg = None
    # process error
    proc_error = False
    # raster counter
    raster_count = 0
    start_time = datetime.datetime.now()
    cfg.logger.log.debug(
        'raster_list: %s; x_size_piece: %s; y_size_piece: %s:'
        % (raster_list, x_size_piece, y_size_piece)
    )
    # iterate over input raster list
    for raster in raster_list:
        # reset section counter
        count_section_progress = 1
        calculation_datatype = calc_data_type[raster_count]
        cfg.logger.log.debug(
            'process raster: %s; calculation_datatype: %s'
            % (raster, calculation_datatype)
        )
        if use_value_as_nodata:
            value_as_nodata = use_value_as_nodata[raster_count]
        else:
            value_as_nodata = None
        (r_gt, r_crs, r_un, r_xy_count, r_nd, band_number, r_block_size,
         r_scale_offset, r_data_type) = raster_vector.raster_info(raster)
        cfg.logger.log.debug('r_gt: {}'.format(str(r_gt)))
        # pixel size and origin from reference
        t_lx = r_gt[0]
        t_ly = r_gt[3]
        p_sx = r_gt[1]
        p_sy = r_gt[5]
        # scale and offset
        if scale:
            if scale[raster_count] is not None:
                scl = float(scale[raster_count])
            else:
                scl = 1.0
        else:
            scl = 1.0
        if offset:
            if offset[raster_count] is not None:
                offs = float(offset[raster_count])
            else:
                offs = 0.0
        else:
            offs = 0.0
        # calculate sections for separate process
        if run_separate_process:
            # raster size
            r_x, r_y = r_xy_count
            # raster blocks
            memory_unit = cfg.memory_unit_array_12
            (block_size_x, block_size_y,
             list_range_x, list_range_y, tot_blocks) = _calculate_block_size(
                x_block=r_x, y_block=r_y, available_ram=memory,
                memory_unit=memory_unit, dummy_bands=dummy_bands
            )
            cfg.logger.log.debug('list_range_y: %s' % str(list_range_y))
            sections = []
            for y_min in range(len(list_range_y)):
                # section y_min size
                s_y_size = block_size_y
                # adapt to raster y_min size
                if list_range_y[y_min] + s_y_size > r_y:
                    s_y_size = r_y - list_range_y[y_min]
                sections.append(
                    RasterSection(
                        x_min=0, y_min=list_range_y[y_min], x_max=block_size_x,
                        y_max=list_range_y[y_min] + s_y_size,
                        x_size=block_size_x, y_size=s_y_size
                    )
                )
        cfg.logger.log.debug('sections: %s' % str(sections))
        # classification rasters
        classification_rasters = []
        # algorithm rasters
        algorithm_rasters = []
        # set raster parameters
        output_argument = None
        # generic output raster
        out_generic = None
        # specific output raster
        out_specific = None
        # output data type
        if output_data_type:
            out_data_type = output_data_type[raster_count]
        else:
            out_data_type = None
        # output nodata
        if output_no_data:
            out_no_data = output_no_data[raster_count]
        else:
            out_no_data = None
        # create output rasters if process with output files
        if output_raster_list[0] is not None:
            # output geo transform
            upper_left_x_range = t_lx + x_min_piece * p_sx
            upper_left_y_range = t_ly + y_min_piece * p_sy
            geo_transform = (upper_left_x_range, p_sx, r_gt[2],
                             upper_left_y_range, r_gt[4], p_sy)
            # classification output
            if classification:
                signatures = function_argument[0][
                    cfg.spectral_signatures_framework]
                # get selected signatures
                signatures_table = signatures.table[
                    signatures.table.selected == 1]
                # list of rasters to be created
                output_classification_raster_list = []
                output_signature_raster = {}
                output_signature_raster_dic = {}
                # create output raster paths
                out_class = cfg.temp.temporary_raster_path(
                    name='class_', name_suffix=process_id
                )
                output_classification_raster_list.append(out_class)
                classification_rasters.append(out_class)
                # signature output rasters
                if signature_raster:
                    for s in signatures_table.signature_id.tolist():
                        sig_name = '{}{}_{}_'.format(
                            s, signatures_table[
                                signatures_table.signature_id ==
                                s].macroclass_id[0],
                            signatures_table[
                                signatures_table.signature_id == s].class_id[0]
                        )
                        signature_file_path = cfg.temp.temporary_raster_path(
                            name=sig_name, name_suffix=process_id
                        )
                        output_signature_raster[s] = signature_file_path
                        if s in output_signature_raster_dic:
                            output_signature_raster_dic[s].append(
                                signature_file_path
                            )
                        else:
                            output_signature_raster_dic[
                                s] = [signature_file_path]
                        output_classification_raster_list.append(
                            signature_file_path
                        )
                # confidence raster
                if classification_confidence:
                    out_alg = cfg.temp.temporary_raster_path(
                        name='alg_', name_suffix=process_id
                    )
                    output_classification_raster_list.append(out_alg)
                    algorithm_rasters.append(out_alg)
                # create rasters
                raster_vector.create_raster_from_reference(
                    path=raster, band_number=1,
                    output_raster_list=output_classification_raster_list,
                    nodata_value=out_no_data, driver='GTiff',
                    gdal_format=out_data_type, compress=compress,
                    compress_format=compress_format,
                    geo_transform=geo_transform, x_size=x_size_piece,
                    y_size=y_size_piece
                )
                cfg.logger.log.debug(
                    'output_classification_raster_list: %s'
                    % str(output_classification_raster_list)
                )
            # specific output
            elif specific_output is not None:
                x_min_piece_s = specific_output_piece.x_min
                y_min_piece_s = specific_output_piece.y_min
                x_size_piece_s = specific_output_piece.x_size
                y_size_piece_s = specific_output_piece.y_size
                # output geo transform
                upper_left_x_range = (specific_output['geo_transform'][0]
                                      + x_min_piece_s
                                      * specific_output['geo_transform'][1])
                upper_left_y_range = (specific_output['geo_transform'][3]
                                      + y_min_piece_s
                                      * specific_output['geo_transform'][5])
                geo_transform = (upper_left_x_range,
                                 specific_output['geo_transform'][1],
                                 specific_output['geo_transform'][2],
                                 upper_left_y_range,
                                 specific_output['geo_transform'][4],
                                 specific_output['geo_transform'][5])
                out_specific = cfg.temp.temporary_raster_path(
                    name_suffix=process_id, name_prefix=process_id
                )
                file_output = raster_vector.create_raster_from_reference(
                    raster, 1, [out_specific], out_no_data, 'GTiff',
                    out_data_type, compress, compress_format,
                    geo_transform=geo_transform, x_size=x_size_piece_s,
                    y_size=y_size_piece_s
                )
                cfg.logger.log.debug('specific_output: %s' % file_output)
            # generic output
            else:
                out_generic = cfg.temp.temporary_raster_path(
                    name_suffix=process_id, name_prefix=process_id
                )
                file_output = raster_vector.create_raster_from_reference(
                    raster, 1, [out_generic], out_no_data, 'GTiff',
                    out_data_type, compress, compress_format,
                    geo_transform=geo_transform, x_size=x_size_piece,
                    y_size=y_size_piece
                )
                cfg.logger.log.debug('file_output: %s' % file_output)
        # iterate over sections
        for sec in sections:
            # open input_raster with GDAL
            _r_d = gdal.Open(raster, gdal.GA_ReadOnly)
            cfg.logger.log.debug('raster: %s' % raster)
            # list of bands
            gdal_band_list = []
            if single_band_number is None:
                for b in range(1, band_number + 1):
                    r_b = _r_d.GetRasterBand(b)
                    gdal_band_list.append(r_b)
            else:
                r_b = _r_d.GetRasterBand(single_band_number + 1)
                gdal_band_list.append(r_b)
            _input_array = np.zeros(
                (sec.y_size, sec.x_size, len(gdal_band_list)),
                dtype=calculation_datatype
            )
            nodata_mask = None
            nd_val = None
            # read bands
            for b in range(len(gdal_band_list)):
                cfg.logger.log.debug(
                    'sec.x_min: %s; sec.y_min: %s; sec.x_size: %s; '
                    'sec.y_size: %s'
                    % (sec.x_min, sec.y_min, sec.x_size, sec.y_size)
                )
                # band array
                _a = raster_vector.read_array_block(
                    gdal_band_list[b], sec.x_min, sec.y_min, sec.x_size,
                    sec.y_size, calculation_datatype
                )
                # get band nodata, scale and offset
                try:
                    # if single band
                    band_b = gdal_band_list[b].GetRasterBand(1)
                    offs_b = band_b.GetOffset()
                    scl_b = band_b.GetScale()
                    if scl_b is not None:
                        scl_b = scl_b
                    if offs_b is not None:
                        offs_b = offs_b
                    offs_b = np.asarray(offs_b).astype(_a.dtype)
                    scl_b = np.asarray(scl_b).astype(_a.dtype)
                    nd_val = band_b.GetNoDataValue()
                    ndv_band = np.asarray(
                        nd_val * scl_b + offs_b
                    ).astype(_a.dtype)
                except Exception as err:
                    str(err)
                    try:
                        offs_b = gdal_band_list[b].GetOffset()
                        scl_b = gdal_band_list[b].GetScale()
                        if scl_b is not None:
                            scl_b = scl_b
                        else:
                            scl_b = 1
                        if offs_b is not None:
                            offs_b = offs_b
                        else:
                            offs_b = 0
                        offs_b = np.asarray(offs_b).astype(_a.dtype)
                        scl_b = np.asarray(scl_b).astype(_a.dtype)
                        nd_val = gdal_band_list[b].GetNoDataValue()
                        ndv_band = np.asarray(
                            nd_val * scl_b + offs_b
                        ).astype(_a.dtype)
                    except Exception as err:
                        str(err)
                        ndv_band = None
                if ndv_band is not None:
                    # adapt NoData to dtype
                    ndv_band = np.asarray(ndv_band).astype(
                        calculation_datatype
                    )
                if _a is None:
                    proc_error = 'input raster'
                else:
                    # apply multiplicative and additive factors to array
                    if multi_add_factors is not None:
                        _a = array_multiplicative_additive_factors(
                            _a, multi_add_factors[0][b],
                            multi_add_factors[1][b]
                        )
                    _input_array[::, ::, b] = _a.astype(calculation_datatype)
                # set nodata value
                if calculation_datatype == np.float32 or \
                        calculation_datatype == np.float64:
                    if ndv_band is not None:
                        try:
                            _input_array[::, ::, b][
                                _input_array[::, ::, b] == ndv_band] = np.nan
                        except Exception as err:
                            str(err)
                    if input_nodata_as_value:
                        try:
                            _input_array[::, ::, b][
                                np.isnan(_input_array[::, ::, b])] = ndv_band
                        except Exception as err:
                            str(err)
                cfg.logger.log.debug(
                    'gdal_band_list[%s].DataType: %s; nd_val: %s; ndv_band: %s'
                    % (b, gdal_band_list[b].DataType, nd_val, ndv_band)
                )
                # apply NoData mask
                if any_nodata_mask:
                    try:
                        nodata_mask[0, 0]
                    except Exception as err:
                        str(err)
                        nodata_mask = np.zeros(
                            (sec.y_size, sec.x_size),
                            dtype=calculation_datatype
                        )
                    if input_nodata_as_value:
                        pass
                    elif ndv_band is not None:
                        try:
                            np.copyto(
                                nodata_mask, out_no_data,
                                where=_input_array[::, ::, b] == np.asarray(
                                    ndv_band
                                )
                            )
                        except Exception as err:
                            str(err)
                    if input_nodata_as_value:
                        pass
                    else:
                        try:
                            np.copyto(
                                nodata_mask, out_no_data,
                                where=np.isnan(_input_array[::, ::, b])
                            )
                        except Exception as err:
                            str(err)
                    if input_nodata_as_value:
                        pass
                    elif value_as_nodata is not None:
                        value_as_nodata = np.asarray(value_as_nodata).astype(
                            _a.dtype
                        ).astype(calculation_datatype)
                        try:
                            np.copyto(
                                nodata_mask, out_no_data,
                                where=_input_array[
                                      ::, ::, b] == value_as_nodata
                            )
                        except Exception as err:
                            str(err)
                # create NoData mask from NoData values
                elif not any_nodata_mask:
                    _all_nodata_mask = np.zeros(
                        (sec.y_size, sec.x_size), dtype=calculation_datatype
                    )
                    try:
                        nodata_mask[0, 0]
                    except Exception as err:
                        str(err)
                        nodata_mask = np.ones(
                            (sec.y_size, sec.x_size),
                            dtype=calculation_datatype
                        ) * out_no_data
                    if input_nodata_as_value:
                        pass
                    elif ndv_band is not None:
                        try:
                            np.copyto(
                                _all_nodata_mask, out_no_data,
                                where=_input_array[::, ::, b] == ndv_band
                            )
                        except Exception as err:
                            str(err)
                    if input_nodata_as_value:
                        pass
                    else:
                        try:
                            np.copyto(
                                _all_nodata_mask, out_no_data,
                                where=np.isnan(_input_array[::, ::, b])
                            )
                        except Exception as err:
                            str(err)
                    if input_nodata_as_value:
                        pass
                    elif value_as_nodata is not None:
                        value_as_nodata = np.asarray(value_as_nodata).astype(
                            _a.dtype
                        ).astype(calculation_datatype)
                        try:
                            np.copyto(
                                _all_nodata_mask, out_no_data,
                                where=_input_array[
                                      ::, ::, b] == value_as_nodata
                            )
                        except Exception as err:
                            str(err)
                    nodata_mask = np.where(
                        (_all_nodata_mask == out_no_data) & (
                                nodata_mask == out_no_data),
                        out_no_data, 0
                    )
                    # release memory
                    _all_nodata_mask = None
            # release memory
            _a = None
            # get function variable
            if function_variable:
                f_variable = function_variable[raster_count]
            else:
                f_variable = None
            # get output raster
            if output_raster_list[0]:
                output_list = output_raster_list[raster_count]
            else:
                output_list = None
            # get output band number
            if output_band_number:
                out_band_number = output_band_number[raster_count]
            else:
                out_band_number = None
            # get function argument
            if function_argument:
                function_arg = function_argument[raster_count]
            else:
                function_arg = None
            # execute main function
            function_output = function(
                [scl, offs, out_no_data], _input_array, nodata_mask,
                sec.y_size, sec.x_min, sec.y_min, output_list,
                function_arg, f_variable, out_band_number,
                [x_min_piece, y_min_piece], output_signature_raster,
                out_class, out_alg
            )
            # release memory
            _input_array = None
            # check function output list
            if isinstance(function_output, list):
                output_argument = function_output[1]
                output_array = function_output[0]
            else:
                proc_error = 'error function'
                output_array = None
            # classification
            if classification:
                output_array = None
                output_argument = None
                output_array_list.append(
                    [classification_rasters, algorithm_rasters,
                     output_signature_raster_dic]
                )
                cfg.logger.log.debug('classification: %s' % function_output[0])
                if not function_output[0] and function_output[0] is not None:
                    proc_error = 'error classification'
                    cfg.logger.log.error('classification failed')
            # keep output array and output argument
            elif keep_output_array and keep_output_array is not None and (
                    keep_output_argument is True and keep_output_argument is
                    not None):
                output_array_list.append([output_array, output_argument])
            # keep output array
            elif keep_output_array and keep_output_array is not None:
                output_array_list.append([output_array, None])
            # keep output argument
            elif keep_output_argument and keep_output_argument is not None:
                output_array_list.append([None, output_argument])
            # output files
            cfg.logger.log.debug('output_raster_list: %s' % output_raster_list)
            # write output array
            if (classification is not True
                    and output_raster_list[0] is not None):
                if specific_output is not None:

                    output_array[np.isnan(output_array)] = out_no_data

                    y_min_piece_s = specific_output_piece.y_min
                    if boundary_size is None:
                        # output minimum y (section of piece)
                        out_y_min = (specific_output_piece.sections[
                                        count_section_progress - 1].y_min
                                     - y_min_piece_s)
                    else:
                        # output minimum y (section of piece)
                        out_y_min = (
                                specific_output_piece.sections[
                                    count_section_progress
                                    - 1].y_min_no_boundary - y_min_piece_s)
                        # reduce array size without boundary
                        output_array = output_array[
                                       specific_output_piece.sections[
                                           count_section_progress
                                           - 1].y_size_boundary_top:(
                                               specific_output_piece.sections[
                                                   count_section_progress
                                                   - 1].y_size_boundary_top +
                                               specific_output_piece.sections[
                                                   count_section_progress
                                                   - 1].y_size_no_boundary),
                                       0:specific_output_piece.sections[
                                           count_section_progress - 1].x_max]
                    try:
                        write_out = raster_vector.write_raster(
                            out_specific, 0, out_y_min, output_array,
                            out_no_data, scl, offs
                        )
                        if write_out == out_specific:
                            if run_separate_process:
                                pass
                            else:
                                out_files.append([out_specific,
                                                  output_argument])
                    except Exception as err:
                        proc_error = 'error output'
                        cfg.logger.log.error(str(err))
                else:
                    try:
                        output_array[np.isnan(output_array)] = out_no_data
                        if boundary_size is None:
                            # output minimum y (section of piece)
                            out_y_min = sec.y_min - y_min_piece
                        else:
                            # output minimum y (section of piece)
                            out_y_min = sec.y_min_no_boundary - y_min_piece
                            # reduce array size without boundary
                            output_array = output_array[
                                           sec.y_size_boundary_top:(
                                                   sec.y_size_boundary_top
                                                   + sec.y_size_no_boundary),
                                           0:sec.x_max]
                        cfg.logger.log.debug('out_generic: %s' % out_generic)
                        write_out = raster_vector.write_raster(
                            out_generic, 0, out_y_min, output_array,
                            out_no_data, scl, offs
                        )
                        if write_out == out_generic:
                            if run_separate_process:
                                pass
                            else:
                                out_files.append([out_generic,
                                                  output_argument])
                    except Exception as err:
                        proc_error = 'error output'
                        cfg.logger.log.error(str(err))
            # progress
            now_time = datetime.datetime.now()
            elapsed_time = (now_time - start_time).total_seconds()
            if (process_id == '0' and elapsed_time > refresh_time
                    and not run_separate_process):
                start_time = now_time
                progress_queue.put(
                    [count_section_progress, len(sections)], False
                )
            elif (process_id == '1' and elapsed_time > refresh_time
                    and run_separate_process):
                start_time = now_time
                progress_queue.put(
                    [count_section_progress
                     + len(sections) * raster_count,
                     len(sections) * len(raster_list)], False
                )
            # close GDAL rasters
            for b in range(len(gdal_band_list)):
                gdal_band_list[b].FlushCache()
                gdal_band_list[b] = None
            _r_d = None
            gc.collect()
            count_section_progress += 1
        if classification is not True and output_raster_list[0] is not None:
            if run_separate_process:
                # move temporary files of separate process
                files_directories.move_file(
                    out_generic, output_raster_list[raster_count]
                )
                out_files.append(
                    [output_raster_list[raster_count],
                     output_argument]
                )
        raster_count += 1
        """
        now_time = datetime.datetime.now()
        elapsed_time = (now_time - start_time).total_seconds()
        if (process_id == '1' and elapsed_time > refresh_time
                and run_separate_process):
            progress_queue.put([raster_count, len(raster_list)], False)
        """
    cfg.logger.log.debug('end')
    logger = cfg.logger.stream.getvalue()
    return output_array_list, out_files, proc_error, logger


# calculate block size and pixel ranges
def _calculate_block_size(
        x_block, y_block, available_ram, memory_unit, dummy_bands
):
    single_block_size = x_block * y_block * memory_unit * (1 + dummy_bands)
    ram_blocks = int(single_block_size / available_ram)
    if ram_blocks == 0:
        ram_blocks = 1
    cfg.logger.log.debug('ram_blocks: %s' % str(ram_blocks))
    block_size_x = x_block
    list_range_x = list(range(0, x_block, block_size_x))
    block_size_y = int(y_block / ram_blocks)
    list_range_y = list(range(0, y_block, block_size_y))
    tot_blocks = len(list_range_x) * len(list_range_y)
    return (block_size_x, block_size_y, list_range_x, list_range_y,
            tot_blocks)


# table join
def table_join(
        table_1_parameters, table_2_parameters, nodata_value, join_type,
        features, output_names, process_parameters
):
    # get parameters
    (table1, field1_name, table1_names, table1_dtypes, table1_features,
     table1_features_index) = table_1_parameters
    (table2, field2_name, table2_names, table2_dtypes, table2_features,
     table2_features_index, table2_output_names,
     features_table_2_outer) = table_2_parameters
    process_id, cfg.temp, progress_queue, refresh_time = process_parameters
    proc_error = False
    cfg.logger = Log(directory=cfg.temp.dir, multiprocess=str(process_id))
    cfg.logger.log.debug('start')
    c = 0
    output = output_part = outer_part = None
    start_time = datetime.datetime.now()
    cfg.logger.log.debug('join_type: %s' % join_type)
    for feature in features:
        c += 1
        now_time = datetime.datetime.now()
        elapsed_time = (now_time - start_time).total_seconds()
        if process_id == 0 and elapsed_time > refresh_time:
            start_time = now_time
            progress_queue.put([c, len(features)], False)
        if join_type == 'left':
            # table 1 features
            output_part = table1[table1[field1_name] == feature]
            # table 2 row of unique features
            if feature in table2_features:
                row_table_2 = table2[table2_features_index[
                    np.where(table2_features == feature)]]
            else:
                row_table_2 = None
            # add table 2 fields and features
            for column_2 in range(len(table2_output_names)):
                if row_table_2 is None:
                    new_field = np.array(
                        [nodata_value] * output_part.shape[0],
                        dtype=table2_dtypes[column_2]
                    )
                else:
                    new_field = np.array(
                        list(row_table_2[table2_names[column_2]])
                        * output_part.shape[0],
                        dtype=table2_dtypes[column_2]
                    )
                output_part = tm.append_field(
                    output_part, table2_output_names[column_2],
                    new_field, table2_dtypes[column_2]
                )
        elif join_type == 'right':
            # table 2 features
            output_part = table2[table2[field2_name] == feature]
            # rename fields
            output_part = tm.redefine_matrix_columns(
                matrix=output_part, input_field_names=table2_names,
                output_field_names=table2_output_names,
                progress_message=False
            )
            # table 1 row of unique features
            if feature in table1_features:
                row_table_1 = table1[table1_features_index[
                    np.where(table1_features == feature)]]
            else:
                row_table_1 = None
            # add table 1 fields and features
            for column_1 in range(len(table1_names)):
                # get field1 name
                if table1_names[column_1] == field1_name:
                    new_field = np.array(
                        [feature] * output_part.shape[0],
                        dtype=table1_dtypes[column_1]
                    )
                elif row_table_1 is None:
                    new_field = np.array(
                        [nodata_value] * output_part.shape[0],
                        dtype=table1_dtypes[column_1]
                    )
                else:
                    new_field = np.array(
                        list(row_table_1[table1_names[column_1]])
                        * output_part.shape[0],
                        dtype=table1_dtypes[column_1]
                    )
                output_part = tm.append_field(
                    output_part, table1_names[column_1], new_field,
                    table1_dtypes[column_1]
                )
        elif join_type == 'inner':
            # table 1 features
            output_part = table1[table1[field1_name] == feature]
            # table 2 row of unique features
            row_table_2 = table2[table2_features_index[
                np.where(table2_features == feature)]]
            # add table 2 fields and features
            for column_2 in range(len(table2_output_names)):
                new_field = np.array(
                    list(row_table_2[table2_names[column_2]])
                    * output_part.shape[0], dtype=table2_dtypes[column_2]
                )
                output_part = tm.append_field(
                    output_part, table2_output_names[column_2],
                    new_field, table2_dtypes[column_2]
                )
        # full outer join
        elif join_type == 'outer':
            # table 1 features
            output_part = table1[table1[field1_name] == feature]
            # table 2 row of unique features
            if feature in table2_features:
                row_table_2 = table2[table2_features_index[
                    np.where(table2_features == feature)]]
            else:
                row_table_2 = None
            # add table 2 fields and features
            for column_2 in range(len(table2_output_names)):
                if row_table_2 is None:
                    new_field = np.array(
                        [nodata_value] * output_part.shape[0],
                        dtype=table2_dtypes[column_2]
                    )
                else:
                    new_field = np.array(
                        list(row_table_2[table2_names[column_2]])
                        * output_part.shape[0],
                        dtype=table2_dtypes[column_2]
                    )
                output_part = tm.append_field(
                    output_part, table2_output_names[column_2],
                    new_field, table2_dtypes[column_2]
                )
        if output is None:
            output = output_part
        else:
            output = rfn.stack_arrays(
                (output, output_part), asrecarray=True, usemask=False
            )
    # complete outer join with remaining features
    if (join_type == 'outer' and process_id == 0
            and len(features_table_2_outer) > 0):
        # iterate main features
        for feature in features_table_2_outer:
            # table 2 features
            outer_part = table2[table2[field2_name] == feature]
            # rename fields
            outer_part = tm.redefine_matrix_columns(
                matrix=outer_part, input_field_names=table2_names,
                output_field_names=table2_output_names,
                progress_message=False
            )
            # add table 1 fields and features
            for column_1 in range(len(table1_names)):
                # get field1 name
                if table1_names[column_1] == field1_name:
                    new_field = np.array(
                        [feature] * outer_part.shape[0],
                        dtype=table1_dtypes[column_1]
                    )
                else:
                    new_field = np.array(
                        [nodata_value] * outer_part.shape[0],
                        dtype=table1_dtypes[column_1]
                    )
                outer_part = tm.append_field(
                    outer_part, table1_names[column_1], new_field,
                    table1_dtypes[column_1]
                )
        if output is None:
            output = outer_part
        else:
            output = rfn.stack_arrays(
                (output, outer_part), asrecarray=True, usemask=False
            )
    # arrange fields
    output = tm.redefine_matrix_columns(
        matrix=output, input_field_names=output_names,
        output_field_names=output_names, progress_message=False
    )
    cfg.logger.log.debug('end')
    logger = cfg.logger.stream.getvalue()
    return output, None, proc_error, logger


# gdal translate processor
def gdal_translate(
        input_file=None, output=None, option_string=None,
        process_parameters=None
):
    process_id = process_parameters[0]
    cfg.temp = process_parameters[1]
    memory = process_parameters[2]
    gdal_path = process_parameters[3]
    progress_queue = process_parameters[4]
    cfg.logger = Log(directory=cfg.temp.dir, multiprocess='0')
    cfg.logger.log.debug('start')
    if gdal_path is not None:
        for d in gdal_path.split(';'):
            try:
                os.add_dll_directory(d)
                cfg.gdal_path = d
            except Exception as err:
                str(err)
    from osgeo import gdal
    # GDAL config
    try:
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(memory))
        gdal.SetConfigOption('VSI_CACHE', 'FALSE')
        gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
    except Exception as err:
        str(err)
    try:
        cfg.logger.log.debug('option_string: %s' % option_string)
        if int(process_id) == 0:
            progress_gdal = (lambda percentage, m, c: progress_queue.put(
                100 if int(percentage * 100) > 100 else int(percentage * 100),
                False
            ))
        else:
            progress_gdal = None
        to = gdal.TranslateOptions(
            gdal.ParseCommandLine(option_string), callback=progress_gdal
        )
        gdal.Translate(output, input_file, options=to)
    except Exception as err:
        cfg.logger.log.error(str(err))
    cfg.logger.log.debug('end; output: %s' % output)
    logger = cfg.logger.stream.getvalue()
    return False, output, False, logger


# gdal warp processor
def gdal_warp(
        input_file=None, output=None, option_string=None,
        process_parameters=None
):
    process_id = process_parameters[0]
    cfg.temp = process_parameters[1]
    gdal_path = process_parameters[2]
    progress_queue = process_parameters[3]
    memory = process_parameters[4]
    cfg.logger = Log(directory=cfg.temp.dir, multiprocess='0')
    cfg.logger.log.debug('start')
    if gdal_path is not None:
        for d in gdal_path.split(';'):
            try:
                os.add_dll_directory(d)
                cfg.gdal_path = d
            except Exception as err:
                str(err)
    from osgeo import gdal
    # GDAL config
    try:
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(memory))
        gdal.SetConfigOption('VSI_CACHE', 'FALSE')
        gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
    except Exception as err:
        str(err)
    try:
        cfg.logger.log.debug('option_string: %s' % option_string)
        if int(process_id) == 0:
            progress_gdal = (lambda percentage, m, c: progress_queue.put(
                100 if int(percentage * 100) > 100 else int(percentage * 100),
                False
            ))
        else:
            progress_gdal = None
        to = gdal.WarpOptions(
            gdal.ParseCommandLine(option_string), callback=progress_gdal
        )
        gdal.Warp(output, input_file, options=to)
    except Exception as err:
        cfg.logger.log.error(str(err))
    cfg.logger.log.debug('end; output: %s' % output)
    logger = cfg.logger.stream.getvalue()
    return False, output, False, logger


# convert raster to vector
def raster_to_vector_process(
        raster_path, output_vector_path, field_name=False,
        process_parameters=None,
        connected=4
):
    # process parameters
    process_id = str(process_parameters[0])
    cfg.temp = process_parameters[1]
    gdal_path = process_parameters[2]
    progress_queue = process_parameters[3]
    memory = process_parameters[4]
    if gdal_path is not None:
        for d in gdal_path.split(';'):
            try:
                os.add_dll_directory(d)
                cfg.gdal_path = d
            except Exception as err:
                str(err)
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
    from pathlib import Path
    cfg.logger = Log(directory=cfg.temp.dir, multiprocess=str(process_id))
    cfg.logger.log.debug('start')
    # GDAL config
    try:
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(memory))
        gdal.SetConfigOption('VSI_CACHE', 'FALSE')
        gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
    except Exception as err:
        str(err)
    # open input with GDAL
    try:
        _r_d = gdal.Open(raster_path, gdal.GA_ReadOnly)
        assert _r_d.RasterXSize
    except Exception as err:
        cfg.logger.log.error(str(err))
        logger = cfg.logger.stream.getvalue()
        return None, str(err), logger
    # create output vector
    d = ogr.GetDriverByName('GPKG')
    cfg.logger.log.debug('output_vector_path: %s' % str(output_vector_path))
    _output_source = d.CreateDataSource(output_vector_path)
    if _output_source is None:
        cfg.logger.log.error('error: %s' % str(output_vector_path))
        logger = cfg.logger.stream.getvalue()
        return None, 'error: %s' % str(output_vector_path), logger
    else:
        sr = osr.SpatialReference()
        cfg.logger.log.debug('sr: %s' % str(sr))
        sr.ImportFromWkt(_r_d.GetProjectionRef())
        p = Path(raster_path)
        name = p.stem
        _output_layer = _output_source.CreateLayer(
            name, sr, ogr.wkbMultiPolygon
        )
        if not field_name:
            field_name = 'DN'
        field_def = ogr.FieldDefn(field_name, ogr.OFTInteger)
        try:
            _output_layer.CreateField(field_def)
        except Exception as err:
            cfg.logger.log.error(str(err))
            logger = cfg.logger.stream.getvalue()
            return None, str(err), logger
        field_index = _output_layer.GetLayerDefn().GetFieldIndex(field_name)
        _raster_band = _r_d.GetRasterBand(1)
        if connected == 8:
            # multipolygon
            opt = ["8CONNECTED=8"]
        else:
            opt = []
        if int(process_id) == 0:
            progress_gdal = (lambda percentage, m, c: progress_queue.put(
                100 if int(percentage * 100) > 100 else int(percentage * 100),
                False
            ))
        else:
            progress_gdal = None
        # raster to polygon
        gdal.Polygonize(
            _raster_band, _raster_band.GetMaskBand(), _output_layer,
            field_index, opt, progress_gdal
        )
        # close raster and vector
        _raster_band = None
        _r_d = None
        _output_layer = None
        _output_source = None
        # open layer
        o_vector = ogr.Open(output_vector_path)
        o_layer = o_vector.GetLayer()
        o_name = o_layer.GetName()
        # get minimum Y
        sql = 'SELECT MIN(miny) FROM "rtree_%s_geom"' % o_name
        sql_output = o_vector.ExecuteSQL(sql, dialect='SQLITE')
        out_feature = sql_output.GetNextFeature()
        min_y = out_feature.GetField(0)
        # release sql results
        o_vector.ReleaseResultSet(sql_output)
        # get maximum Y
        sql = 'SELECT MAX(maxy) FROM "rtree_%s_geom"' % o_name
        sql_output = o_vector.ExecuteSQL(sql, dialect='SQLITE')
        out_feature = sql_output.GetNextFeature()
        max_y = out_feature.GetField(0)
        # release sql results
        o_vector.ReleaseResultSet(sql_output)
        cfg.logger.log.debug('end')
        logger = cfg.logger.stream.getvalue()
        return [output_vector_path, min_y, max_y], False, logger


# raster sieve
def raster_sieve_process(
        process_parameters=None, input_parameters=None, output_parameters=None
):
    # process parameters
    process_id = str(process_parameters[0])
    cfg.temp = process_parameters[1]
    gdal_path = process_parameters[2]
    progress_queue = process_parameters[3]
    memory = process_parameters[4]
    # input_raster parameters
    raster = input_parameters[0]
    sieve_size = input_parameters[1]
    connected = input_parameters[2]
    # output parameters
    output = output_parameters[0]
    data_type = output_parameters[1]
    compress = output_parameters[2]
    compress_format = output_parameters[3]
    output_no_data = output_parameters[4]
    if gdal_path is not None:
        for d in gdal_path.split(';'):
            try:
                os.add_dll_directory(d)
                cfg.gdal_path = d
            except Exception as err:
                str(err)
    from osgeo import gdal
    cfg.logger = Log(directory=cfg.temp.dir, multiprocess=str(process_id))
    cfg.logger.log.debug('start')
    # GDAL config
    try:
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(memory))
        gdal.SetConfigOption('VSI_CACHE', 'FALSE')
        gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
    except Exception as err:
        str(err)
    # open input with GDAL
    try:
        _r_d = gdal.Open(raster, gdal.GA_ReadOnly)
        cfg.logger.log.debug('raster_path: %s' % raster)
        assert _r_d.RasterXSize
    except Exception as err:
        cfg.logger.log.error(str(err))
        logger = cfg.logger.stream.getvalue()
        return None, str(err), logger
    # create output raster
    out_file = cfg.temp.temporary_raster_path(name_suffix=process_id,
                                              name_prefix=process_id)
    cfg.logger.log.debug('out_file: %s' % str(out_file))
    # start progress
    progress_queue.put(1, False)
    _raster_band = _r_d.GetRasterBand(1)
    offs = _raster_band.GetOffset()
    scl = _raster_band.GetScale()
    if output_no_data is None:
        # nodata value
        output_no_data = _raster_band.GetNoDataValue()
    raster_vector.create_raster_from_reference(
        raster, 1, [out_file], output_no_data, 'GTiff', data_type, compress,
        compress_format, scale=scl, offset=offs
    )
    # raster sieve
    if int(process_id) == 0:
        progress_gdal = (lambda percentage, m, c: progress_queue.put(
            int(percentage * 100), False
        ))
    else:
        progress_gdal = None
    _t_raster = gdal.Open(out_file, gdal.GA_Update)
    _t_band = _t_raster.GetRasterBand(1)
    mask = _raster_band.GetMaskBand()
    gdal.SieveFilter(
        _raster_band, mask, _t_band, sieve_size, connected,
        callback=progress_gdal
    )
    # close rasters
    _raster_band = None
    _r_d = None
    _t_band = None
    _t_raster = None
    if os.path.isfile(out_file):
        try:
            import shutil
            shutil.move(out_file, output)
        except Exception as err:
            cfg.logger.log.error(str(err))
    cfg.logger.log.debug('end')
    # end progress
    progress_queue.put(100, False)
    logger = cfg.logger.stream.getvalue()
    return [[out_file]], False, logger


# convert vector to raster based on the resolution of a raster
def vector_to_raster(
        process_parameters=None, input_parameters=None, output_parameters=None
):
    # process parameters
    process_id = str(process_parameters[0])
    cfg.temp = process_parameters[1]
    gdal_path = process_parameters[2]
    progress_queue = process_parameters[3]
    memory = process_parameters[4]
    # input_raster parameters
    vector_path = input_parameters[0]
    field_name = input_parameters[1]
    reference_raster_path = input_parameters[2]
    nodata_value = input_parameters[3]
    background_value = input_parameters[4]
    burn_values = input_parameters[5]
    x_y_size = input_parameters[6]
    all_touched = input_parameters[7]
    minimum_extent = input_parameters[8]
    # output parameters
    output_path = output_parameters[0]
    output_format = output_parameters[1]
    compress = output_parameters[2]
    compress_format = output_parameters[3]
    if background_value is None:
        background_value = 0
    if nodata_value is None:
        nodata_value = 0
    if output_format is None:
        output_format = 'GTiff'
    if gdal_path is not None:
        for d in gdal_path.split(';'):
            try:
                os.add_dll_directory(d)
                cfg.gdal_path = d
            except Exception as err:
                str(err)
    from osgeo import gdal, ogr
    cfg.logger = Log(directory=cfg.temp.dir, multiprocess=str(process_id))
    cfg.logger.log.debug('start')
    # GDAL config
    try:
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(memory))
        gdal.SetConfigOption('VSI_CACHE', 'FALSE')
        gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
    except Exception as err:
        str(err)
    # open input with GDAL
    cfg.logger.log.debug('vector_path: %s' % vector_path)
    try:
        _vector = ogr.Open(vector_path)
        _v_layer = _vector.GetLayer()
        # check projection
        proj = _v_layer.GetSpatialRef()
        crs = proj.ExportToWkt()
        crs = crs.replace(' ', '')
        if len(crs) == 0:
            crs = None
        vector_crs = crs
    except Exception as err:
        cfg.logger.log.error(str(err))
        logger = cfg.logger.stream.getvalue()
        return None, str(err), logger
    # create output file
    out_file = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix,
                                            name_prefix=process_id)
    cfg.logger.log.debug('out_file: %s' % str(out_file))
    (gt, reference_crs, unit, xy_count, nd, number_of_bands, block_size,
     scale_offset, data_type) = raster_vector.raster_info(
        reference_raster_path)
    orig_x = gt[0]
    orig_y = gt[3]
    cfg.logger.log.debug('orig_x, orig_y: %s,%s' % (orig_x, orig_y))
    if x_y_size is not None:
        x_size = x_y_size[0]
        y_size = x_y_size[1]
    else:
        x_size = gt[1]
        y_size = abs(gt[5])
    cfg.logger.log.debug('x_size, y_size: %s,%s' % (x_size, y_size))
    # number of x pixels
    grid_columns = int(round(xy_count[0] * gt[1] / x_size))
    # number of y pixels
    grid_rows = int(round(xy_count[1] * abs(gt[5]) / y_size))
    cfg.logger.log.debug('grid_columns, grid_rows: %s,%s'
                         % (grid_columns, grid_rows))
    # check crs
    same_crs = raster_vector.compare_crs(reference_crs, vector_crs)
    cfg.logger.log.debug('same_crs: %s' % str(same_crs))
    if not same_crs:
        cfg.logger.log.error('different crs')
        logger = cfg.logger.stream.getvalue()
        return None, 'different crs', logger
    # calculate minimum extent
    if minimum_extent:
        min_x, max_x, min_y, max_y = _v_layer.GetExtent()
        orig_x = orig_x + x_size * int(round((min_x - orig_x) / x_size))
        orig_y = orig_y + y_size * int(round((max_y - orig_y) / y_size))
        cfg.logger.log.debug('orig_x, orig_y: %s,%s' % (orig_x, orig_y))
        grid_columns = abs(int(round((max_x - min_x) / x_size)))
        grid_rows = abs(int(round((max_y - min_y) / y_size)))
        cfg.logger.log.debug('grid_columns, grid_rows: %s,%s'
                             % (grid_columns, grid_rows))
    driver = gdal.GetDriverByName(output_format)
    temporary_grid = cfg.temp.temporary_raster_path(extension=cfg.tif_suffix)
    # create raster _grid
    _grid = driver.Create(
        temporary_grid, grid_columns, grid_rows, 1, gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'BIGTIFF=YES']
    )
    if _grid is None:
        _grid = driver.Create(
            temporary_grid, grid_columns, grid_rows, 1, gdal.GDT_Int16,
            options=['COMPRESS=LZW', 'BIGTIFF=YES']
        )
    if _grid is None:
        cfg.logger.log.error('error output raster')
        logger = cfg.logger.stream.getvalue()
        return None, 'error grid', logger
    try:
        _grid.GetRasterBand(1)
    except Exception as err:
        cfg.logger.log.error(err)
        logger = cfg.logger.stream.getvalue()
        return None, str(err), logger
    # set raster projection from reference
    _grid.SetGeoTransform([orig_x, x_size, 0, orig_y, 0, -y_size])
    _grid.SetProjection(reference_crs)
    _grid = None
    # start progress
    progress_queue.put(1, False)
    if int(process_id) == 0:
        progress_gdal = (lambda percentage, m, c: progress_queue.put(
            int(percentage * 100), False
        ))
    else:
        progress_gdal = None
    # create output raster
    raster_vector.create_raster_from_reference(
        path=temporary_grid, band_number=1,
        output_raster_list=[out_file],
        nodata_value=nodata_value, driver='GTiff',
        gdal_format='Int32', compress=compress,
        compress_format=compress_format, constant_value=background_value
    )
    # convert reference layer to raster
    _output_raster = gdal.Open(out_file, gdal.GA_Update)
    if all_touched is False or all_touched is None:
        if burn_values is None:
            _o_c = gdal.RasterizeLayer(
                _output_raster, [1], _v_layer, options=[
                    'ATTRIBUTE=%s' % str(field_name), 'COMPRESS=DEFLATE',
                    'PREDICTOR=2', 'ZLEVEL=1'],
                callback=progress_gdal
            )

        else:
            _o_c = gdal.RasterizeLayer(
                _output_raster, [1], _v_layer, burn_values=[burn_values],
                options=['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=1'],
                callback=progress_gdal
            )
    else:
        if burn_values is None:
            _o_c = gdal.RasterizeLayer(
                _output_raster, [1], _v_layer,
                options=['ATTRIBUTE=%s' % str(field_name),
                         'all_touched=TRUE', 'COMPRESS=DEFLATE',
                         'PREDICTOR=2', 'ZLEVEL=1'], callback=progress_gdal
            )
        else:
            _o_c = gdal.RasterizeLayer(
                _output_raster, [1], _v_layer, burn_values=[burn_values],
                options=['all_touched=TRUE', 'COMPRESS=DEFLATE',
                         'PREDICTOR=2', 'ZLEVEL=1'], callback=progress_gdal
            )
    _output_raster = None
    if os.path.isfile(out_file):
        try:
            import shutil
            shutil.move(out_file, output_path)
        except Exception as err:
            cfg.logger.log.error(str(err))
    cfg.logger.log.debug('end')
    # end progress
    progress_queue.put(100, False)
    logger = cfg.logger.stream.getvalue()
    return [[output_path]], False, logger


# class to create a raster piece from raster sections
class RasterPiece(object):

    def __init__(
            self, section_list, x_min, y_min, x_max, y_max, x_size, y_size,
            y_min_no_boundary=None, y_max_no_boundary=None,
            y_size_boundary_top=None, y_size_boundary_bottom=None,
            y_size_no_boundary=None
    ):
        """
        :param section_list: list of sections
        :param x_min: left position
        :param y_min: top position
        :param x_max: right position
        :param y_max: bottom position
        :param x_size: piece x size
        :param y_size: piece y size
        :param y_min_no_boundary: top position without boundary
        :param y_max_no_boundary: bottom position without boundary
        :param y_size_boundary_top: top pixel buffer
        :param y_size_boundary_bottom: bottom pixel buffer
        :param y_size_no_boundary: piece size without buffer
        """
        self.sections = section_list
        # raster piece size
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x_size = x_size
        self.y_size = y_size
        # raster piece size with boundary
        self.y_min_no_boundary = y_min_no_boundary
        self.y_max_no_boundary = y_max_no_boundary
        # raster size without boundary
        self.y_size_no_boundary = y_size_no_boundary
        # buffer size
        self.y_size_boundary_top = y_size_boundary_top
        self.y_size_boundary_bottom = y_size_boundary_bottom

    # return piece y details as string
    def y_details(self):
        return ('y_min: %s; y_max: %s; y_size: %s; y_min_no_boundary: %s; '
                'y_max_no_boundary: %s; y_size_no_boundary: '
                '%s; y_size_boundary_top: %s; y_size_boundary_bottom: %s'
                % (str(self.y_min), str(self.y_max), str(self.y_size),
                   str(self.y_min_no_boundary),
                   str(self.y_max_no_boundary), str(self.y_size_no_boundary),
                   str(self.y_size_boundary_top),
                   str(self.y_size_boundary_bottom)))


# class to create raster sections
class RasterSection(object):

    def __init__(
            self, x_min, y_min, x_max, y_max, x_size, y_size,
            y_min_no_boundary=None, y_max_no_boundary=None,
            y_size_boundary_top=None, y_size_boundary_bottom=None,
            y_size_no_boundary=None
    ):
        """
        :param x_min: left position
        :param y_min: top position
        :param x_max: right position
        :param y_max: bottom position
        :param x_size: section x size
        :param y_size: section y size
        :param y_min_no_boundary: top position without boundary
        :param y_max_no_boundary: bottom position without boundary
        :param y_size_boundary_top: top pixel buffer
        :param y_size_boundary_bottom: bottom pixel buffer
        :param y_size_no_boundary: section size without buffer
        """
        # section size
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x_size = x_size
        self.y_size = y_size
        # section size with boundary
        self.y_min_no_boundary = y_min_no_boundary
        self.y_max_no_boundary = y_max_no_boundary
        # section size without boundary
        self.y_size_no_boundary = y_size_no_boundary
        # buffer size
        self.y_size_boundary_top = y_size_boundary_top
        self.y_size_boundary_bottom = y_size_boundary_bottom

    # return section y details as string
    def y_details(self):
        return ('y_min: %s; y_max: %s; y_size: %s; y_min_no_boundary: %s; '
                'y_max_no_boundary: %s; y_size_no_boundary: '
                '%s; y_size_boundary_top: %s; y_size_boundary_bottom: %s'
                % (str(self.y_min), str(self.y_max), str(self.y_size),
                   str(self.y_min_no_boundary),
                   str(self.y_max_no_boundary), str(self.y_size_no_boundary),
                   str(self.y_size_boundary_top),
                   str(self.y_size_boundary_bottom)))


# apply multiplicative and additive factors to array
def array_multiplicative_additive_factors(
        array, multiplicative_factor, additive_factor
):
    a = array * float(multiplicative_factor) + float(additive_factor)
    return a
