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

import gc
import multiprocessing
import os
import time
from typing import Union

import numpy as np
from numpy.lib import recfunctions as rfn

from remotior_sensus.core import configurations as cfg, processor
from remotior_sensus.util import files_directories, raster_vector

try:
    multiprocessing.set_start_method('spawn')
except Exception as error:
    str(error)


# class for multiprocess functions
class Multiprocess(object):

    def __init__(self, n_processes: int = None, multiprocess_module=None):
        if multiprocess_module is None:
            self.pool = multiprocessing.Pool(processes=n_processes)
            self.manager = multiprocessing.Manager()
        else:
            self.pool = multiprocess_module.Pool(processes=n_processes)
            self.manager = multiprocess_module.Manager()
        self.multiprocess_module = multiprocess_module
        self.n_processes = n_processes
        self.output = False

    # start multiprocess
    def start(self, n_processes, multiprocess_module=None):
        self.stop()
        cfg.multiprocess = self
        if multiprocess_module is None:
            self.pool = multiprocessing.Pool(processes=n_processes)
            self.manager = multiprocessing.Manager()
        else:
            self.pool = multiprocess_module.Pool(processes=n_processes)
            self.manager = multiprocess_module.Manager()
        self.multiprocess_module = multiprocess_module
        self.n_processes = n_processes
        self.output = False

    # stop multiprocess
    def stop(self):
        try:
            self.pool.close()
            self.pool.terminate()
        except Exception as err:
            str(err)

    # run multiprocess
    def run(
            self, raster_path, function=None, function_argument=None,
            function_variable=None, calculation_datatype=None,
            use_value_as_nodata=None, any_nodata_mask=True,
            output_raster_path=None, output_data_type=None,
            output_nodata_value=None, compress=None, compress_format='LZW',
            n_processes: int = None, available_ram: int = None,
            dummy_bands=1, output_band_number=1, boundary_size=None,
            unique_section=False, keep_output_array=False,
            keep_output_argument=False, delete_array=True, scale=None,
            offset=None, input_nodata_as_value=None,
            classification=False, classification_confidence=False,
            signature_raster=False, virtual_raster=False,
            multi_add_factors=None, separate_bands=False,
            progress_message=None, device=None, multiple_block=None,
            specific_output=None,
            min_progress=None, max_progress=None
    ):
        """
        :param device: processing device 'cpu' or 'cuda' if available.
        :param classification_confidence: if True, write also additional
            classification confidence rasters as output.
        :param signature_raster: if True, write additional rasters for each
            spectral signature as output.
        :param raster_path: input path.
        :param multi_add_factors: list of multiplicative and additive factors.
        :param virtual_raster: if True, create virtual raster output.
        :param offset: integer number of output offset.
        :param scale: integer number of output scale.
        :param delete_array: if True delete output array.
        :param keep_output_argument: if True keep output argument for post processing.
        :param keep_output_array: if True keep output array for post processing.
        :param unique_section: if True consider the whole raster as unique section.
        :param dummy_bands: integer number of dummy bands to be counted for
            calculating block size
        :param available_ram: integer value of RAM in MB.
        :param any_nodata_mask:  True to apply the nodata where any input is
            nodata, False to apply nodata where all inputs are nodata, None not apply nodata
        :param input_nodata_as_value: consider nodata as value in calculation
        :param output_data_type: string of data type for output raster such as Float32 or Int16
        :param output_band_number: number of bands of the output
        :param output_nodata_value: output nodata value
        :param compress_format: string of format of raster compression
        :param compress: True to compress the output raster or False not to compress
        :param boundary_size: integer number of pixels used to extend the
            boundary of calculations
        :param output_raster_path: list of output path strings
        :param calculation_datatype: datatype use during calculation
        :param function_variable: list of variables for function
        :param function_argument: arguments of function
        :param n_processes: number of parallel processes.
        :param function: function name
        :param specific_output: dictionary of values for specific output raster
        :param classification: if True, settings are defined for a classification output
        :param use_value_as_nodata: integer value as nodata in calculation
        :param separate_bands: if True, calculate a section for each raster range
        :param progress_message: progress message
        :param multiple_block: allows for setting block size as a multiple of the pixel count here defined
        :param min_progress: minimum progress value
        :param max_progress: maximum progress value
        """  # noqa: E501
        cfg.logger.log.debug('multiprocess: %s' % function)
        self.output = False
        process_result = {}
        process_output_files = {}
        # unit of memory
        memory_unit = cfg.memory_unit_array_12
        # calculation data type
        if calculation_datatype is None:
            calc_datatype = [np.float32]
        else:
            calc_datatype = [calculation_datatype]
        if (calculation_datatype == np.uint32
                or calculation_datatype == np.int32):
            memory_unit = cfg.memory_unit_array_8
        elif (calculation_datatype == np.uint16
              or calculation_datatype == np.int16):
            memory_unit = cfg.memory_unit_array_4
        if output_nodata_value is None:
            output_nodata_value = cfg.nodata_val
        if output_data_type is None:
            output_data_type = cfg.raster_data_type
        if n_processes is None:
            n_processes = self.n_processes
        elif n_processes > self.n_processes:
            self.start(self.n_processes, self.multiprocess_module)
        if compress is None:
            compress = cfg.raster_compression
        if compress_format is None:
            compress_format = cfg.raster_compression_format
        if device == 'cuda':
            n_processes = 1
        if min_progress is None:
            min_progress = 0
        if max_progress is None:
            max_progress = 100
        # reserve progress for writing output
        max_progress = round(max_progress * 0.9)
        if max_progress < min_progress:
            max_progress = min_progress
        max_progress_write = round(max_progress * 1.1)
        if max_progress_write > 100:
            max_progress_write = 100
        if progress_message is None:
            progress_message = 'processing'
        if available_ram is None:
            available_ram = cfg.available_ram
        if compress is None:
            compress = cfg.raster_compression
        if compress_format is None:
            compress_format = cfg.raster_compression_format
        # calculate block size
        block_size = _calculate_block_size(
            raster_path, n_processes, memory_unit, dummy_bands,
            available_ram=available_ram, multiple=multiple_block
        )
        if block_size is not False:
            (raster_x_size, raster_y_size, block_size_x, block_size_y,
             list_range_x, list_range_y, tot_blocks,
             number_of_bands) = block_size
        else:
            cfg.logger.log.error('unable to get raster: %s' % raster_path)
            cfg.messages.error('unable to get raster: %s' % raster_path)
            return
        # compute raster pieces
        pieces = _compute_raster_pieces(
            raster_x_size, raster_y_size, block_size_x, block_size_y,
            list_range_y, n_processes, separate_bands=separate_bands,
            boundary_size=boundary_size, unique_section=unique_section,
            specific_output=specific_output
        )
        # progress queue
        p_mq = self.manager.Queue()
        results = []
        for p in range(len(pieces)):
            process_parameters = [p, cfg.temp, available_ram, cfg.gdal_path,
                                  p_mq, cfg.refresh_time, memory_unit]
            input_parameters = [[raster_path], calc_datatype, boundary_size,
                                pieces[p], [scale], [offset],
                                [use_value_as_nodata], None,
                                input_nodata_as_value, multi_add_factors,
                                dummy_bands, specific_output]
            output_parameters = [[output_raster_path], [output_data_type],
                                 compress, compress_format, any_nodata_mask,
                                 [output_nodata_value], [output_band_number],
                                 keep_output_array, keep_output_argument]
            c = self.pool.apply_async(
                processor.function_initiator,
                args=(process_parameters, input_parameters, output_parameters,
                      function, [function_argument], [function_variable],
                      False, classification, classification_confidence,
                      signature_raster)
            )
            results.append([c, p])
        while True:
            if cfg.action is True:
                # update progress
                p_r = []
                for r in results:
                    p_r.append(r[0].ready())
                if all(p_r):
                    break
                time.sleep(cfg.refresh_time)
                # progress message
                try:
                    p_m_qp = p_mq.get(False)
                    count_progress = int(p_m_qp[0])
                    length = int(p_m_qp[1])
                    progress = int(100 * count_progress / length)
                    cfg.progress.update(
                        message=progress_message, step=count_progress,
                        steps=length, minimum=min_progress,
                        maximum=max_progress, percentage=progress
                    )
                except Exception as err:
                    str(err)
                    cfg.progress.update(ping=True)
            else:
                cfg.logger.log.error('cancel multiprocess')
                cfg.messages.error('cancel multiprocess')
                gc.collect()
                self.stop()
                self.start(self.n_processes, self.multiprocess_module)
                return
        for r in results:
            res = r[0].get()
            if classification:
                process_result[r[1]] = res[0]
                process_output_files[r[1]] = res[1]
            else:
                process_result[r[1]] = res[0]
                process_output_files[r[1]] = res[1]
            if cfg.logger.level < 20:
                cfg.logger.log.debug(res[3])
            # error
            if res[2] is not False:
                cfg.logger.log.error('multiprocess: %s' % res[2])
                cfg.messages.error('multiprocess: %s' % res[2])
                gc.collect()
                return
        gc.collect()
        if classification:
            multiprocess_result = self._collect_classification_results(
                process_result, output_raster_path,
                function_argument=function_argument, scale=scale,
                offset=offset, virtual_raster=virtual_raster,
                output_nodata_value=output_nodata_value,
                output_data_type=output_data_type, compress=compress,
                compress_format=compress_format,
                delete_array=delete_array, n_processes=n_processes,
                available_ram=available_ram, min_progress=max_progress,
                max_progress=max_progress_write
            )
        else:
            multiprocess_result = self._collect_results(
                process_result, process_output_files, output_raster_path,
                scale=scale, offset=offset, virtual_raster=virtual_raster,
                output_nodata_value=output_nodata_value,
                output_data_type=output_data_type, compress=compress,
                compress_format=compress_format, delete_array=delete_array,
                n_processes=n_processes, available_ram=available_ram,
                min_progress=max_progress, max_progress=max_progress_write
            )
        cfg.progress.update(percentage=False)
        cfg.logger.log.debug('end')
        self.output = multiprocess_result

    # separated process for each input
    def run_separated(
            self, raster_path_list, function=None, function_argument=None,
            function_variable=None, calculation_datatype=None,
            use_value_as_nodata=None, any_nodata_mask=True,
            output_raster_list=None, output_data_type=None,
            output_nodata_value=None, compress=None, compress_format='LZW',
            n_processes: int = None, available_ram: int = None,
            output_band_number_list=None, boundary_size=None,
            dummy_bands=0,
            keep_output_array=False, keep_output_argument=False,
            scale=None, offset=None, input_nodata_as_value=None,
            multi_add_factors=None, progress_message=None, min_progress=None,
            max_progress=None
    ):
        """
        :param multi_add_factors: list of multiplicative and additive factors
        :param offset: list integer number of output offset
        :param scale: list of integer number of output scale
        :param keep_output_argument: if True keep output argument for
            post-processing
        :param dummy_bands: integer number of dummy bands to be counted for
            calculating block size
        :param keep_output_array: if True keep output array for post-processing
        :param available_ram: integer value of RAM in MB
        :param any_nodata_mask:  True to apply the nodata where any input is
            nodata, False to apply nodata where all
            inputs are nodata, None not apply nodata
        :param input_nodata_as_value: consider nodata as value in calculation
        :param output_data_type: list of data type string for output raster
            such as Float32 or Int16
        :param output_band_number_list: list of number of bands of the output
        :param output_nodata_value: nodata value of the output
        :param compress_format: string of format of raster compression
        :param compress: True to compress the output raster or False not to
            compress
        :param boundary_size: integer number of pixels used to extend the
            boundary of calculations
        :param output_raster_list: list of output path strings
        :param calculation_datatype: datatype use during calculation
        :param function_variable: list of variables for function
        :param function_argument: arguments of function
        :param n_processes: number of parallel processes.
        :param function: function name
        :param raster_path_list: input path
        :param use_value_as_nodata: list of integer values as nodata in
            calculation
        :param progress_message: progress message
        :param min_progress: minimum progress value
        :param max_progress: maximum progress value
        """
        cfg.logger.log.debug('multiprocess: %s' % function)
        self.output = False
        process_result = {}
        process_output_files = {}
        # calculation data type
        if calculation_datatype is None:
            calculation_datatype = [np.float32] * len(raster_path_list)
        if output_nodata_value is None:
            output_nodata_value = [cfg.nodata_val] * len(raster_path_list)
        if output_data_type is None:
            output_data_type = [cfg.raster_data_type] * len(raster_path_list)
        if n_processes is None:
            n_processes = self.n_processes
        elif n_processes > self.n_processes:
            self.start(self.n_processes, self.multiprocess_module)
        if n_processes > len(raster_path_list):
            n_processes = len(raster_path_list)
        if min_progress is None:
            min_progress = 0
        if max_progress is None:
            max_progress = 100
        # reserve progress for writing output
        max_progress = round(max_progress * 0.9)
        if max_progress < min_progress:
            max_progress = min_progress
        if progress_message is None:
            progress_message = 'processing'
        if available_ram is None:
            available_ram = cfg.available_ram
        if compress is None:
            compress = cfg.raster_compression
        if compress_format is None:
            compress_format = cfg.raster_compression_format
        # progress queue
        p_mq = self.manager.Queue()
        # calculate block size
        results = []
        # one process per raster
        if len(raster_path_list) <= n_processes:
            ranges = list(range(len(raster_path_list)))
        # n threads running 2 rasters and m running 1 raster
        # 2 * n + m = raster_number
        # n + m = n_processes
        elif len(raster_path_list) / n_processes < 2:
            n = len(raster_path_list) - n_processes
            ranges = list(range(0, n * 2, 2))
            ranges.extend(list(range(n * 2, len(raster_path_list))))
        # calculate rasters per process
        else:
            ranges = list(
                range(
                    0, len(raster_path_list),
                    int(len(raster_path_list) / n_processes)
                )
            )
        ranges.append(len(raster_path_list))
        for p in range(1, len(ranges)):
            raster_paths = raster_path_list[ranges[p - 1]: ranges[p]]
            if output_raster_list:
                output_list = output_raster_list[ranges[p - 1]: ranges[p]]
            else:
                output_list = [None]
            if output_band_number_list is None:
                output_band_number = [1] * len(raster_paths)
            else:
                output_band_number = output_band_number_list[
                                     ranges[p - 1]: ranges[p]]
            if scale:
                scl = scale[ranges[p - 1]: ranges[p]]
            else:
                scl = [None] * len(raster_paths)
            if offset:
                offs = offset[ranges[p - 1]: ranges[p]]
            else:
                offs = [None] * len(raster_paths)
            if use_value_as_nodata is None:
                use_value_as_nodata = [None] * len(raster_paths)
            process_parameters = [p, cfg.temp, available_ram / n_processes,
                                  cfg.gdal_path, p_mq, cfg.refresh_time]
            input_parameters = [raster_paths,
                                calculation_datatype[ranges[p - 1]: ranges[p]],
                                boundary_size, None, scl, offs,
                                use_value_as_nodata[ranges[p - 1]: ranges[p]],
                                None, input_nodata_as_value, multi_add_factors,
                                dummy_bands, None]
            output_parameters = [output_list,
                                 output_data_type[ranges[p - 1]: ranges[p]],
                                 compress, compress_format, any_nodata_mask,
                                 output_nodata_value[ranges[p - 1]: ranges[p]],
                                 output_band_number, keep_output_array,
                                 keep_output_argument]
            if function_argument:
                f_arg = function_argument[ranges[p - 1]: ranges[p]]
            else:
                f_arg = None
            if function_variable:
                f_var = function_variable[ranges[p - 1]: ranges[p]]
            else:
                f_var = None
            c = self.pool.apply_async(
                processor.function_initiator,
                args=(process_parameters, input_parameters, output_parameters,
                      function, f_arg, f_var, True, False, False)
            )
            results.append([c, p])
        while True:
            if cfg.action is True:
                # update progress
                p_r = []
                for r in results:
                    p_r.append(r[0].ready())
                if all(p_r):
                    break
                time.sleep(cfg.refresh_time)
                # progress message
                try:
                    p_m_qp = p_mq.get(False)
                    count_progress = int(p_m_qp[0])
                    length = int(p_m_qp[1])
                    progress = int(100 * count_progress / length)
                    cfg.progress.update(
                        message=progress_message, step=count_progress,
                        steps=length, minimum=min_progress,
                        maximum=max_progress, percentage=progress
                    )
                except Exception as err:
                    str(err)
                    cfg.progress.update(ping=True)
            else:
                cfg.logger.log.error('cancel multiprocess')
                cfg.messages.error('cancel multiprocess')
                gc.collect()
                self.stop()
                self.start(self.n_processes, self.multiprocess_module)
                return
        for r in results:
            res = r[0].get()
            process_result[r[1]] = res[0]
            process_output_files[r[1]] = res[1]
            cfg.logger.log.debug(res[3])
            # error
            if res[2] is not False:
                cfg.logger.log.error('multiprocess: %s' % res[2])
                cfg.messages.error('multiprocess: %s' % res[2])
                gc.collect()
                return
        gc.collect()
        multiprocess_dictionary = self._collect_results(
            process_result, process_output_files, None, scale=None,
            offset=None, output_nodata_value=None,
            output_data_type=output_data_type, compress=compress,
            compress_format=compress_format, n_processes=n_processes,
            available_ram=available_ram, min_progress=max_progress
        )
        cfg.progress.update(percentage=False)
        cfg.logger.log.debug('end')
        self.output = multiprocess_dictionary

    # collect multiprocess results
    def _collect_results(
            self, process_result, process_output_files, output_raster_path,
            scale=None, offset=None, virtual_raster=None,
            output_nodata_value=None, output_data_type=None, compress=False,
            compress_format='LZW', delete_array=None, n_processes=1,
            available_ram: int = None, min_progress=None, max_progress=None
    ):
        cfg.logger.log.debug('start')
        multiprocess_dictionary = {}
        # temporary raster output
        tmp_rast_list = []
        p = None
        # get parallel dictionary result
        for p in sorted(process_result):
            try:
                # following iteration
                multiprocess_dictionary[p].extend(process_result[p])
            except Exception as err:
                str(err)
                # first iteration
                multiprocess_dictionary[p] = process_result[p]
        # output files
        if output_raster_path is not None:
            try:
                len(process_output_files[p]) * len(process_output_files)
            except Exception as err:
                cfg.logger.log.error(str(err))
                gc.collect()
                cfg.messages.error(str(err))
                return False
            # collect temporary raster paths
            for r in process_output_files:
                for op in process_output_files[r]:
                    tmp_rast_list.append(op[0])
            # get unique rasters and sort by process name
            tmp_rast_list = sorted(list(set(tmp_rast_list)))
            cfg.logger.log.debug('tmp_rast_list: %s' % str(tmp_rast_list))
            if scale is not None:
                scl = scale
            else:
                scl = 1
            if offset is not None:
                offs = offset
            else:
                offs = 0
            if scale is not None or offset is not None:
                par_scale_offset = ' -a_scale %s -a_offset %s' % (
                    str(scl), str(offs))
            else:
                par_scale_offset = ''
            # virtual raster output
            if virtual_raster:
                tmp_list = []
                dir_path = files_directories.parent_directory(
                    output_raster_path
                )
                file_count = 1
                f_name = files_directories.file_name(output_raster_path, False)
                # move temporary files
                for tR in tmp_rast_list:
                    out_r = '%s/%s_%02d%s' % (
                        dir_path, f_name, file_count, cfg.tif_suffix)
                    files_directories.create_parent_directory(out_r)
                    file_count += 1
                    tmp_list.append(out_r)
                    files_directories.move_file(tR, out_r)
                # create virtual raster
                raster_vector.create_virtual_raster_2_mosaic(
                    input_raster_list=tmp_list, output=output_raster_path,
                    dst_nodata=output_nodata_value, relative_to_vrt=1,
                    data_type=output_data_type
                )
                # fix relative to vrt in xml
                raster_vector.force_relative_to_vrt(output_raster_path)
            # copy raster output
            else:
                vrt_file = cfg.temp.temporary_raster_path(
                    extension=cfg.vrt_suffix
                )
                try:
                    # create virtual raster
                    raster_vector.create_virtual_raster_2_mosaic(
                        input_raster_list=tmp_rast_list, output=vrt_file,
                        dst_nodata=output_nodata_value,
                        data_type=output_data_type
                    )
                    files_directories.create_parent_directory(
                        output_raster_path
                    )
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    cfg.messages.error(str(err))
                try:
                    # copy raster
                    self.gdal_copy_raster(
                        vrt_file, output_raster_path, 'GTiff', compress,
                        compress_format, additional_params='-ot %s%s' % (
                            str(output_data_type), par_scale_offset),
                        n_processes=n_processes, min_progress=min_progress,
                        max_progress=max_progress
                    )
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    cfg.messages.error(str(err))
                    try:
                        # try to create different virtual raster then copy
                        vrt_file = cfg.temp.temporary_raster_path(
                            extension=cfg.vrt_suffix
                        )
                        raster_vector.create_virtual_raster(
                            input_raster_list=tmp_rast_list, output=vrt_file,
                            nodata_value=output_nodata_value,
                            data_type=output_data_type
                        )
                        self.gdal_copy_raster(
                            vrt_file, output_raster_path, 'GTiff', compress,
                            compress_format, additional_params='-ot %s%s' % (
                                str(output_data_type), par_scale_offset),
                            available_ram=available_ram,
                            min_progress=min_progress,
                            max_progress=max_progress
                        )
                    except Exception as err:
                        cfg.logger.log.error(str(err))
                        gc.collect()
                        cfg.messages.error(str(err))
                        return False
        # delete temporary rasters
        if delete_array and not virtual_raster:
            for n in tmp_rast_list:
                try:
                    os.remove(n)
                except Exception as err:
                    str(err)
        gc.collect()
        cfg.logger.log.debug('end')
        return multiprocess_dictionary

    # collect classification results
    def _collect_classification_results(
            self, process_result, output_raster_path, function_argument=None,
            scale=None, offset=None, virtual_raster=None,
            output_nodata_value=None, output_data_type=None,
            compress=None, compress_format='LZW',
            delete_array=None, n_processes=1, available_ram: int = None,
            min_progress=None, max_progress=None
    ):
        cfg.logger.log.debug('start')
        # classification output
        output_classification_list = []
        # algorithm output
        output_algorithm_list = []
        # signatures output
        signature_raster_list = []
        # get parallel dictionary result
        for p in sorted(process_result):
            if len(output_classification_list) == 0:
                # first iteration
                output_classification_list = process_result[p][0][0]
            else:
                output_classification_list.extend(process_result[p][0][0])
            if len(output_algorithm_list) == 0:
                # first iteration
                output_algorithm_list = process_result[p][0][1]
            else:
                output_algorithm_list.extend(process_result[p][0][1])
            if len(signature_raster_list) == 0:
                # first iteration
                signature_raster_list = process_result[p][0][2]
            else:
                for s in process_result[p][0][2]:
                    if s in signature_raster_list:
                        signature_raster_list[s].extend(
                            process_result[p][0][2][s]
                        )
                    else:
                        signature_raster_list[s] = process_result[p][0][2][s]
        cfg.logger.log.debug(
            'output_classification_list: %s; output_algorithm_list: %s; '
            'signature_raster_list: %s'
            % (str(output_classification_list), str(output_algorithm_list),
               str(signature_raster_list))
        )
        # output files
        if output_raster_path is not None:
            signatures = function_argument[cfg.spectral_signatures_framework]
            # get selected signatures
            signatures_table = signatures.table[signatures.table.selected == 1]
            if scale is not None:
                scl = scale
            else:
                scl = 1
            if offset is not None:
                offs = offset
            else:
                offs = 0
            if scale is not None or offset is not None:
                par_scale_offset = ' -a_scale %s -a_offset %s' % (
                    str(scl), str(offs))
            else:
                par_scale_offset = ''
            # virtual raster output
            if virtual_raster:
                tmp_list = []
                dir_path = files_directories.parent_directory(
                    output_raster_path
                )
                f_count = 1
                f_name = files_directories.file_name(output_raster_path, False)
                # move temporary files
                for r in output_classification_list:
                    out_r = '%s/%s_%02d%s' % (
                        dir_path, f_name, f_count, cfg.tif_suffix)
                    files_directories.create_parent_directory(out_r)
                    f_count += 1
                    tmp_list.append(out_r)
                    files_directories.move_file(r, out_r)
                # create virtual raster
                raster_vector.create_virtual_raster_2_mosaic(
                    input_raster_list=tmp_list, output=output_raster_path,
                    dst_nodata=output_nodata_value, relative_to_vrt=1,
                    data_type=output_data_type
                )
                # fix relative to vrt in xml
                raster_vector.force_relative_to_vrt(output_raster_path)
                # algorithm raster
                if len(output_algorithm_list) > 0:
                    tmp_list = []
                    f_count = 1
                    # move temporary files
                    for r in output_algorithm_list:
                        out_r = '%s/%s_alg_%02d%s' % (
                            dir_path, f_name, f_count, cfg.tif_suffix)
                        f_count += 1
                        tmp_list.append(out_r)
                        files_directories.move_file(r, out_r)
                    # create virtual raster
                    raster_vector.create_virtual_raster_2_mosaic(
                        input_raster_list=tmp_list,
                        output='%s/%s_alg%s' % (
                            dir_path, f_name, cfg.vrt_suffix),
                        dst_nodata=output_nodata_value, relative_to_vrt=1,
                        data_type=output_data_type
                    )
                    # fix relative to vrt in xml
                    raster_vector.force_relative_to_vrt(
                        '%s/%s_alg%s' % (dir_path, f_name, cfg.vrt_suffix)
                    )
                # signature rasters
                if len(signature_raster_list) > 0:
                    for s in signature_raster_list:
                        tmp_list = []
                        sig_name = '%s_mc_%s_c_%s' % (
                            f_name, str(
                                signatures_table[
                                    signatures_table.signature_id ==
                                    s].macroclass_id[
                                    0]
                            ),
                            str(
                                signatures_table[
                                    signatures_table.signature_id ==
                                    s].class_id[
                                    0]
                            ))
                        # move temporary files
                        for r in signature_raster_list[s]:
                            out_r = '%s/%s_%02d%s' % (
                                dir_path, sig_name, f_count, cfg.tif_suffix)
                            f_count += 1
                            tmp_list.append(out_r)
                            files_directories.move_file(r, out_r)
                        # create virtual raster
                        raster_vector.create_virtual_raster_2_mosaic(
                            input_raster_list=tmp_list, output='%s/%s%s' % (
                                dir_path, sig_name, cfg.vrt_suffix),
                            dst_nodata=output_nodata_value, relative_to_vrt=1,
                            data_type=output_data_type
                        )
                        # fix relative to vrt in xml
                        raster_vector.force_relative_to_vrt(
                            '%s/%s%s' % (dir_path, sig_name, cfg.vrt_suffix)
                        )
            # copy raster output
            else:
                dir_path = files_directories.parent_directory(
                    output_raster_path
                )
                f_name = files_directories.file_name(output_raster_path, False)
                vrt_file = cfg.temp.temporary_raster_path(
                    extension=cfg.vrt_suffix
                )
                try:
                    # create virtual raster
                    raster_vector.create_virtual_raster_2_mosaic(
                        input_raster_list=output_classification_list,
                        output=vrt_file, dst_nodata=output_nodata_value,
                        data_type=output_data_type
                    )
                    files_directories.create_parent_directory(
                        output_raster_path
                    )
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    cfg.messages.error(str(err))
                try:
                    # copy raster
                    self.gdal_copy_raster(
                        vrt_file, output_raster_path, 'GTiff', compress,
                        compress_format, additional_params='-ot %s%s' % (
                            str(output_data_type), par_scale_offset),
                        n_processes=n_processes, min_progress=min_progress,
                        max_progress=max_progress
                    )
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    cfg.messages.error(str(err))
                    try:
                        # try to create different virtual raster then copy
                        vrt_file = cfg.temp.temporary_raster_path(
                            extension=cfg.vrt_suffix
                        )
                        raster_vector.create_virtual_raster(
                            input_raster_list=output_classification_list,
                            output=vrt_file, nodata_value=output_nodata_value,
                            data_type=output_data_type
                        )
                        self.gdal_copy_raster(
                            vrt_file, output_raster_path, 'GTiff', compress,
                            compress_format, additional_params='-ot %s%s' % (
                                str(output_data_type), par_scale_offset),
                            available_ram=available_ram,
                            min_progress=min_progress,
                            max_progress=max_progress
                        )
                    except Exception as err:
                        cfg.logger.log.error(str(err))
                        gc.collect()
                        cfg.messages.error(str(err))
                        return False
                # algorithm raster
                if len(output_algorithm_list) > 0:
                    vrt_file = cfg.temp.temporary_raster_path(
                        extension=cfg.vrt_suffix
                    )
                    try:
                        # create virtual raster
                        raster_vector.create_virtual_raster_2_mosaic(
                            input_raster_list=output_algorithm_list,
                            output=vrt_file, dst_nodata=output_nodata_value,
                            data_type=output_data_type
                        )
                    except Exception as err:
                        cfg.logger.log.error(str(err))
                        cfg.messages.error(str(err))
                    try:
                        # copy raster
                        self.gdal_copy_raster(
                            vrt_file,
                            '%s/%s_alg%s' % (dir_path, f_name, cfg.tif_suffix),
                            'GTiff', compress, compress_format,
                            additional_params='-ot %s%s' % (
                                str(output_data_type), par_scale_offset),
                            n_processes=n_processes, min_progress=min_progress,
                            max_progress=max_progress
                        )
                    except Exception as err:
                        cfg.logger.log.error(str(err))
                        cfg.messages.error(str(err))
                # signature rasters
                if len(signature_raster_list) > 0:
                    for s in signature_raster_list:
                        sig_name = '%s_mc_%s_c_%s' % (
                            f_name, str(
                                signatures_table[
                                    signatures_table.signature_id ==
                                    s].macroclass_id[
                                    0]
                            ),
                            str(
                                signatures_table[
                                    signatures_table.signature_id ==
                                    s].class_id[
                                    0]
                            ))
                        vrt_file = cfg.temp.temporary_raster_path(
                            extension=cfg.vrt_suffix
                        )
                        try:
                            # create virtual raster
                            raster_vector.create_virtual_raster_2_mosaic(
                                input_raster_list=signature_raster_list[s],
                                output=vrt_file,
                                dst_nodata=output_nodata_value,
                                data_type=output_data_type
                            )
                        except Exception as err:
                            cfg.logger.log.error(str(err))
                            cfg.messages.error(str(err))
                        try:
                            # copy raster
                            self.gdal_copy_raster(
                                vrt_file, '%s/%s%s' % (
                                    dir_path, sig_name, cfg.tif_suffix),
                                'GTiff', compress, compress_format,
                                additional_params='-ot %s%s' % (
                                    str(output_data_type), par_scale_offset),
                                n_processes=n_processes,
                                min_progress=min_progress,
                                max_progress=max_progress
                            )
                        except Exception as err:
                            cfg.logger.log.error(str(err))
                            cfg.messages.error(str(err))
        # delete temporary rasters
        if delete_array and not virtual_raster:
            output_classification_list.extend(output_algorithm_list)
            output_classification_list.extend(signature_raster_list)
            for n in output_classification_list:
                try:
                    os.remove(n)
                except Exception as err:
                    str(err)
        gc.collect()
        cfg.logger.log.debug('end')
        return output_raster_path

    # warp with GDAL
    def gdal_warping(
            self, input_raster, output, output_format='GTiff', s_srs=None,
            t_srs=None, resample_method=None, raster_data_type=None,
            compression=None, compress_format='DEFLATE', additional_params='',
            n_processes: int = None, available_ram: int = None,
            src_nodata=None, dst_nodata=None, min_progress=None,
            max_progress=None
    ):
        cfg.logger.log.debug('start')
        out_dir = files_directories.parent_directory(output)
        files_directories.create_directory(out_dir)
        if resample_method is None:
            resample_method = 'near'
        elif resample_method == 'sum':
            gdal_v = raster_vector.get_gdal_version()
            if float('%s.%s' % (gdal_v[0], gdal_v[1])) < 3.1:
                cfg.logger.log.error('Error GDAL version')
                return False
        if n_processes is None:
            n_processes = cfg.n_processes
        op = ' -r %s -co BIGTIFF=YES -multi -wo NUM_THREADS=%s' % (
            resample_method, str(n_processes))
        if compression is None:
            if cfg.raster_compression:
                op += ' -co COMPRESS=%s' % compress_format
        elif compression:
            op += ' -co COMPRESS=%s' % compress_format
        if s_srs is not None:
            op += ' -s_srs %s' % s_srs
        if t_srs is not None:
            op += ' -t_srs %s' % t_srs
        if raster_data_type is not None:
            op += ' -ot %s' % raster_data_type
        if src_nodata is not None:
            op += ' -srcnodata %s' % str(src_nodata)
        if dst_nodata is not None:
            op += ' -dstnodata %s' % str(dst_nodata)
        op += ' -of %s' % output_format
        if additional_params is not None:
            op = ' %s %s' % (additional_params, op)
        p = 0
        # progress queue
        p_mq = self.manager.Queue()
        if available_ram is None:
            available_ram = cfg.available_ram
        available_ram = str(int(available_ram) * 1000000)
        process_parameters = [p, cfg.temp, cfg.gdal_path, p_mq, available_ram]
        cfg.logger.log.debug(
            'process_parameters: %s' % str(process_parameters)
        )
        results = []
        c = self.pool.apply_async(
            processor.gdal_warp,
            args=(input_raster, output, op, process_parameters)
        )
        results.append([c, p])
        while True:
            if cfg.action is True:
                p_r = []
                for r in results:
                    p_r.append(r[0].ready())
                if all(p_r):
                    break
                time.sleep(cfg.refresh_time)
                # progress message
                try:
                    p_m_qp = p_mq.get(False)
                    progress = int(p_m_qp)
                    cfg.progress.update(
                        message='writing raster', step=progress, steps=100,
                        minimum=min_progress, maximum=max_progress,
                        percentage=progress
                    )
                except Exception as err:
                    str(err)
                    cfg.progress.update(ping=True)
            else:
                cfg.logger.log.error('cancel multiprocess')
                cfg.messages.error('cancel multiprocess')
                gc.collect()
                self.stop()
                self.start(self.n_processes, self.multiprocess_module)
                return
        for r in results:
            res = r[0].get()
            cfg.logger.log.debug('res[3]: %s' % str(res[3]))
            # error
            if res[2] is not False:
                cfg.logger.log.error(
                    'Error proc %s-%s' % (str(p), str(res[1]))
                )
                return False
        cfg.logger.log.debug('end; output: %s' % str(output))
        return output

    # create warped virtual raster
    def create_warped_vrt(
            self, raster_path, output_path, output_wkt=None,
            align_raster_path=None, same_extent=False,
            n_processes: int = None, src_nodata=None, dst_nodata=None,
            extra_params=None
    ):
        cfg.logger.log.debug('start')
        # calculate minimal extent
        if align_raster_path is not None:
            # align raster extent and pixel size
            try:
                (left_align, top_align, right_align, bottom_align, p_x_align,
                 p_y_align, output_wkt,
                 unit) = raster_vector.image_geotransformation(
                    align_raster_path
                )
                # check projections
                align_sys_ref = raster_vector.get_spatial_reference(output_wkt)
            except Exception as err:
                cfg.logger.log.error(str(err))
                return False
            # input_path raster extent and pixel size
            try:
                (left_input, top_input, right_input, bottom_input, p_x_input,
                 p_y_input, proj_input,
                 unit_input) = raster_vector.image_geotransformation(
                    raster_path
                )
                input_sys_ref = raster_vector.get_spatial_reference(proj_input)
                left_projected, top_projected = \
                    raster_vector.project_point_coordinates(
                        left_input, top_input, input_sys_ref, align_sys_ref
                    )
                right_projected, bottom_projected = \
                    raster_vector.project_point_coordinates(
                        right_input, bottom_input, input_sys_ref, align_sys_ref
                    )
            # Error latitude or longitude exceeded limits
            except Exception as err:
                cfg.logger.log.error(str(err))
                return False
            if not same_extent:
                # minimum extent
                if left_projected < left_align:
                    left_r = left_align - int(
                        2 + (left_align - left_projected) / p_x_align
                    ) * p_x_align
                else:
                    left_r = left_align + int(
                        (left_projected - left_align) / p_x_align - 2
                    ) * p_x_align
                if right_projected > right_align:
                    right_r = right_align + int(
                        2 + (right_projected - right_align) / p_x_align
                    ) * p_x_align
                else:
                    right_r = right_align - int(
                        (right_align - right_projected) / p_x_align - 2
                    ) * p_x_align
                if top_projected > top_align:
                    top_r = top_align + int(
                        2 + (top_projected - top_align) / p_y_align
                    ) * p_y_align
                else:
                    top_r = top_align - int(
                        (top_align - top_projected) / p_y_align - 2
                    ) * p_y_align
                if bottom_projected > bottom_align:
                    bottom_r = bottom_align + int(
                        (bottom_projected - bottom_align) / p_y_align - 2
                    ) * p_y_align
                else:
                    bottom_r = bottom_align - int(
                        2 + (bottom_align - bottom_projected) / p_y_align
                    ) * p_y_align
            else:
                left_r = left_align
                right_r = right_align
                top_r = top_align
                bottom_r = bottom_align
            extra_params = '-tr %s %s -te %s %s %s %s' % (
                str(p_x_align), str(p_y_align), str(left_r), str(bottom_r),
                str(right_r), str(top_r))
        self.gdal_warping(
            input_raster=raster_path, output=output_path, output_format='VRT',
            s_srs=None, t_srs=output_wkt, additional_params=extra_params,
            n_processes=n_processes, src_nodata=src_nodata,
            dst_nodata=dst_nodata
            )
        # workaround to gdalwarp issue ignoring scale and offset
        try:
            (gt, r_p, unit, xy_count, nd, number_of_bands, block_size,
             scale_offset, data_type) = raster_vector.raster_info(raster_path)
            scale = scale_offset[0]
            offset = scale_offset[1]
            if scale != 1 or offset != 0:
                o_r = raster_vector.open_raster(output_path)
                b_o = o_r.GetRasterBand(1)
                try:
                    b_o.SetScale(scale)
                except Exception as err:
                    str(err)
                try:
                    b_o.SetOffset(offset)
                except Exception as err:
                    str(err)
        except Exception as err:
            cfg.logger.log.error(str(err))
        cfg.logger.log.debug('end; output_path: %s' % output_path)
        return output_path

    # run multiprocess join table
    def join_tables_multiprocess(
            self, table1, table2, field1_name, field2_name, nodata_value=None,
            join_type=None, postfix=None, n_processes: int = None,
            progress_message=None, min_progress=None, max_progress=None
    ):
        """
        :param table1: input numpy table 1
        :param table2: input numpy table 2
        :param field1_name: input field table 1
        :param field2_name: input field table 2
        :param nodata_value: input nodata value
        :param join_type: join type
        :param postfix: postfix string
        :param n_processes: number of parallel processes.
        :param progress_message: progress message
        :param min_progress: minimum progress value
        :param max_progress: maximum progress value
        """
        cfg.logger.log.debug('multiprocess join')
        # progress queue
        p_mq = self.manager.Queue()
        self.output = False
        process_result = {}
        if n_processes is None:
            n_processes = self.n_processes
        elif n_processes > self.n_processes:
            self.start(self.n_processes, self.multiprocess_module)
        if min_progress is None:
            min_progress = 0
        if max_progress is None:
            max_progress = 100
        if progress_message is None:
            progress_message = 'processing'
        # output field names
        output_names = list(table1.dtype.names)
        # output field dtypes
        output_dtypes = []
        # fields table 1
        for name in table1.dtype.names:
            output_dtypes.append(table1[name].dtype)
        # table 1 field names
        table1_names = list(table1.dtype.names)
        # table 1 field dtypes
        table1_dtypes = []
        for name in table1.dtype.names:
            table1_dtypes.append(table1[name].dtype)
        # table 2 field names and dtypes
        table2_dtypes = []
        table2_names = []
        table2_output_names = []
        for name in table2.dtype.names:
            if field2_name != str(name):
                table2_names.append(str(name))
                # rename duplicate field names
                if str(name) not in table1.dtype.names:
                    output_names.append(str(name))
                    table2_output_names.append(str(name))
                else:
                    output_names.append('%s%s' % (str(name), postfix))
                    table2_output_names.append('%s%s' % (str(name), postfix))
                output_dtypes.append(table2[name].dtype)
                table2_dtypes.append(table2[name].dtype)
        # identify table features
        table1_features = table1_features_index = table2_features_index = None
        features_table_2_outer = None
        if join_type == 'left':
            # unique features
            table1_features = list(np.unique(table1[field1_name]))
            table_2_unique = np.unique(table2[field2_name], return_index=True)
            table2_features = table_2_unique[0]
            table2_features_index = table_2_unique[1]
            # main features
            features = table1_features
        elif join_type == 'right':
            # unique features
            table_1_unique = np.unique(table1[field1_name], return_index=True)
            table1_features = table_1_unique[0]
            table1_features_index = table_1_unique[1]
            table2_features = list(np.unique(table2[field2_name]))
            # main features
            features = table2_features
        elif join_type == 'inner':
            # find common features
            table1_unique = np.unique(table1[field1_name])
            table_2_unique = np.unique(table2[field2_name], return_index=True)
            table2_features = table_2_unique[0]
            table2_features_index = table_2_unique[1]
            features = list(np.intersect1d(table1_unique, table2_features))
        elif join_type == 'outer':
            # find all unique features
            table1_features = list(np.unique(table1[field1_name]))
            table_2_unique = np.unique(table2[field2_name], return_index=True)
            table2_features = table_2_unique[0]
            table2_features_index = table_2_unique[1]
            # main features as left join
            features = table1_features
            # features from table 2 not in table 1
            features_table_2_outer = list(
                np.setdiff1d(table2_features, table1_features)
            )
        else:
            self.output = False
            cfg.logger.log.error('join type not found')
            return
        features_count = len(features)
        # calculate process ranges
        if features_count < n_processes:
            n_processes = features_count
        try:
            sections_range = list(
                range(0, features_count, round(features_count / n_processes))
            )
        except Exception as err:
            str(err)
            sections_range = [0]
        sections_range.append(features_count)
        cfg.logger.log.debug(
            'features_count: %s; len(sections_range): %s'
            % (str(features_count), str(len(sections_range)))
        )
        ranges = []
        for index_range in range(1, len(sections_range)):
            ranges.append(
                features[sections_range[index_range - 1]:sections_range[
                    index_range]]
            )
        results = []
        for p in range(len(ranges)):
            feature_range = ranges[p]
            table_1_parameters = [table1, field1_name, table1_names,
                                  table1_dtypes, table1_features,
                                  table1_features_index]
            table_2_parameters = [table2, field2_name, table2_names,
                                  table2_dtypes, table2_features,
                                  table2_features_index, table2_output_names,
                                  features_table_2_outer]
            process_parameters = [p, cfg.temp, p_mq, cfg.refresh_time]
            c = self.pool.apply_async(
                processor.table_join,
                args=(table_1_parameters, table_2_parameters, nodata_value,
                      join_type, feature_range, output_names,
                      process_parameters)
            )
            results.append([c, p])
        while True:
            if cfg.action is True:
                # update progress
                p_r = []
                for r in results:
                    p_r.append(r[0].ready())
                if all(p_r):
                    break
                time.sleep(cfg.refresh_time)
                # progress message
                try:
                    p_m_qp = p_mq.get(False)
                    count_progress = int(p_m_qp[0])
                    length = int(p_m_qp[1])
                    progress = int(100 * count_progress / length)
                    cfg.progress.update(
                        message=progress_message, step=count_progress,
                        steps=length, minimum=min_progress,
                        maximum=max_progress, percentage=progress
                    )
                except Exception as err:
                    str(err)
                    cfg.progress.update(ping=True)
            else:
                cfg.logger.log.error('cancel multiprocess')
                cfg.messages.error('cancel multiprocess')
                gc.collect()
                self.stop()
                self.start(self.n_processes, self.multiprocess_module)
                return
        for r in results:
            res = r[0].get()
            process_result[r[1]] = res[0]
            cfg.logger.log.debug(res[3])
            # error
            if res[2] is not False:
                cfg.logger.log.error('multiprocess: %s' % str(res[2]))
                cfg.messages.error('multiprocess: %s' % str(res[2]))
                gc.collect()
                return
        gc.collect()
        output_table = _join_results(process_result)
        cfg.progress.update(percentage=False)
        cfg.logger.log.debug('end')
        self.output = output_table

    # convert raster to vector
    def multiprocess_raster_to_vector(
            self, raster_path, output_vector_path, field_name=None,
            n_processes: int = None,
            dissolve_output=True, min_progress=0, max_progress=100,
            available_ram: int = None
    ):
        max_progress_1 = int(max_progress / 3)
        cfg.logger.log.debug('start')
        process_result = []
        self.output = False
        if available_ram is None:
            available_ram = cfg.available_ram
        # raster blocks
        memory_unit = cfg.memory_unit_array_8
        (raster_x_size, raster_y_size, block_size_x, block_size_y,
         list_range_x, list_range_y, tot_blocks, number_of_bands) = \
            _calculate_block_size(
                raster_path, n_processes, memory_unit, 0,
                available_ram=available_ram
            )
        results = []
        # temporary raster output
        tmp_rast_list = []
        # calculate raster ranges
        for x in list_range_x:
            bs_x = block_size_x
            if x + bs_x > raster_x_size:
                bs_x = raster_x_size - x
            for y in list_range_y:
                bs_y = block_size_y
                if y + bs_y > raster_y_size:
                    bs_y = raster_y_size - y
                vrt_file = cfg.temp.temporary_raster_path(
                    extension=cfg.vrt_suffix
                )
                tmp_rast_list.append(vrt_file)
                # create virtual raster
                raster_vector.create_virtual_raster(
                    input_raster_list=[raster_path], output=vrt_file,
                    relative_extent_list=[x, y, bs_x, bs_y]
                )
        if field_name is None:
            field_name = 'DN'
        # progress queue
        p_mq = self.manager.Queue()
        # multiple parallel processes
        for p in range(len(tmp_rast_list)):
            vrt_path = tmp_rast_list[p]
            process_parameters = [p, cfg.temp, cfg.gdal_path, p_mq, int(
                int(available_ram) * 1000000 / len(tmp_rast_list)
            )]
            t_vector = cfg.temp.temporary_raster_path(
                extension=cfg.gpkg_suffix
            )
            c = self.pool.apply_async(
                processor.raster_to_vector_process,
                args=(vrt_path, t_vector, field_name, process_parameters)
            )
            results.append([c, p])
        while True:
            if cfg.action is True:
                # update progress
                p_r = []
                for r in results:
                    p_r.append(r[0].ready())
                if all(p_r):
                    break
                time.sleep(cfg.refresh_time)
                # progress message
                try:
                    progress = round(p_mq.get(False))
                    step = round(
                        min_progress + progress * (
                                max_progress_1 - min_progress) / 100
                    )
                    cfg.progress.update(
                        message='processing to vector', step=step,
                        percentage=progress
                    )
                except Exception as err:
                    str(err)
                    cfg.progress.update(ping=True)
            else:
                cfg.logger.log.error('cancel multiprocess')
                cfg.messages.error('cancel multiprocess')
                gc.collect()
                self.stop()
                self.start(self.n_processes, self.multiprocess_module)
                return
        # get results
        for r in results:
            res = r[0].get()
            process_result.append(res[0])
            cfg.logger.log.debug(res[2])
            # error
            if res[1] is not False:
                cfg.logger.log.error('error multiprocess: %s' % str(res[1]))
                gc.collect()
                return
        gc.collect()
        cfg.progress.update(
            message='merging vectors', step=max_progress_1, percentage=False
        )
        # merge layers to new layer
        if len(tmp_rast_list) > 1:
            # input layers
            input_layers_list = []
            # y coordinates of polygons on borders
            y_list = []
            for i in range(len(process_result)):
                vector_path, min_y, max_y = process_result[i]
                input_layers_list.append(vector_path)
                if i > 0:
                    if max_y is not None:
                        y_list.append(str(max_y))
                if i < len(tmp_rast_list) - 1:
                    if min_y is not None:
                        y_list.append(str(min_y))
            if dissolve_output:
                # merged vector
                t_vector = cfg.temp.temporary_raster_path(
                    extension=cfg.gpkg_suffix
                )
            else:
                t_vector = output_vector_path
            merge = raster_vector.merge_all_layers(
                input_layers_list, t_vector, min_progress=max_progress_1,
                max_progress=max_progress_1 * 2,
                dissolve_output=dissolve_output
            )
            if dissolve_output:
                merge = raster_vector.merge_dissolve_layer(
                    t_vector, output_vector_path, field_name, y_list,
                    min_progress=max_progress_1 * 2, max_progress=max_progress
                )
        else:
            merge = tmp_rast_list[0]
        self.output = merge
        cfg.logger.log.debug('end')

    # convert raster sieve
    def multiprocess_raster_sieve(
            self, raster_path, n_processes: int = None, sieve_size=None,
            connected=None, output_nodata_value=None, output=None,
            output_data_type=None, compress=None, compress_format=None,
            available_ram: int = None, min_progress=0, max_progress=100
    ):
        cfg.logger.log.debug('start')
        if compress_format is None:
            compress_format = 'LZW'
        # progress queue
        p_mq = self.manager.Queue()
        results = []
        self.output = False
        if available_ram is None:
            available_ram = cfg.available_ram
        available_ram = int(available_ram / (n_processes * 2))
        # temporary raster output
        process_result = {}
        process_output_files = {}
        p = 0
        process_parameters = [p, cfg.temp, cfg.gdal_path, p_mq, available_ram]
        input_parameters = [raster_path, sieve_size, connected]
        output_parameters = [output, output_data_type, compress,
                             compress_format, output_nodata_value]
        c = self.pool.apply_async(
            processor.raster_sieve_process,
            args=(process_parameters, input_parameters,
                  output_parameters)
        )
        results.append([c, p])
        while True:
            if cfg.action is True:
                # update progress
                p_r = []
                for r in results:
                    p_r.append(r[0].ready())
                if all(p_r):
                    break
                time.sleep(cfg.refresh_time)
                # progress message
                try:
                    p_m_qp = p_mq.get(False)
                    progress = int(p_m_qp)
                    cfg.progress.update(
                        step=progress, steps=100, minimum=min_progress,
                        maximum=max_progress, percentage=progress
                    )
                except Exception as err:
                    str(err)
                    cfg.progress.update(ping=True)
            else:
                cfg.logger.log.error('cancel multiprocess')
                cfg.messages.error('cancel multiprocess')
                gc.collect()
                self.stop()
                self.start(self.n_processes, self.multiprocess_module)
                return
        for r in results:
            res = r[0].get()
            process_result[r[1]] = res[0]
            process_output_files[r[1]] = res[0]
            cfg.logger.log.debug(res[2])
            # error
            if res[1] is not False:
                cfg.logger.log.error('error multiprocess: %s' % str(res[1]))
                gc.collect()
                return
        gc.collect()
        cfg.progress.update(percentage=False)
        self.output = output
        cfg.logger.log.debug('end')

    # convert vector to raster
    def multiprocess_vector_to_raster(
            self, vector_path, field_name=None,
            output_path=None, reference_raster_path=None,
            output_format=None, nodata_value=None,
            background_value=None, burn_values=None, minimum_extent=None,
            x_y_size=None, all_touched=None,
            compress=None, compress_format=None,
            available_ram: int = None, min_progress=0, max_progress=100
    ):
        cfg.logger.log.debug('start')
        if compress_format is None:
            compress_format = 'LZW'
        self.output = False
        if available_ram is None:
            available_ram = cfg.available_ram
        # temporary raster output
        process_result = {}
        process_output_files = {}
        # progress queue
        p_mq = self.manager.Queue()
        results = []
        p = 0
        process_parameters = [p, cfg.temp, cfg.gdal_path, p_mq,
                              available_ram]
        input_parameters = [vector_path, field_name, reference_raster_path,
                            nodata_value, background_value, burn_values,
                            x_y_size, all_touched, minimum_extent]
        output_parameters = [output_path, output_format, compress,
                             compress_format]
        c = self.pool.apply_async(
            processor.vector_to_raster,
            args=(process_parameters, input_parameters, output_parameters)
        )
        results.append([c, p])
        while True:
            if cfg.action is True:
                # update progress
                p_r = []
                for r in results:
                    p_r.append(r[0].ready())
                if all(p_r):
                    break
                time.sleep(cfg.refresh_time)
                # progress message
                try:
                    p_m_qp = p_mq.get(False)
                    progress = int(p_m_qp)
                    cfg.progress.update(
                        step=progress, steps=100, minimum=min_progress,
                        maximum=max_progress, percentage=progress
                    )
                except Exception as err:
                    str(err)
                    cfg.progress.update(ping=True)
            else:
                cfg.logger.log.error('cancel multiprocess')
                cfg.messages.error('cancel multiprocess')
                gc.collect()
                self.stop()
                self.start(self.n_processes, self.multiprocess_module)
                return
        for r in results:
            res = r[0].get()
            process_result[r[1]] = res[0]
            process_output_files[r[1]] = res[0]
            cfg.logger.log.debug(res[2])
            # error
            if res[1] is not False:
                cfg.logger.log.error('error multiprocess: %s' % str(res[1]))
                gc.collect()
                return
        gc.collect()
        cfg.progress.update(percentage=False)
        self.output = output_path
        cfg.logger.log.debug('end')

    # sum the output of multiprocess
    def multiprocess_sum_array(self, nodata=None):
        cfg.logger.log.debug('start')
        multiprocess_dictionary: Union[dict, bool] = self.output
        if not multiprocess_dictionary:
            cfg.logger.log.error('unable to process')
            return
        # calculate unique values and sum
        values = np.array([])
        sum_val = np.array([])
        for x in sorted(multiprocess_dictionary):
            for arr_x in multiprocess_dictionary[x]:
                try:
                    values = np.append(values, arr_x[1][0, ::].ravel())
                    sum_val = np.append(sum_val, arr_x[1][1, ::].ravel())
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    return
        unique = list(np.unique(values, return_counts=False))
        val_sum = []
        for v in unique:
            if v != nodata:
                val_sum.append([v, sum_val[values == v].sum()])
        # dictionary of values and sum
        dtype_list = [('new_val', 'int64'), ('sum', 'int64')]
        unique_val = np.rec.fromarrays(np.asarray(val_sum).T, dtype=dtype_list)
        cfg.logger.log.debug(
            'end; unique_val.shape: %s'
            % str(unique_val.shape)
            )
        self.output = unique_val

    # get dictionary of sums
    def get_dictionary_sum(self):
        cfg.logger.log.debug('start')
        multiprocess_dictionary: Union[dict, bool] = self.output
        if not multiprocess_dictionary:
            cfg.logger.log.error('unable to process')
            return
        _dict = {}
        # get dictionaries
        for x in sorted(multiprocess_dictionary):
            try:
                for ar in multiprocess_dictionary[x]:
                    for i in ar[1]:
                        # sum values if multiple processes
                        if i in _dict:
                            _dict[i] = _dict[i] + ar[1][i]
                        else:
                            _dict[i] = ar[1][i]
            except Exception as err:
                cfg.logger.log.error(str(err))
                return
        cfg.logger.log.debug('end; _dict: %s' % str(_dict))
        self.output = _dict

    # find minimum DN
    def find_minimum_dn(self):
        cfg.logger.log.debug('start')
        multiprocess_dictionary: Union[dict, bool] = self.output
        if not multiprocess_dictionary:
            cfg.logger.log.error('unable to process')
            return
        dn_minimum_list = []
        for x in sorted(multiprocess_dictionary):
            try:
                # each array is an input
                for ar in multiprocess_dictionary[x]:
                    values = ar[1][0, ::].tolist()
                    count = ar[1][1, ::]
                    count_list = count.tolist()
                    total = count.sum()
                    min_threshold = total * 0.0001
                    sum_v = 0
                    i = 0
                    for v in values:
                        sum_v = sum_v + count_list[i]
                        i += 1
                        if sum_v >= min_threshold:
                            dn_minimum_list.append(v)
                            break
            except Exception as err:
                cfg.logger.log.error(str(err))
                return
        cfg.logger.log.debug('end; dn_minimum_list: %s' % str(dn_minimum_list))
        self.output = dn_minimum_list

    # roi arrays from output of multiprocess
    def multiprocess_roi_arrays(self):
        cfg.logger.log.debug('start')
        multiprocess_dictionary: Union[dict, bool] = self.output
        if not multiprocess_dictionary:
            cfg.logger.log.error('unable to process')
            return
        array_values = {}
        # calculate values
        for x in sorted(multiprocess_dictionary):
            for arr_x in multiprocess_dictionary[x]:
                for s in arr_x[1]:
                    try:
                        if s in array_values:
                            array_values[s].append(arr_x[1][s])
                        else:
                            array_values[s] = [arr_x[1][s]]
                    except Exception as err:
                        cfg.logger.log.error(str(err))
                        return
        self.output = array_values
        cfg.logger.log.debug('end')

    # spectral signature from output of multiprocess
    def multiprocess_spectral_signature(self):
        cfg.logger.log.debug('start')
        multiprocess_dictionary: Union[dict, bool] = self.output
        if not multiprocess_dictionary:
            cfg.logger.log.error('unable to process')
            return
        mean_values = []
        std = []
        # calculate values
        for x in sorted(multiprocess_dictionary):
            for arr_x in multiprocess_dictionary[x]:
                try:
                    mean_values.append(arr_x[1][0])
                    std.append(arr_x[1][1])
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    return
        self.output = mean_values, std
        cfg.logger.log.debug('end')

    # unique values from output of multiprocess
    def multiprocess_unique_values(self):
        cfg.logger.log.debug('start')
        multiprocess_dictionary: Union[dict, bool] = self.output
        if not multiprocess_dictionary:
            cfg.logger.log.error('unable to process')
            return
        # calculate unique values
        values = None
        for x in sorted(multiprocess_dictionary):
            for arr_x in multiprocess_dictionary[x]:
                try:
                    if values is None:
                        values = arr_x[1]
                    else:
                        try:
                            values = np.vstack((values, arr_x[1]))
                        except Exception as err:
                            cfg.logger.log.error(str(err))
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    return
        cfg.logger.log.debug('len(values): %s' % str(len(values)))
        # adapted from Jaime answer at
        # https://stackoverflow.com/questions/16970982/find-unique-rows-in
        # -numpy-array
        try:
            b_values = values.view(
                np.dtype((np.void, values.dtype.itemsize * values.shape[1]))
            )
            ff, index_a = np.unique(
                b_values, return_index=True, return_counts=False
            )
            output = values[index_a].tolist()
            self.output = output
            cfg.logger.log.debug('end; len(output): %s' % str(len(output)))
        except Exception as err:
            cfg.logger.log.error(str(err))
            return

    # run scikit process
    def run_scikit(
            self, function, classifier_list=None,
            list_train_argument_dictionaries=None, n_processes=None,
            available_ram: int = None, min_progress=None, max_progress=None
    ):
        cfg.logger.log.debug('start')
        self.output = False
        if n_processes is None:
            n_processes = self.n_processes
        elif n_processes > self.n_processes:
            self.start(self.n_processes, self.multiprocess_module)
        if available_ram is None:
            available_ram = cfg.available_ram
        if min_progress is None:
            min_progress = 0
        if max_progress is None:
            max_progress = 100
        # divide argument dictionaries for every process
        len_list_of_argument_dictionaries = len(
            list_train_argument_dictionaries
        )
        if len_list_of_argument_dictionaries <= n_processes:
            ranges = list(range(len_list_of_argument_dictionaries))
        # n threads running 2 dictionaries and m running 1 dictionary
        # 2 * n + m = len_list_of_argument_dictionaries
        # n + m = n_processes
        elif len_list_of_argument_dictionaries / n_processes < 2:
            n = len_list_of_argument_dictionaries - n_processes
            ranges = list(range(0, n * 2, 2))
            ranges.extend(
                list(range(n * 2, len_list_of_argument_dictionaries))
            )
        # calculate dictionary per process
        else:
            ranges = list(
                range(
                    0, len_list_of_argument_dictionaries,
                    int(len_list_of_argument_dictionaries / n_processes)
                )
            )
        ranges.append(len_list_of_argument_dictionaries)
        argument_dict_list = []
        list_of_classifier_process = []
        len_of_argument_dict_list = []
        for i in range(1, len(ranges)):
            argument_dict_list.append(
                list_train_argument_dictionaries[ranges[i - 1]: ranges[i]]
            )
            list_of_classifier_process.append(
                classifier_list[ranges[i - 1]: ranges[i]]
            )
            len_of_argument_dict_list.append(
                len(list_train_argument_dictionaries[ranges[i - 1]: ranges[i]])
            )
        logger_argument_index = len_of_argument_dict_list.index(
            max(len_of_argument_dict_list)
        )
        # result list
        results = []
        # iterate over arg dict list
        for p in range(0, len(argument_dict_list)):
            cfg.logger.log.debug('process %s' % str(p))
            if p == logger_argument_index:
                log_process = True
            else:
                log_process = False
            process_parameters = [p, cfg.temp, available_ram, log_process,
                                  cfg.logger]
            c = self.pool.apply_async(
                function, args=(
                    process_parameters, list_of_classifier_process[p],
                    argument_dict_list[p])
            )
            results.append([c, p])
        # divide progress max for process length
        max_progress_part = int(max_progress / max(len_of_argument_dict_list))
        max_progress_process = max_progress_part
        old_progress = 0
        while True:
            if cfg.action is True:
                p_r = []
                for r in results:
                    p_r.append(r[0].ready())
                if all(p_r):
                    break
                time.sleep(cfg.refresh_time)
                # progress message
                try:
                    # read progress from file
                    with open('%s/scikit' % cfg.temp.dir, 'r') as f:
                        progress_line = f.readlines()[-1].split(' ')
                        progress = round(
                            int(progress_line[-3]) / int(
                                progress_line[-1].replace('\\n', '')
                            ) * 100
                        )
                    cfg.progress.update(
                        message='fitting', step=progress, steps=100,
                        minimum=min_progress,
                        maximum=max_progress_process, percentage=progress
                    )
                    # scale progress for next process
                    if progress < old_progress:
                        min_progress = int(max_progress_process)
                        max_progress_process += max_progress_part
                    old_progress = int(progress)
                except Exception as err:
                    str(err)
                    cfg.progress.update(ping=True)
            else:
                cfg.logger.log.error('cancel multiprocess')
                cfg.messages.error('cancel multiprocess')
                gc.collect()
                self.stop()
                self.start(self.n_processes, self.multiprocess_module)
                return
        # get results
        process_result = []
        for r in results:
            res = r[0].get()
            process_result.append(res[0])
            cfg.logger.log.debug(res[2])
            # error
            if res[1] is not False:
                cfg.logger.log.error('error multiprocess: %s' % str(res[1]))
                gc.collect()
                return
        gc.collect()
        self.output = process_result
        cfg.logger.log.debug('end; function: %s' % str(function))

    # run iterative process
    def run_iterative_process(
            self, function_list, argument_list, min_progress=None,
            max_progress=None, n_processes=None
    ):
        cfg.logger.log.debug('start')
        self.output = False
        if n_processes is None:
            n_processes = self.n_processes
        elif n_processes > self.n_processes:
            self.start(self.n_processes, self.multiprocess_module)
        if min_progress is None:
            min_progress = 0
        if max_progress is None:
            max_progress = 100
        # progress queue
        p_mq = self.manager.Queue()
        # divide argument dictionaries for every process
        len_list_of_argument_dictionaries = len(argument_list)
        if len_list_of_argument_dictionaries <= n_processes:
            ranges = list(range(len_list_of_argument_dictionaries))
        # n threads running 2 dictionaries and m running 1 dictionary
        # 2 * n + m = len_list_of_argument_dictionaries
        # n + m = n_processes
        elif len_list_of_argument_dictionaries / n_processes < 2:
            n = len_list_of_argument_dictionaries - n_processes
            ranges = list(range(0, n * 2, 2))
            ranges.extend(
                list(range(n * 2, len_list_of_argument_dictionaries))
            )
        # calculate dictionary per process
        else:
            ranges = list(
                range(
                    0, len_list_of_argument_dictionaries,
                    int(len_list_of_argument_dictionaries / n_processes)
                )
            )
        ranges.append(len_list_of_argument_dictionaries)
        argument_dict_list = []
        for i in range(1, len(ranges)):
            argument_dict_list.append(argument_list[ranges[i - 1]: ranges[i]])
        # result list
        results = []
        # iterate over arg dict list
        for p in range(0, len(argument_dict_list)):
            cfg.logger.log.debug('process %s' % str(p))
            if p == 0:
                progress_queue = p_mq
            else:
                progress_queue = None
            c = self.pool.apply_async(
                function_list[p], args=(
                    p, progress_queue, argument_dict_list[p], cfg.logger)
            )
            results.append([c, p])
        while True:
            if cfg.action is True:
                p_r = []
                for r in results:
                    p_r.append(r[0].ready())
                if all(p_r):
                    break
                time.sleep(cfg.refresh_time)
                # progress message
                try:
                    p_m_qp = p_mq.get(False)
                    count_progress = int(p_m_qp[0])
                    length = int(p_m_qp[1])
                    progress = int(100 * count_progress / length)
                    cfg.progress.update(
                        message='processing', step=progress, steps=100,
                        minimum=min_progress,
                        maximum=max_progress, percentage=progress
                    )
                except Exception as err:
                    str(err)
                    cfg.progress.update(ping=True)
            else:
                cfg.logger.log.error('cancel multiprocess')
                cfg.messages.error('cancel multiprocess')
                gc.collect()
                self.stop()
                self.start(self.n_processes, self.multiprocess_module)
                return
        # get results
        process_result = []
        for r in results:
            res = r[0].get()
            process_result.append(res[0])
            cfg.logger.log.debug(res[2])
            # error
            if res[1] is not False:
                cfg.logger.log.error('error multiprocess: %s' % str(res[1]))
                gc.collect()
                return
        gc.collect()
        self.output = process_result
        cfg.logger.log.debug('end; function_list: %s' % str(function_list))

    # copy raster with GDAL
    def gdal_copy_raster(
            self, input_raster, output, output_format='GTiff', compress=None,
            compress_format='DEFLATE',
            additional_params='', n_processes=1, available_ram: int = None,
            min_progress=None, max_progress=None
    ):
        cfg.logger.log.debug('start')
        # progress queue
        p_mq = self.manager.Queue()
        out_dir = files_directories.parent_directory(output)
        files_directories.create_directory(out_dir)
        op = ' -co BIGTIFF=YES -co NUM_THREADS=%s' % str(n_processes)
        if compress is None:
            compress = cfg.raster_compression
        if compress_format is None:
            compress_format = cfg.raster_compression_format
        if not compress:
            op += ' -of %s' % output_format
        else:
            op += ' -co COMPRESS=%s -of %s' % (
                compress_format, output_format)
        parameters = '%s %s' % (additional_params, op)
        p = 0
        if available_ram is None:
            available_ram = cfg.available_ram
        available_ram = str(int(available_ram) * 1000000)
        process_parameters = [p, cfg.temp, available_ram, cfg.gdal_path,
                              p_mq]
        cfg.logger.log.debug(str(process_parameters))
        results = []
        c = self.pool.apply_async(
            processor.gdal_translate, args=(
                input_raster, output, parameters, process_parameters)
        )
        results.append([c, p])
        while True:
            if cfg.action is True:
                p_r = []
                for r in results:
                    p_r.append(r[0].ready())
                if all(p_r):
                    break
                time.sleep(cfg.refresh_time)
                # progress message
                try:
                    p_m_qp = p_mq.get(False)
                    progress = round(p_m_qp)
                    cfg.progress.update(
                        message='writing raster', step=progress, steps=100,
                        minimum=min_progress, maximum=max_progress,
                        percentage=progress
                    )
                except Exception as err:
                    str(err)
                    cfg.progress.update(ping=True)
            else:
                cfg.logger.log.error('cancel multiprocess')
                cfg.messages.error('cancel multiprocess')
                gc.collect()
                self.stop()
                self.start(self.n_processes, self.multiprocess_module)
                return
        for r in results:
            res = r[0].get()
            # log
            cfg.logger.log.debug(res[3])
            # error
            if res[2] is not False:
                cfg.logger.log.error(
                    'Error proc %s-%s' % (str(p), str(res[1]))
                )
                return False
        cfg.logger.log.debug('end; output: %s' % str(output))
        return output


# calculate block size and pixel ranges
def _calculate_block_size(
        raster_path, n_processes, memory_unit, dummy_bands,
        available_ram: int = None, multiple: int = None
):
    cfg.logger.log.debug('start')
    info = raster_vector.raster_info(raster_path)
    if info is not False:
        (gt, crs, crs_unit, xy_count, nd, number_of_bands, block_size,
         scale_offset, data_type) = info
    else:
        cfg.logger.log.error(
            'unable to get raster info: %s' % str(raster_path)
        )
        return False
    r_x, r_y = xy_count
    # list of range pixels
    y_block = block_size[1]
    if y_block == r_y:
        y_block = block_size[0]
    if multiple is not None:
        multiple_block = (y_block // multiple) * multiple
        if multiple_block > 0:
            y_block = int(multiple_block)
        else:
            y_block = int(multiple)
    if y_block > r_y:
        y_block = r_y
    single_block_size = r_x * y_block * memory_unit * (
            number_of_bands + dummy_bands)
    if available_ram is None:
        available_ram = cfg.available_ram
    ram_blocks = int(available_ram / (single_block_size * n_processes))
    if ram_blocks == 0:
        ram_blocks = 1
    block_size_x = r_x
    block_size_y = ram_blocks * y_block
    if block_size_x > r_x:
        block_size_x = r_x
    list_range_x = list(range(0, r_x, block_size_x))
    if block_size_y > r_y:
        block_size_y = int(r_y / float(n_processes)) + 1
    list_range_y = list(range(0, r_y, block_size_y))
    if len(list_range_y) < n_processes:
        block_size_y = int(r_y / float(n_processes)) + 1
        list_range_y = list(range(0, r_y, block_size_y))
    tot_blocks = len(list_range_x) * len(list_range_y)
    cfg.logger.log.debug(
        'end; r_x: %s; r_y: %s; block_size_x: %s; block_size_y: %s; '
        'list_range_x: %s; list_range_y: %s; tot_blocks: %s; '
        'number_of_bands: %s' % (
            r_x, r_y, block_size_x, block_size_y, list_range_x, list_range_y,
            tot_blocks, number_of_bands)
    )
    return (r_x, r_y, block_size_x, block_size_y, list_range_x, list_range_y,
            tot_blocks, number_of_bands)


# compute a raster piece (portions of raster to be processed) for each
# process divided in sections along the axis y
def _compute_raster_pieces(
        raster_x_size, raster_y_size, block_size_x, block_size_y, list_range_y,
        n_processes, separate_bands=False, boundary_size=None,
        unique_section=False, specific_output=None
):
    """
    :param raster_x_size: number of raster columns
    :param raster_y_size: number of raster rows
    :param block_size_x: raster block size for columns
    :param block_size_y: raster block size for rows
    :param list_range_y: list of y pieces using block_size_y
    :param n_processes: number of parallel processes. to divide pieces
    :param separate_bands: if True, calculate a section for each raster piece
    :param boundary_size: pixel number to be added as boundary to each
    section along the axis y
    :param unique_section: if True, consider the whole raster as unique section
    """
    # list of pieces (one per process) where each range is based on sections
    pieces = []
    if specific_output is not None:
        specific_output['pieces'] = []
    if unique_section:
        sections = [processor.RasterSection(
            x_min=0, y_min=0, x_max=raster_x_size, y_max=raster_y_size,
            x_size=raster_x_size, y_size=raster_y_size
        )]
        pieces.append(
            processor.RasterPiece(
                section_list=sections, x_min=0, y_min=0, x_max=raster_x_size,
                y_max=raster_y_size, x_size=raster_x_size, y_size=raster_y_size
            )
        )
    else:
        # create a list of indices of y pieces for each process
        try:
            y_range_index_list = list(
                range(
                    0, len(list_range_y),
                    round(len(list_range_y) / n_processes)
                )
            )
        except Exception as err:
            str(err)
            y_range_index_list = [0]
        y_range_index_list.append(len(list_range_y))
        # iterate over range index list
        for y_range_index in range(1, len(y_range_index_list)):
            # range variables
            y_min_range = list_range_y[y_range_index_list[y_range_index - 1]]
            y_max_range = None
            y_size_range = 0
            y_size_no_boundary_range = 0
            y_min_no_boundary_range_list = []
            y_max_no_boundary_range_list = []
            y_size_boundary_top_range = None
            y_size_boundary_bottom_range = None
            # list of Sections per process
            sections_list = []
            if specific_output is not None:
                specific_output_sections = []
            else:
                specific_output_sections = None
            # iterate over process y pieces
            for process_y_range in range(
                    y_range_index_list[y_range_index - 1],
                    y_range_index_list[y_range_index]
            ):
                y = list_range_y[process_y_range]
                # multiple band processes
                if not separate_bands:
                    # without boundary
                    if boundary_size is None:
                        # section y size
                        y_size = block_size_y
                        # adapt to raster y size
                        if y + y_size > raster_y_size:
                            y_size = raster_y_size - y
                        y_max = y + y_size
                        sections_list.append(
                            processor.RasterSection(
                                x_min=0, y_min=y, x_max=block_size_x,
                                y_max=y_max, x_size=block_size_x, y_size=y_size
                            )
                        )
                        if specific_output is not None:
                            specific_output_sections.append(
                                processor.RasterSection(
                                    x_min=0,
                                    y_min=int(
                                        y * specific_output['resize_factor']),
                                    x_max=int(
                                        block_size_x
                                        * specific_output['resize_factor']),
                                    y_max=int(
                                        y_max
                                        * specific_output['resize_factor']),
                                    x_size=int(
                                        block_size_x
                                        * specific_output['resize_factor']),
                                    y_size=(int(
                                        y_max
                                        * specific_output['resize_factor'])
                                            - int(y * specific_output[
                                                'resize_factor'])),
                                )
                            )
                        # range variables
                        y_size_range += y_size
                        y_size_no_boundary_range = y_size_range
                        y_min_no_boundary_range_list.append(y)
                        y_max_no_boundary_range_list.append(y_max)
                        # keep the last one
                        y_max_range = y_max
                    # with y boundary
                    else:
                        y_min_no_boundary = y
                        # nominal boundary
                        y_size_boundary_top = boundary_size
                        y_size_boundary_bottom = boundary_size
                        # nominal section y size without boundary
                        y_size_no_boundary = block_size_y
                        # minimum y with boundary
                        y_min = y_min_no_boundary - y_size_boundary_top
                        if y_min < 0:
                            y_min = 0
                            y_size_boundary_top = y_min_no_boundary
                        # maximum y without boundary
                        y_max_no_boundary = (
                                y_min_no_boundary + y_size_no_boundary)
                        if y_max_no_boundary >= raster_y_size:
                            y_max_no_boundary = raster_y_size
                            y_size_boundary_bottom = 0
                            y_size_no_boundary = (
                                    y_max_no_boundary - y_min_no_boundary)
                        # maximum y with boundary
                        y_max = y_max_no_boundary + y_size_boundary_bottom
                        if y_max >= raster_y_size:
                            y_max = raster_y_size
                            y_size_boundary_bottom = (
                                    raster_y_size - y_max_no_boundary)
                        y_size = y_max - y_min
                        sections_list.append(
                            processor.RasterSection(
                                x_min=0, y_min=y_min, x_max=block_size_x,
                                y_max=y_max, x_size=block_size_x,
                                y_size=y_size,
                                y_min_no_boundary=y_min_no_boundary,
                                y_max_no_boundary=y_max_no_boundary,
                                y_size_boundary_top=y_size_boundary_top,
                                y_size_boundary_bottom=y_size_boundary_bottom,
                                y_size_no_boundary=y_size_no_boundary
                            )
                        )
                        if specific_output is not None:
                            specific_output_sections.append(
                                processor.RasterSection(
                                    x_min=0,
                                    y_min=int(
                                        y_min * specific_output[
                                            'resize_factor']),
                                    x_max=int(
                                        block_size_x * specific_output[
                                            'resize_factor']),
                                    y_max=int(y_max * specific_output[
                                        'resize_factor']),
                                    x_size=int(block_size_x * specific_output[
                                        'resize_factor']),
                                    y_size=(int(y_max * specific_output[
                                        'resize_factor'])
                                            - int(y_min * specific_output[
                                                'resize_factor'])),
                                    y_min_no_boundary=int(
                                        y_min_no_boundary * specific_output[
                                            'resize_factor']),
                                    y_max_no_boundary=int(
                                        y_max_no_boundary * specific_output[
                                            'resize_factor']),
                                    y_size_boundary_top=int(
                                        y_size_boundary_top * specific_output[
                                            'resize_factor']),
                                    y_size_boundary_bottom=int(
                                        y_size_boundary_bottom
                                        * specific_output['resize_factor']),
                                    y_size_no_boundary=int(
                                        y_size_no_boundary
                                        * specific_output['resize_factor'])
                                )
                            )

                        # range variables
                        y_size_range += y_size
                        y_size_no_boundary_range += y_size_no_boundary
                        y_min_no_boundary_range_list.append(y_min_no_boundary)
                        y_max_no_boundary_range_list.append(y_max_no_boundary)
                        # keep the first boundary
                        if y_size_boundary_top_range is None:
                            y_size_boundary_top_range = y_size_boundary_top
                        # keep the last one boundary
                        y_max_range = y_max
                        y_size_boundary_bottom_range = y_size_boundary_bottom
                # separate bands processes
                else:
                    # section y size
                    y_size = block_size_y
                    # adapt to raster y size
                    if y + y_size > raster_y_size:
                        y_size = raster_y_size - y
                    sections = processor.RasterSection(
                        x_min=0, y_min=y, x_max=block_size_x, y_max=y + y_size,
                        x_size=block_size_x, y_size=y_size
                    )
                    pieces.append(
                        processor.RasterPiece(
                            section_list=sections, x_min=0, y_min=y,
                            x_max=block_size_x, y_max=y + y_size,
                            x_size=block_size_x, y_size=y_size
                        )
                    )
                    y_min_no_boundary_range_list.append(y)
                    y_max_no_boundary_range_list.append(y + y_size)
            # process with sections
            if not separate_bands:
                pieces.append(
                    processor.RasterPiece(
                        section_list=sections_list, x_min=0, y_min=y_min_range,
                        x_max=block_size_x, y_max=y_max_range,
                        x_size=block_size_x, y_size=y_size_range,
                        y_min_no_boundary=min(y_min_no_boundary_range_list),
                        y_max_no_boundary=max(y_max_no_boundary_range_list),
                        y_size_boundary_top=y_size_boundary_top_range,
                        y_size_no_boundary=y_size_no_boundary_range,
                        y_size_boundary_bottom=y_size_boundary_bottom_range
                    )
                )
                if specific_output is not None:
                    if y_size_boundary_top_range is None:
                        y_size_boundary_top_range_s = None
                    else:
                        y_size_boundary_top_range_s = int(
                            y_size_boundary_top_range
                            * specific_output['resize_factor'])
                    if y_size_boundary_bottom_range is None:
                        y_size_boundary_bottom_range_s = None
                    else:
                        y_size_boundary_bottom_range_s = int(
                            y_size_boundary_bottom_range
                            * specific_output['resize_factor'])
                    specific_output['pieces'].append(
                        processor.RasterPiece(
                            section_list=specific_output_sections, x_min=0,
                            y_min=int(y_min_range
                                      * specific_output['resize_factor']),
                            x_max=int(block_size_x
                                      * specific_output['resize_factor']),
                            y_max=int(y_max_range
                                      * specific_output['resize_factor']),
                            x_size=int(block_size_x
                                       * specific_output['resize_factor']),
                            y_size=int(y_size_range
                                       * specific_output['resize_factor']),
                            y_min_no_boundary=int(
                                min(y_min_no_boundary_range_list)
                                * specific_output['resize_factor']),
                            y_max_no_boundary=int(
                                max(y_max_no_boundary_range_list)
                                * specific_output['resize_factor']),
                            y_size_boundary_top=y_size_boundary_top_range_s,
                            y_size_no_boundary=int(
                                y_size_no_boundary_range
                                * specific_output['resize_factor']),
                            y_size_boundary_bottom=(
                                y_size_boundary_bottom_range_s
                            )
                        )
                    )
    cfg.logger.log.debug('len(pieces): %s' % str(len(pieces)))
    return pieces


# join multiprocess table results
def _join_results(process_result):
    cfg.logger.log.debug('start')
    output = None
    # get parallel dictionary result
    for p in sorted(process_result):
        if output is None:
            output = process_result[p]
        else:
            output = rfn.stack_arrays(
                (output, process_result[p]), asrecarray=True, usemask=False
            )
    gc.collect()
    cfg.logger.log.debug('end')
    return output
