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
Raster label.

This tool allows for the label of raster pixels based on contiguous patches.
Any value not equal to 0 is considered as input, while 0 is considered as
background.
Contiguous pixels having different values (not equal to 0) are
considered as a patch.
The 4 pixel connection is considered for retrieving patches.
The output is a raster where each pixel value represents the pixel count
of the patch thereof.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> label = rs.raster_label(raster_path='file1.tif',
    ... output_path='output.tif')
"""

from typing import Optional

import numpy as np
from copy import deepcopy

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.core.processor_functions import raster_label_part
from remotior_sensus.util import files_directories, raster_vector, shared_tools
from remotior_sensus.core.processor_functions import raster_reclass


def raster_label(
        raster_path: str, output_path: Optional[str] = None,
        overwrite: Optional[bool] = False,
        extent_list: Optional[list] = None,
        # TODO implement multiple processes
        n_processes: Optional[int] = None,
        available_ram: Optional[int] = None,
        virtual_output: Optional[bool] = None,
        progress_message: Optional[bool] = True
) -> OutputManager:
    """Performs raster label.

    This tool allows for the label of raster pixels based on contiguous 
    patches.
    Please note that the argument n_processes is currently ignored, and only 1 
    process is used for performance reasons.

    Args:
        raster_path: path of raster used as input.
        output_path: string of output path.
        overwrite: if True, output overwrites existing files.
        extent_list: list of boundary coordinates left top right bottom.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        virtual_output: if True (and output_path is directory), save output
            as virtual raster of multiprocess parts.
        progress_message: if True then start progress message, if False does 
            not start the progress message (useful if launched from other 
            tools).

    Returns:
        Object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - path = output path

    Examples:
        Perform the raster label
            >>> # import Remotior Sensus and start the session
            >>> import remotior_sensus
            >>> rs = remotior_sensus.Session()
            >>> # start the process
            >>> label = rs.raster_label(raster_path='file1.tif',output_path='output.tif')
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=progress_message
    )
    cfg.logger.log.debug('raster_path: %s' % str(raster_path))
    # force 1 process because of performance issues when iterating patches
    n_processes = 1
    if n_processes is None:
        n_processes = cfg.n_processes
    if available_ram is None:
        available_ram = cfg.available_ram
    ram = int(available_ram / n_processes)
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
    # check output path
    out_path, vrt_r = files_directories.raster_output_path(
        output_path, virtual_output, overwrite=overwrite
    )
    # dummy bands for memory calculation
    dummy_bands = 4
    # process calculation
    cfg.multiprocess.run(
        raster_path=raster_path, function=raster_label_part,
        function_argument=[gt, crs], boundary_size=1,
        output_raster_path=out_path, calculation_datatype=np.int32,
        n_processes=n_processes, available_ram=available_ram,
        output_nodata_value=nd, compress=cfg.raster_compression,
        progress_message='extracting raster patches', skip_output=True,
        dummy_bands=dummy_bands, min_progress=1, max_progress=35
    )
    # get unique values
    y_dictionaries = []
    for d in cfg.multiprocess.output:
        for i in cfg.multiprocess.output[d]:
            y_dictionaries.append(i)
    sorted_y_dictionaries = sorted(y_dictionaries, key=lambda x: x['orig_y'],
                                   reverse=True)
    len_sorted_y_dictionaries = len(sorted_y_dictionaries)
    cfg.logger.log.debug(
        'len_sorted_y_dictionaries: %s' % len_sorted_y_dictionaries
    )
    # dictionary of section values to reclass [old_value, count, new_value]
    reclass_values = {}
    # dictionary of section values count
    reclass_count = {}
    increment_section_value = {}
    # arrays of shared values between sections
    stacked_arrays = None
    # arrays of reclassified shared values between sections
    b_stacked_arrays = None
    max_value = 0
    # iterate sections and create dictionaries
    for sec_y in range(len_sorted_y_dictionaries):
        cfg.progress.update(
            message='iterating sections', step=sec_y,
            steps=len_sorted_y_dictionaries, minimum=35, maximum=36,
            percentage=int(100 * sec_y / len_sorted_y_dictionaries)
        )
        increment_section_value[sec_y] = max_value
        # section y count of values
        unique_counts = sorted_y_dictionaries[sec_y]['unique_counts'].copy()
        if unique_counts.shape[0] > 0:
            reclass_values[sec_y] = np.column_stack(
                (unique_counts, unique_counts[:, 0] + max_value)
            )
            # increment patch value
            check_top = sorted_y_dictionaries[sec_y]['check_top']
            if check_top is None:
                check_top = np.array(0)
            check_bottom = sorted_y_dictionaries[sec_y]['check_bottom']
            if check_bottom is None:
                check_bottom = np.array(0)
            max_value += max(unique_counts[:, 0].max(), check_top.max(),
                             check_bottom.max())
            reclass_count[sec_y] = unique_counts
        else:
            reclass_values[sec_y] = None
            reclass_count[sec_y] = None
    # unique combinations from section top to bottom to reclass
    unique_combinations = []
    # iterate sections and create stacked array
    for sec_top in range(len_sorted_y_dictionaries - 1):
        cfg.progress.update(
            message='iterating sections', step=sec_top,
            steps=len_sorted_y_dictionaries, minimum=36, maximum=39,
            percentage=int(100 * sec_top / (len_sorted_y_dictionaries - 1))
        )
        # patch values to be merged from upper section
        top_check_bottom = sorted_y_dictionaries[sec_top]['check_bottom']
        sec_bottom = sec_top + 1
        # patch values to be merged from lower section
        bottom_check_top = sorted_y_dictionaries[sec_bottom]['check_top']
        # stack top bottom layers original values
        if stacked_arrays is None:
            stacked_arrays = np.vstack((top_check_bottom.ravel(),
                                        bottom_check_top.ravel()))
        else:
            stacked_arrays = np.vstack((stacked_arrays,
                                        top_check_bottom.ravel(),
                                        bottom_check_top.ravel()))
        # stack top bottom layers reclassified values
        if b_stacked_arrays is None:
            b_stacked_arrays = np.vstack((
                top_check_bottom.ravel() + increment_section_value[sec_top],
                bottom_check_top.ravel() + increment_section_value[sec_bottom]
            ))
        else:
            b_stacked_arrays = np.vstack((
                b_stacked_arrays,
                top_check_bottom.ravel() + increment_section_value[sec_top],
                bottom_check_top.ravel() + increment_section_value[sec_bottom]
            ))
        # remove 0 from combinations
        b_stacked_arrays[stacked_arrays == 0] = 0
        # unique combinations from section top to bottom to reclass
        unique_comb = np.unique(
            b_stacked_arrays[-2:, :], axis=1, return_counts=False
        )
        # remove 0 from combinations
        unique_combinations.append(
            unique_comb[:, np.all(unique_comb != 0, axis=0)].T
        )

    # iterate sections and get shared values between sections
    len_unique_combinations = len(unique_combinations)
    for sec_top in range(len_sorted_y_dictionaries - 1):
        unique_comb = unique_combinations[sec_top]
        # unique values in the top section
        unique_top = np.unique(unique_comb[:, 0])
        len_unique_top = len(unique_top)
        u_count = 0
        for u in unique_top:
            u_count += 1
            # get combination bottom values where combination == u
            shared_bottom_values = unique_comb[unique_comb[:, 0] == u][:, 1]
            shared_top_values = unique_comb[
                                    np.isin(unique_comb[:, 1],
                                            shared_bottom_values)][:, 0]
            # create comparison array for shared_bottom_values
            old_bottom_values = shared_bottom_values + 1
            # iterate until all shared values are found
            while not np.array_equal(shared_bottom_values, old_bottom_values):
                cfg.progress.update(
                    message='iterating patches', step=sec_top,
                    steps=len_sorted_y_dictionaries, minimum=40, maximum=60,
                    percentage=int(100 * u_count / len_unique_top))
                # get top values in case of shared bottom values
                shared_top_values = unique_comb[
                                        np.isin(unique_comb[:, 1],
                                                shared_bottom_values)][:, 0]
                old_bottom_values = shared_bottom_values.copy()
                # get shared bottom values
                shared_bottom_values = unique_comb[
                                           np.isin(unique_comb[:, 0],
                                                   shared_top_values)][:, 1]
            # replace values with shared top values
            b_stacked_arrays[
                np.isin(b_stacked_arrays, shared_top_values)] = min(
                shared_top_values)
            # replace values with shared bottom values
            b_stacked_arrays[
                np.isin(b_stacked_arrays, shared_bottom_values)] = min(
                shared_top_values)
            # replace values in unique_combinations
            for i, u_comb in enumerate(unique_combinations):
                cfg.progress.update(
                    message='iterating patches', step=sec_top,
                    steps=len_sorted_y_dictionaries, minimum=40, maximum=60,
                    percentage=int(100 * i / len_unique_combinations)
                )
                if i > sec_top:
                    u_comb[np.isin(u_comb, shared_top_values)] = min(
                        shared_top_values)
                    u_comb[np.isin(u_comb, shared_bottom_values)] = min(
                        shared_top_values)
                    unique_combinations[i] = u_comb

    # iterate sections and get reclass values
    target_sec = []
    for sec_y in range(len_sorted_y_dictionaries):
        cfg.progress.update(
            message='getting patch values', step=sec_y,
            steps=len_sorted_y_dictionaries, minimum=61, maximum=62,
            percentage=int(100 * sec_y / len_sorted_y_dictionaries)
        )
        if (sorted_y_dictionaries[sec_y]['check_top'] is None
                and sorted_y_dictionaries[sec_y]['check_bottom'] is None):
            a = np.array(0)
            b = np.array(0)
        elif sorted_y_dictionaries[sec_y]['check_top'] is None:
            a = stacked_arrays[0, :]
            b = b_stacked_arrays[0, :]
        elif sorted_y_dictionaries[sec_y]['check_bottom'] is None:
            a = stacked_arrays[-1, :]
            b = b_stacked_arrays[-1, :]
        else:
            a = stacked_arrays[(sec_y * 2 - 1):(sec_y * 2 + 1), :]
            b = b_stacked_arrays[(sec_y * 2 - 1):(sec_y * 2 + 1), :]
        # get reclass combinations
        reclass_combinations = set(zip(a.ravel().tolist(), b.ravel().tolist()))
        target_sec.append(np.unique(b, return_counts=False))
        # set reclassification values
        for c in reclass_combinations:
            if 0 not in c:
                reclass_values[sec_y][
                    reclass_values[sec_y][:, 0] == c[0], 2] = c[1]

    # get count of merged patches
    reclass_count_total = deepcopy(reclass_count)
    for sec_x in range(len_sorted_y_dictionaries):
        cfg.progress.update(
            message='getting patch count', step=sec_x,
            steps=len_sorted_y_dictionaries, minimum=63, maximum=85,
            percentage=int(100 * sec_x / len_sorted_y_dictionaries)
        )
        target_values = target_sec[sec_x]
        # get total count for reclassified values
        for t in target_values:
            tot_count = 0
            # count values in all sections
            for sec_y in range(len_sorted_y_dictionaries):
                if reclass_count[sec_y] is not None:
                    tot_count += reclass_count[sec_y][
                        reclass_values[sec_y][:, 2] == t, 1].sum()
            # apply sum in all sections
            for sec_z in range(len_sorted_y_dictionaries):
                if reclass_count[sec_z] is not None:
                    reclass_count_total[sec_z][
                        reclass_values[sec_z][:, 2] == t, 1] = tot_count

    # lists for multiprocess reclassification
    argument_list = []
    function_list = []
    output_raster_list = []
    for sec_z in range(len_sorted_y_dictionaries):
        # section raster to be reclassified
        section_raster = sorted_y_dictionaries[sec_z]['section_raster']
        tif_file = cfg.temp.temporary_raster_path(extension=cfg.tif_suffix)
        # reclassify raster based on count
        argument_list.append(
            {
                'input_raster': section_raster,
                'reclass_table': reclass_count_total[sec_z],
                'available_ram': ram,
                'output': tif_file,
                'gdal_path': cfg.gdal_path
            }
        )
        function_list.append(raster_reclass)
        output_raster_list.append(tif_file)
    cfg.multiprocess.run_iterative_process(
        function_list=function_list, argument_list=argument_list,
        min_progress=86, max_progress=90, message='merging patches'
    )
    if virtual_output:
        tmp_list = []
        dir_path = files_directories.parent_directory(out_path)
        file_count = 1
        f_name = files_directories.file_name(out_path, False)
        # move temporary files
        for tR in output_raster_list:
            out_r = '%s/%s_%02d%s' % (
                dir_path, f_name, file_count, cfg.tif_suffix)
            files_directories.create_parent_directory(out_r)
            file_count += 1
            tmp_list.append(out_r)
            files_directories.move_file(tR, out_r)
        # create virtual raster
        raster_vector.create_virtual_raster_2_mosaic(
            input_raster_list=tmp_list, output=out_path,
            dst_nodata=0, relative_to_vrt=1, data_type='UInt32'
        )
        # fix relative to vrt in xml
        raster_vector.force_relative_to_vrt(out_path)
        # copy raster output
    else:
        vrt_file = cfg.temp.temporary_raster_path(extension=cfg.vrt_suffix)
        try:
            # create virtual raster
            raster_vector.create_virtual_raster_2_mosaic(
                input_raster_list=output_raster_list, output=vrt_file,
                dst_nodata=0, data_type='UInt32'
            )
            files_directories.create_parent_directory(out_path)
            # copy raster
            cfg.multiprocess.gdal_copy_raster(
                vrt_file, out_path, 'GTiff', cfg.raster_compression,
                cfg.raster_compression_format,
                additional_params='-ot %s' % (str('UInt32')),
                n_processes=n_processes, min_progress=90, max_progress=100
            )
        except Exception as err:
            cfg.logger.log.error(err)
            cfg.messages.error(str(err))
            cfg.progress.update(failed=True)
            return OutputManager(check=False)
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; raster label: %s' % str(out_path))
    return OutputManager(path=out_path)
