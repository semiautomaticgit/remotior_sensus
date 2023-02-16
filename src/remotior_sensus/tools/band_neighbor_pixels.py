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
"""Band neighbor pixels.

This tool allows for the calculation of a function over neighbor pixels
defined by size (i.e. number of pixels) or a structure.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> neighbor = rs.band_neighbor_pixels(
    ... input_bands=['file1.tif', 'file2.tif'], output_path='directory_path',
    ... size=1, circular_structure=True, stat_name='Mean', prefix='neighbor_'
    ... )
"""

from remotior_sensus.core import configurations as cfg, messages
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.core.processor_functions import raster_neighbor
from remotior_sensus.util import shared_tools


def band_neighbor_pixels(
        input_bands, output_path=None, size=None, structure=None,
        circular_structure=True, stat_name=None, stat_percentile=None,
        output_data_type=None, virtual_output=None, prefix='',
        n_processes: int = None, available_ram: int = None,
        bandset_catalog=None
) -> OutputManager:
    """Performs band neighbor pixels.

    This tool calculates a function over neighbor pixels defined by
    size (i.e. number of pixels) or structure.
    A new raster is created for each input band, where each pixel is the result
    of the calculation of the function over the neighbor pixels (e.g. the mean
    of the pixel values of a 3x3 window around the pixel).
    Available functions are:
    
        - Count
        - Max
        - Mean
        - Median
        - Min
        - Percentile
        - StandardDeviation
        - Sum

    Args:
        input_bands: input of type BandSet or list of paths or integer
            number of BandSet.
        output_path: string of output path directory or list of paths.
        size: size of dilation in pixels.
        structure: optional path to csv file of structures, if None then the
            structure is created from size.
        circular_structure: if True use circular structure.
        stat_percentile: integer value for percentile parameter.
        stat_name: statistic name as in configurations.statistics_list.
        output_data_type: optional raster output data type, if None the data type is the same as input raster.
        virtual_output: if True (and output_path is directory), save output as virtual raster of multiprocess parts.
        prefix: optional string for output name prefix.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        bandset_catalog: optional type BandSetCatalog for BandSet number.

    Returns:
        Object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - paths = output list

    Examples:
        Perform the band neighbor of size 10 pixels with the function Sum
            >>> neighbor = band_neighbor_pixels(
            ... input_bands=['file1.tif', 'file2.tif'],
            ... output_path='directory_path',
            ... size=10, circular_structure=True, stat_name='Sum'
            ... )
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=True
    )
    # prepare process files
    (input_raster_list, raster_info, nodata_list, name_list, warped, out_path,
     vrt_r, vrt_path, n_processes,
     output_list, vrt_list) = shared_tools.prepare_process_files(
        input_bands=input_bands, output_path=output_path,
        n_processes=n_processes,
        bandset_catalog=bandset_catalog, prefix=prefix,
        multiple_output=True, virtual_output=virtual_output
    )
    stat_numpy = None
    for i in cfg.statistics_list:
        if i[0].lower() == stat_name.lower():
            stat_numpy = i[1]
            break
    cfg.logger.log.debug('stat_numpy: %s' % str(stat_numpy))
    if cfg.stat_percentile in stat_numpy:
        function_numpy = stat_numpy.replace('array', 'A')
        try:
            stat_percentile = int(stat_percentile)
            function_numpy = function_numpy.replace(
                cfg.stat_percentile, str(stat_percentile)
            )
        except Exception as err:
            cfg.logger.log.error(err)
            messages.error(str(err))
            return OutputManager(check=False)
    else:
        function_numpy = stat_numpy.replace('array', 'A, axis=2')
    cfg.logger.log.debug('function_numpy: %s' % str(function_numpy))
    if structure is None:
        if not circular_structure:
            structure = shared_tools.create_base_structure(size * 2 + 1)
        else:
            structure = shared_tools.create_circular_structure(size)
    else:
        try:
            structure = shared_tools.open_structure(structure)
        except Exception as err:
            cfg.logger.log.error(err)
            messages.error(str(err))
            return OutputManager(check=False)
    # process calculation
    n = 0
    min_p = 1
    max_p = int((99 - 1) / len(input_raster_list))
    for i in input_raster_list:
        out = output_list[n]
        nd = nodata_list[n]
        if output_data_type is None:
            output_data_type = raster_info[n][8]
        cfg.multiprocess.run(
            raster_path=i, function=raster_neighbor,
            function_argument=structure,
            function_variable=[function_numpy], output_raster_path=out,
            output_data_type=output_data_type, output_nodata_value=nd,
            compress=cfg.raster_compression, dummy_bands=3,
            n_processes=n_processes, available_ram=available_ram,
            boundary_size=structure.shape[0] + 1, virtual_raster=vrt_list[n],
            progress_message='processing raster %s' % str(n + 1),
            min_progress=min_p + max_p * n,
            max_progress=min_p + max_p * (n + 1)
        )
        n += 1
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; neighbor pixel: %s' % str(output_list))
    return OutputManager(paths=output_list)
