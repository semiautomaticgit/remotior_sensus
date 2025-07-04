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
"""Band spectral distance.

This tool allows for calculating the spectral distance pixel by pixel between 
two BandSets.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> catalog = rs.bandset_catalog()
    >>> # create three BandSets
    >>> file_list_1 = ['file1_b1.tif', 'file1_b2.tif', 'file1_b3.tif']
    >>> file_list_2 = ['file2_b1.tif', 'file2_b2.tif', 'file2_b3.tif']
    >>> catalog.create_bandset(file_list_1, bandset_number=1)
    >>> catalog.create_bandset(file_list_2, bandset_number=2)
    >>> distance = rs.band_spectral_distance(
    ... input_bandsets=[catalog.get(1), catalog.get(2)], output_path='output_path',
    ... )
"""  # noqa: E501

from typing import Optional

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.core.processor_functions import spectral_distance
from remotior_sensus.util import shared_tools


def band_spectral_distance(
        input_bandsets: list, output_path: Optional[str] = None,
        algorithm_name: Optional[str] = None,
        nodata_value: Optional[int] = None, threshold: Optional[float] = None,
        n_processes: Optional[int] = None, available_ram: Optional[int] = None,
        virtual_output: Optional[bool] = False,
        overwrite: Optional[bool] = False,
        bandset_catalog: Optional[BandSetCatalog] = None,
        progress_message: Optional[bool] = True
) -> OutputManager:
    """Band spectral distance.

    This tool allows for calculating the spectral distance pixel by pixel 
    between two BandSets, which can be useful for change detection.
    A new raster is created where pixel values rapresent the spectral distance 
    between pixels of input BandSets.
    Optionally, a threshold value can be defined to create a binary raster 
    where pixel value is 1 if distance > threshold or 0 otherwise.

    Args:
        input_bandsets: list of number of BandSets, or BandSet objects.
        output_path: string of output path directory.
        algorithm_name: algorithm name selected form cfg.classification_algorithms.
        nodata_value: value to be considered as nodata.
        threshold: threshold value to create a binary raster if distance > threshold.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        virtual_output: if True (and output_path is directory), save output
            as virtual raster of multiprocess parts.
        overwrite: if True, output overwrites existing files.
        bandset_catalog: optional type BandSetCatalog for BandSet number.
        progress_message: if True then start progress message, if False does 
            not start the progress message (useful if launched from other tools).

    Returns:
        Object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - path = output path

    Examples:
        Perform the spectral distance of two BandSets using threshold
            >>> # import Remotior Sensus and start the session
            >>> import remotior_sensus
            >>> rs = remotior_sensus.Session()
            >>> catalog = rs.bandset_catalog()
            >>> # create three BandSets
            >>> file_list_1 = ['file1_b1.tif', 'file1_b2.tif', 'file1_b3.tif']
            >>> file_list_2 = ['file2_b1.tif', 'file2_b2.tif', 'file2_b3.tif']
            >>> catalog.create_bandset(file_list_1, bandset_number=1)
            >>> catalog.create_bandset(file_list_2, bandset_number=2)
            >>> # start the process
            >>> distance = rs.band_spectral_distance(
            ... input_bandsets=[1, 2], output_path='output_path', 
            ... threshold=1000, bandset_catalog=catalog
            ... )
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=progress_message
    )
    cfg.logger.log.debug('input_bands: %s' % str(input_bandsets))
    if n_processes is None:
        n_processes = cfg.n_processes
    if algorithm_name is None:
        algorithm_name = cfg.minimum_distance_a
    elif algorithm_name.lower() == cfg.minimum_distance:
        algorithm_name = cfg.minimum_distance_a
    elif algorithm_name.lower() == cfg.minimum_distance_a:
        algorithm_name = cfg.minimum_distance_a
    elif algorithm_name.lower() == cfg.spectral_angle_mapping:
        algorithm_name = cfg.spectral_angle_mapping_a
    elif algorithm_name.lower() == cfg.spectral_angle_mapping_a:
        algorithm_name = cfg.spectral_angle_mapping_a
    else:
        cfg.logger.log.error('algorithm name')
        cfg.messages.error('algorithm name')
        cfg.progress.update(failed=True)
        return OutputManager(check=False)
    # list of band lists to mosaic
    if len(input_bandsets) != 2:
        cfg.logger.log.error('input bandsets')
        cfg.messages.error('input bandsets')
        cfg.progress.update(failed=True)
        return OutputManager(check=False)
    combination_band_list = []
    band_extended_list = []
    for i in input_bandsets:
        # list of band sets
        if type(i) is BandSet or type(i) is int:
            # get input list
            if i is None:
                cfg.logger.log.error('input None:%s' % str(i))
                cfg.messages.error('input None:%s' % str(i))
                cfg.progress.update(failed=True)
                return OutputManager(check=False)
            else:
                band_list = BandSetCatalog.get_band_list(i, bandset_catalog)
                combination_band_list.append(band_list)
                band_extended_list.extend(band_list)
        # list of raster paths
        else:
            cfg.logger.log.error('input bands')
            cfg.messages.error('input bands')
            cfg.progress.update(failed=True)
            return OutputManager(check=False)
    if len(combination_band_list) == 0:
        cfg.logger.log.error('input bands')
        cfg.messages.error('input bands')
        cfg.progress.update(failed=True)
        return OutputManager(check=False)
    if len(combination_band_list[0]) != len(combination_band_list[1]):
        cfg.logger.log.error('input bands')
        cfg.messages.error('input bands')
        cfg.progress.update(failed=True)
        return OutputManager(check=False)
    # prepare process files
    prepared = shared_tools.prepare_process_files(
        input_bands=band_extended_list, output_path=output_path,
        multiple_input=True, overwrite=overwrite, n_processes=n_processes,
        bandset_catalog=bandset_catalog
    )
    out_path = prepared['output_path']
    n_processes = prepared['n_processes']
    vrt_path = prepared['temporary_virtual_raster']
    # calculate spectral distance
    cfg.multiprocess.run(
        raster_path=vrt_path, function=spectral_distance,
        function_argument=len(combination_band_list[0]),
        output_raster_path=out_path,
        function_variable=[algorithm_name, threshold],
        use_value_as_nodata=nodata_value, any_nodata_mask=True,
        virtual_raster=virtual_output, n_processes=n_processes,
        available_ram=available_ram,
        progress_message='calculate spectral distance'
    )
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; band spectral distance: %s' % str(out_path))
    return OutputManager(path=out_path)
