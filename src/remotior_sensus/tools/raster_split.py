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
"""
Raster split.

This tool allows for splitting a raster to single bands.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> split = rs.raster_split(raster_path='input_path', 
    ... output_path='output_path')
"""  # noqa: E501

from typing import Optional

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.util import (
    files_directories, raster_vector, shared_tools
)


def raster_split(
        raster_path: str, output_path: str = None,
        prefix: Optional[str] = None,
        extent_list: Optional[list] = None,
        n_processes: Optional[int] = None,
        virtual_output: Optional[bool] = None
):
    """Split a multiband raster to single bands.

    This tool allows for splitting a multiband raster to single bands.

    Args:
        raster_path: path of raster used as input.
        output_path: string of output directory path.
        prefix: optional string for output name prefix.
        extent_list: list of boundary coordinates left top right bottom.
        n_processes: number of parallel processes.
        virtual_output: if True (and output_path is directory), save output
            as virtual raster.

    Returns:
        object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - path = output path

    Examples:
        Perform the split of a raster
            >>> split = raster_split(raster_path='input_path', 
            ... output_path='output_path')
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=True
    )
    raster_path = files_directories.input_path(raster_path)
    # prepare process files
    prepared = shared_tools.prepare_process_files(
        input_bands=[raster_path], output_path=output_path,
        n_processes=n_processes, box_coordinate_list=extent_list
    )
    raster_info = prepared['raster_info']
    output_list = []
    bands = raster_info[0][5]
    output_path = output_path.replace('\\', '/').replace('//', '/')
    if output_path.endswith('/'):
        output_path = output_path[:-1]
    if prefix is None:
        prefix = 'band'
    for band in range(bands):
        files_directories.create_parent_directory(output_path)
        out_path = '%s/%s%s' % (output_path, prefix, str(band + 1))
        if virtual_output is True:
            virtual_path = files_directories.output_path(out_path,
                                                         cfg.vrt_suffix)
            output = virtual_path
        else:
            virtual_path = cfg.temp.temporary_file_path(
                name_suffix=cfg.vrt_suffix)
            output = files_directories.output_path(out_path,
                                                   cfg.tif_suffix)
        raster_vector.create_virtual_raster(
            input_raster_list=[raster_path], output=virtual_path,
            band_number_list=[[band + 1]], box_coordinate_list=extent_list,
            relative_to_vrt=False
        )
        if virtual_output is not True:
            raster_vector.gdal_copy_raster(
                input_raster=virtual_path, output=output
                )
        output_list.append(output)
        cfg.progress.update(
            message='splitting', step=band, steps=bands, minimum=1,
            maximum=99, percentage=int(100 * band / bands)
        )
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; raster split: %s' % str(output_list))
    return OutputManager(paths=output_list)
