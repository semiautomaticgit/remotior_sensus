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
Raster to vector.

This tool allows for the conversion from raster to vector.
A new geopackage is created from the raster conversion.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> vector = rs.raster_to_vector(raster_path='file.tif',output_path='vector.gpkg')
)
"""  # noqa: E501

from typing import Optional

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.util import files_directories, shared_tools


def raster_to_vector(
        raster_path, output_path: Optional[str] = None,
        dissolve: Optional[bool] = None, field_name: Optional[str] = None,
        extent_list: Optional[list] = None,
        n_processes: Optional[int] = None, available_ram: Optional[int] = None
) -> OutputManager:
    """Performs the conversion from raster to vector.

    This tool performs the conversion from raster to vector.
    Parallel processes are used for the conversion, resulting in a vector output
    which is split as many in portions as the process numbers.
    The argument dissolve allows for merging these portions,
    but it requires additional processing time depending on vector size.

    Args:
        raster_path: path of raster used as input.
        output_path: string of output path.
        dissolve: if True, dissolve adjacent polygons having the same values;
            if False, polygons are not dissolved and the process is rapider.
        field_name: name of the output vector field to store raster values (default = DN).
        extent_list: list of boundary coordinates left top right bottom.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.

    Returns:
        object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - path = output path

    Examples:
        Perform the conversion to vector of a raster
            >>> raster_to_vector(raster_path='file.tif',output_path='vector.gpkg')
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=True
    )
    raster_path = files_directories.input_path(raster_path)
    if extent_list is not None:
        # prepare process files
        prepared = shared_tools.prepare_process_files(
            input_bands=[raster_path], output_path=output_path,
            n_processes=n_processes, box_coordinate_list=extent_list
        )
        input_raster_list = prepared['input_raster_list']
        raster_path = input_raster_list[0]
    if output_path is None:
        output_path = cfg.temp.temporary_file_path(name_suffix=cfg.gpkg_suffix)
    output_path = files_directories.output_path(output_path, cfg.gpkg_suffix)
    files_directories.create_parent_directory(output_path)
    if n_processes is None:
        n_processes = cfg.n_processes
    # perform conversion
    cfg.multiprocess.multiprocess_raster_to_vector(
        raster_path=raster_path, output_vector_path=output_path,
        field_name=field_name, n_processes=n_processes,
        dissolve_output=dissolve, min_progress=1, max_progress=100,
        available_ram=available_ram
    )
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; output_path: %s' % output_path)
    return OutputManager(path=output_path)
