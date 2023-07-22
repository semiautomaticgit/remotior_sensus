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
"""Band dilation.

This tool allows for the spatial dilation, through a moving window,
of band pixels selected by values.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> dilation = rs.band_dilation(input_bands=['file1.tif', 'file2.tif'],
    ... value_list=[1], size=3, output_path='directory_path',
    ... circular_structure=True, prefix='dilation_')
"""

from typing import Union, Optional

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.core.processor_functions import raster_dilation
from remotior_sensus.util import shared_tools


def band_dilation(
        input_bands: Union[list, int, BandSet], value_list: list, size: int,
        output_path: Union[list, str] = None,
        overwrite: Optional[bool] = False,
        circular_structure: Optional[bool] = None,
        prefix: Optional[str] = '', extent_list: Optional[list] = None,
        n_processes: Optional[int] = None,
        available_ram: Optional[int] = None,
        bandset_catalog: Optional[BandSetCatalog] = None,
        virtual_output: Optional[bool] = None
) -> OutputManager:
    """Perform dilation of band pixels.

    This tool performs the dilation of pixels identified by a list of values.
    A new raster is created for each input band.

    Args:
        input_bands: input of type BandSet or list of paths or integer
            number of BandSet.
        output_path: string of output path directory or list of paths.
        overwrite: if True, output overwrites existing files.
        value_list: list of values for dilation.
        size: size of dilation in pixels.
        virtual_output: if True (and output_path is directory), save output
            as virtual raster of multiprocess parts
        circular_structure: if True, use circular structure; if False, square structure.
        prefix: optional string for output name prefix.
        extent_list: list of boundary coordinates left top right bottom.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        bandset_catalog: optional type BandSetCatalog for BandSet number

    Returns:
        Object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - paths = output list

    Examples:
        Perform the dilation of size 5 for value 1 and 2
            >>> dilation = band_dilation(input_bands=['path_1', 'path_2'],value_list=[1, 2],size=5,output_path='directory_path',circular_structure=True)
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=True
    )
    # prepare process files
    prepared = shared_tools.prepare_process_files(
        input_bands=input_bands, output_path=output_path, overwrite=overwrite,
        n_processes=n_processes, box_coordinate_list=extent_list,
        bandset_catalog=bandset_catalog, prefix=prefix,
        multiple_output=True, multiple_input=True,
        virtual_output=virtual_output
    )
    input_raster_list = prepared['input_raster_list']
    raster_info = prepared['raster_info']
    n_processes = prepared['n_processes']
    nodata_list = prepared['nodata_list']
    output_list = prepared['output_list']
    vrt_list = prepared['vrt_list']
    if not circular_structure:
        structure = shared_tools.create_base_structure(size * 2 + 1)
    else:
        structure = shared_tools.create_circular_structure(size)
    # process calculation
    n = 0
    min_p = 1
    max_p = int((99 - 1) / len(input_raster_list))
    # dummy bands for memory calculation as the number of values
    dummy_bands = len(value_list) + 4
    for i in input_raster_list:
        out = output_list[n]
        nd = nodata_list[n]
        data_type = raster_info[n][8]
        cfg.multiprocess.run(
            raster_path=i, function=raster_dilation,
            function_argument=structure, n_processes=n_processes,
            available_ram=available_ram,
            function_variable=value_list, output_raster_path=out,
            output_data_type=data_type, output_nodata_value=nd,
            compress=cfg.raster_compression, dummy_bands=dummy_bands,
            boundary_size=structure.shape[0] + 1, virtual_raster=vrt_list[n],
            progress_message='processing raster %s' % (n + 1),
            min_progress=min_p + max_p * n,
            max_progress=min_p + max_p * (n + 1)
        )
        n += 1
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; band dilation: %s' % output_list)
    return OutputManager(paths=output_list)
