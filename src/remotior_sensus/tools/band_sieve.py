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
"""Band sieve.

This tool allows for performing the sieve of raster bands removing
patches having size lower than a threshold (i.e. number of pixels).

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> sieve = rs.band_sieve(input_bands=['file1.tif', 'file2.tif'],size=2,
    ... output_path='directory_path',connected=False,prefix='sieve_')
"""

from typing import Union, Optional

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.util import shared_tools


def band_sieve(
        input_bands: Union[list, int, BandSet], size: int,
        output_path: Union[list, str] = None, connected: Optional[bool] = None,
        overwrite: Optional[bool] = False,
        prefix: Optional[str] = '', extent_list: Optional[list] = None,
        n_processes: Optional[int] = None,
        available_ram: Optional[int] = None,
        bandset_catalog: Optional[BandSetCatalog] = None,
        virtual_output: Optional[bool] = None
) -> OutputManager:
    """Perform band sieve.

    This tool allows for performing the sieve of raster bands removing
    patches having size lower than a threshold (i.e. number of pixels).

    Args:
        input_bands: input of type BandSet or list of paths or integer
            number of BandSet.
        output_path: string of output path directory or list of paths.
        overwrite: if True, output overwrites existing files.
        size: size of dilation in pixels.
        virtual_output: if True (and output_path is directory), save output
            as virtual raster of multiprocess parts
        connected: if True, consider 8 pixel connection; if False, consider 4 pixel connection.
        prefix: optional string for output name prefix.
        extent_list: list of boundary coordinates left top right bottom.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        bandset_catalog: optional type BandSetCatalog for BandSet number

    Returns:
        Object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - paths = output list
        
    Examples:
        Perform the sieve of size 3 with connected pixel (8 connection)
            >>> sieve = band_sieve(input_bands=['file1.tif', 'file2.tif'],size=3,output_path='directory_path',connected=True,prefix='sieve_')
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=True
    )
    # prepare process files
    prepared = shared_tools.prepare_process_files(
        input_bands=input_bands, output_path=output_path, overwrite=overwrite,
        n_processes=n_processes, bandset_catalog=bandset_catalog,
        box_coordinate_list=extent_list,
        prefix=prefix, multiple_output=True, multiple_input=True,
        virtual_output=virtual_output
    )
    input_raster_list = prepared['input_raster_list']
    raster_info = prepared['raster_info']
    n_processes = prepared['n_processes']
    nodata_list = prepared['nodata_list']
    output_list = prepared['output_list']
    # 4 connected pixels
    if connected:
        connected = 8
    elif not connected:
        connected = 4
    else:
        connected = 4
    # process calculation
    n = 0
    min_p = 1
    max_p = int((99 - 1) / len(input_raster_list))
    for i in input_raster_list:
        cfg.progress.update(message='processing raster %s' % (n + 1))
        out = output_list[n]
        nd = nodata_list[n]
        data_type = raster_info[n][8]
        # perform sieve
        cfg.multiprocess.multiprocess_raster_sieve(
            raster_path=i, n_processes=n_processes,
            available_ram=available_ram, sieve_size=size,
            connected=connected, output_nodata_value=nd, output=out,
            output_data_type=data_type, compress=cfg.raster_compression,
            min_progress=min_p + max_p * n,
            max_progress=min_p + max_p * (n + 1)
        )
        n += 1
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; band sieve: %s' % output_list)
    return OutputManager(paths=output_list)
