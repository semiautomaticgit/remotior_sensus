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
Band stack.

This tool allows for stacking single bands in a multiband raster.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> stack = rs.band_stack(input_bands=['path_1', 'path_2'],
    ... output_path='output_path')
"""  # noqa: E501

from typing import Union, Optional

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.util import (raster_vector, shared_tools)


def band_stack(
        input_bands: Union[list, int, BandSet],
        output_path: Optional[str] = None,
        overwrite: Optional[bool] = False,
        extent_list: Optional[list] = None,
        bandset_catalog: Optional[BandSetCatalog] = None,
        n_processes: Optional[int] = None,
        virtual_output: Optional[bool] = None
):
    """Stack single bands.

    This tool allows for stacking single bands in a multiband raster.

    Args:
        input_bands: list of paths of input rasters, or number of BandSet, or BandSet object.
        output_path: string of output path.
        overwrite: if True, output overwrites existing files.
        extent_list: list of boundary coordinates left top right bottom.
        bandset_catalog: BandSetCatalog object required if input_bands is a BandSet number.
        n_processes: number of parallel processes.
        virtual_output: if True (and output_path is directory), save output
            as virtual raster.

    Returns:
        object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - path = output path

    Examples:
        Perform band stack
            >>> stack = band_stack(input_bands=['path_1', 'path_2'],
            ... output_path='output_path')
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
        box_coordinate_list=extent_list, virtual_output=virtual_output
    )
    input_raster_list = prepared['input_raster_list']
    out_path = prepared['output_path']
    if input_bands is BandSet:
        bandset_x = input_bands
    elif input_bands is int:
        bandset_x = bandset_catalog.get(input_bands)
    else:
        bandset_x = BandSet.create(paths=input_raster_list)
    if virtual_output:
        virtual_path = out_path
    else:
        virtual_path = cfg.temp.temporary_file_path(name_suffix=cfg.vrt_suffix)
    raster_vector.create_virtual_raster(output=virtual_path, bandset=bandset_x)
    cfg.progress.update(message='stack', step=2, steps=2, minimum=1,
                        maximum=99, percentage=50)
    if virtual_output is not True:
        raster_vector.gdal_copy_raster(input_raster=virtual_path,
                                       output=out_path)
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; band stack: %s' % str(out_path))
    return OutputManager(path=out_path)
