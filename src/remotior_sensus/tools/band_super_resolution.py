# Remotior Sensus , software to process remote sensing and GIS data.
# Copyright (C) 2022-2026 Luca Congedo.
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
Band super-resolution.

This tool allows for super-resolution of a multiband raster.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> stack = rs.band_super_resolution(input_bands=['path_1', 'path_2', 'path_3'],
    ... output_path='output_path')
"""  # noqa: E501

from typing import Union, Optional

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.util import (shared_tools)
from remotior_sensus.core.processor_functions import (
    super_resolution_pytorch_pretrained)

try:
    import torch
    from remotior_sensus.util.pytorch_tools import (
        superresolution_pytorch_model_s2)

except Exception as error:
    torch = None
    superresolution_pytorch_model_s2 = None
    if cfg.logger is not None:
        cfg.logger.log.error(str(error))


def band_super_resolution(
        input_bands: Union[list, int, BandSet],
        pretrained_model_path: Optional[str] = None,
        super_resolution_factor: Optional[int] = None,
        pytorch_device: Optional[str] = None,
        output_path: Optional[str] = None,
        overwrite: Optional[bool] = False,
        extent_list: Optional[list] = None,
        bandset_catalog: Optional[BandSetCatalog] = None,
        n_processes: Optional[int] = None,
        virtual_output: Optional[bool] = None,
        progress_message: Optional[bool] = True
) -> OutputManager:
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=progress_message
    )
    # TODO multiprocess while writing raster
    n_processes = 1
    # prepare process files
    prepared = shared_tools.prepare_process_files(
        input_bands=input_bands, output_path=output_path, overwrite=overwrite,
        n_processes=n_processes, bandset_catalog=bandset_catalog,
        box_coordinate_list=extent_list, virtual_output=virtual_output,
        multiple_output=True
    )
    input_raster_list = prepared['input_raster_list']
    out_path = prepared['output_path']
    n_processes = prepared['n_processes']
    virtual_raster = False
    dummy_bands = 2
    available_ram = None
    function_argument = {
        cfg.model_path_framework: pretrained_model_path,
        cfg.pytorch_framework: pytorch_device,
        cfg.n_processes_framework: n_processes,
    }
    cfg.multiprocess.run(
        raster_path=input_raster_list,
        function=super_resolution_pytorch_pretrained,
        function_argument=function_argument,
        n_processes=n_processes, output_data_type=cfg.float32_dt,
        output_nodata_value=cfg.nodata_val_UInt32,
        available_ram=available_ram, dummy_bands=dummy_bands,
        function_variable=None,
        output_raster_path=out_path, super_resolution=True,
        super_resolution_factor=super_resolution_factor,
        virtual_raster=virtual_raster,
        progress_message='super-resolution'
    )

    cfg.progress.update(end=True)
    cfg.logger.log.info('end; band stack: %s' % str(out_path))
    return OutputManager(path=out_path)
