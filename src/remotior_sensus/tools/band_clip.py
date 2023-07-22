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
"""Band clip.

This tool allows for clipping the bands of a BandSet.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # box coordinate list
    >>> extent_list = [230250, 4674510, 230320, 4674440]
    >>> # start the process
    >>> clip = rs.band_clip(input_bands=['path_1', 'path_2'],
    ... output_path='output_path', extent_list=extent_list)
"""

from typing import Union, Optional

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.util import shared_tools, raster_vector
from remotior_sensus.core.processor_functions import clip_raster


def band_clip(
        input_bands: Union[list, int, BandSet],
        output_path: Union[list, str] = None,
        vector_path: Optional[str] = '',
        vector_field: Optional[str] = None,
        overwrite: Optional[bool] = False,
        prefix: Optional[str] = '',
        extent_list: Optional[list] = None,
        n_processes: Optional[int] = None,
        available_ram: Optional[int] = None,
        bandset_catalog: Optional[BandSetCatalog] = None,
        virtual_output: Optional[bool] = None
) -> OutputManager:
    """Perform band clip.

    This tool allows for clipping the bands of a BandSet based on a vector or list of boundary coordinates left top right bottom.

    Args:
        input_bands: input of type BandSet or list of paths or integer
            number of BandSet.
        output_path: string of output path directory or list of paths.
        overwrite: if True, output overwrites existing files.
        vector_path: path of vector used to clip.
        vector_field: vector field name used to clip for every unique ID.
        virtual_output: if True (and output_path is directory), save output
            as virtual raster of multiprocess parts
        prefix: optional string for output name prefix.
        extent_list: list of boundary coordinates left top right bottom.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        bandset_catalog: optional type BandSetCatalog for BandSet number

    Returns:
        Object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - paths = output list
        
    Examples:
        Clip using vector
            >>> # start the process  
            >>> clip = band_clip(input_bands=['path_1', 'path_2'],
            ... output_path='output_path', vector_path='vector_path',
            ... prefix='clip_')
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=True
    )
    if n_processes is None:
        n_processes = cfg.n_processes
    if available_ram is None:
        available_ram = cfg.available_ram
    ram = int(available_ram / n_processes)
    # prepare process files
    prepared = shared_tools.prepare_process_files(
        input_bands=input_bands, output_path=output_path, overwrite=overwrite,
        n_processes=n_processes, bandset_catalog=bandset_catalog,
        box_coordinate_list=extent_list,
        prefix=prefix, multiple_output=True, multiple_input=True,
        virtual_output=virtual_output
    )
    input_raster_list = prepared['input_raster_list']
    output_list = prepared['output_list']
    # build function argument list of dictionaries
    argument_list = []
    function_list = []
    output_raster_list = []
    if vector_field is not None:
        # find unique values of vector_field
        unique_values = raster_vector.get_vector_values(
            vector_path=vector_path, field_name=vector_field)
        for value in unique_values:
            for raster in range(0, len(input_raster_list)):
                output_p = '%s_%s_%s.tif' % (
                        output_list[raster][:-4], str(vector_field),
                        str(value))
                output_raster_list.append(output_p)
                argument_list.append(
                    {
                        'input_raster': input_raster_list[raster],
                        'extent_list': None,
                        'vector_path': vector_path,
                        'available_ram': ram,
                        'output': output_p,
                        'gdal_path': cfg.gdal_path,
                        'compress_format': 'LZW',
                        'where': "%s = %s" % (vector_field, value)
                    }
                )
                function_list.append(clip_raster)
    else:
        for raster in range(0, len(input_raster_list)):
            argument_list.append(
                {
                    'input_raster': input_raster_list[raster],
                    'extent_list': extent_list,
                    'vector_path': vector_path,
                    'available_ram': ram,
                    'output': output_list[raster],
                    'gdal_path': cfg.gdal_path,
                    'compress_format': 'LZW',
                    'where': None
                }
            )
            function_list.append(clip_raster)
            output_raster_list.append(output_list[raster])
    cfg.multiprocess.run_iterative_process(
        function_list=function_list, argument_list=argument_list
    )
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; band clip: %s' % output_raster_list)
    return OutputManager(paths=output_raster_list)
