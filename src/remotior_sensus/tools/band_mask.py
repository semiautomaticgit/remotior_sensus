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
Band mask.

This tool allows for masking bands using a vector or raster mask.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> rs.band_mask(input_bands=['path_1', 'path_2'], 
    ... input_mask='mask_path', mask_values=[1, 2]
    ... output_path='output_path')
"""  # noqa: E501

from typing import Union, Optional

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.util import shared_tools, raster_vector
from remotior_sensus.tools.band_dilation import band_dilation
from remotior_sensus.core.processor_functions import band_calculation


def band_mask(
        input_bands: Union[list, int, BandSet],
        input_mask: str = None,
        mask_values: list = None,
        output_path: Optional[str] = None,
        overwrite: Optional[bool] = False,
        buffer: Optional[int] = None,
        nodata_value: Optional[int] = None,
        virtual_output: Optional[bool] = None,
        compress=None, compress_format=None,
        prefix: Optional[str] = '',
        extent_list: Optional[list] = None,
        n_processes: Optional[int] = None,
        available_ram: Optional[int] = None,
        bandset_catalog: Optional[BandSetCatalog] = None
):
    """Performs band mask.

    This tool allows for masking bands using a vector or raster mask.

    Args:
        input_bands: reference_raster of type BandSet or list of paths or integer
            number of BandSet.
        input_mask: string path of raster or vector used for masking.
        mask_values: list of values in the mask to be used for masking.
        output_path: string of output path directory or list of paths.
        overwrite: if True, output overwrites existing files.
        buffer: optional buffer size, in number of pixels, to expand the mask.
        nodata_value: value to be used as nodata in the output.
        virtual_output: if True (and output_path is directory), save output as virtual raster of multiprocess parts.
        prefix: optional string for output name prefix.
        extent_list: list of boundary coordinates left top right bottom.
        compress: if True, compress output.
        compress_format: format of compressions such as LZW or DEFLATE.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        bandset_catalog: optional type BandSetCatalog for BandSet number.

    Returns:
        Object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - paths = output list

    Examples:
        Perform the band resample
            >>> band_mask(input_bands=['path_1', 'path_2'], 
            ... input_mask='mask_path', mask_values=[1, 2],
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
        n_processes=n_processes, box_coordinate_list=extent_list,
        bandset_catalog=bandset_catalog, prefix=prefix,
        temporary_virtual_raster=True,
        multiple_output=True, virtual_output=virtual_output
    )
    input_raster_list = prepared['input_raster_list']
    raster_info = prepared['raster_info']
    n_processes = prepared['n_processes']
    nodata_list = prepared['nodata_list']
    output_list = prepared['output_list']
    # if vector convert to raster
    vector, raster, mask_crs = raster_vector.raster_or_vector_input(
        input_mask
    )
    # check crs
    same_crs = raster_vector.compare_crs(raster_info[0][1], mask_crs)
    # if reference is raster
    if raster:
        if not same_crs:
            t_pmd = cfg.temp.temporary_raster_path(extension=cfg.vrt_suffix)
            reference_raster = cfg.multiprocess.create_warped_vrt(
                raster_path=input_mask, output_path=t_pmd,
                output_wkt=str(mask_crs)
            )
        else:
            reference_raster = input_mask
    # if reference is vector
    else:
        if not same_crs:
            # project vector to raster crs
            t_vector = cfg.temp.temporary_file_path(
                name_suffix=cfg.gpkg_suffix
            )
            try:
                raster_vector.reproject_vector(
                    input_mask, t_vector, raster_info[0][1], mask_crs
                )
            except Exception as err:
                cfg.logger.log.error(str(err))
                cfg.messages.error(str(err))
                return OutputManager(check=False)
            input_mask = t_vector
        # convert vector to raster
        reference_raster = cfg.temp.temporary_raster_path(
            extension=cfg.tif_suffix
        )
        mask_values = [1]
        # perform conversion
        cfg.multiprocess.multiprocess_vector_to_raster(
            vector_path=input_mask,
            output_path=reference_raster, burn_values=1,
            reference_raster_path=input_raster_list[0], nodata_value=0,
            background_value=0, available_ram=available_ram,
            minimum_extent=False
        )
    if buffer is not None:
        size = int(buffer)
        vrt_file = cfg.temp.temporary_raster_path(extension=cfg.vrt_suffix)
        band_dilation(
            input_bands=[reference_raster], value_list=[1], size=size,
            output_path=[vrt_file], circular_structure=True
        )
        reference_raster = vrt_file
    cfg.logger.log.debug('reference_raster: %s' % reference_raster)
    argument_list = []
    scale_list = []
    offset_list = []
    output_nodata_list = []
    calculation_datatype = []
    output_datatype = []
    variables = []
    variable_list = []
    for b in range(0, len(input_raster_list)):
        if nodata_value is None:
            nodata_value = raster_info[b][4]
        output_nodata_list.append(nodata_value)
        output_datatype.append(raster_info[b][8])
        calculation_datatype.append(
            shared_tools.data_type_conversion(raster_info[b][8]))
        scale_list.append(raster_info[b][7][0])
        offset_list.append(raster_info[b][7][1])
        variables.append('"band%i"' % b)
        # function
        expression = ''
        closing = ''
        for c in mask_values:
            expression += 'np.where(%s[::, ::, 0] == %i, %i, ' % (
                cfg.array_function_placeholder, c, nodata_value)
            closing += ')'
        expression += '%s[::, ::, %i]%s' % (
            cfg.array_function_placeholder, b + 1, closing)
        argument_list.append(expression)
        variable_list.append(variables)
    cfg.logger.log.debug('argument_list: %s' % argument_list)
    # insert reference raster
    input_raster_list.insert(0, reference_raster)
    cfg.logger.log.debug('reference_raster: %s' % reference_raster)
    cfg.logger.log.debug('input_raster_list: %s' % input_raster_list)
    # prepare process files with reference
    prepared = shared_tools.prepare_process_files(
        input_bands=input_raster_list, output_path=output_path,
        overwrite=overwrite, n_processes=n_processes, prefix=prefix,
        temporary_virtual_raster=True,
        multiple_output=True, virtual_output=virtual_output
    )
    vrt_path_x = prepared['temporary_virtual_raster']
    # dummy bands for memory calculation
    dummy_bands = 2
    min_progress = 1
    one_progress = int((99 - 1) / (len(input_raster_list) - 1))
    max_progress = one_progress
    for band in range(len(input_raster_list) - 1):
        # run calculation
        cfg.multiprocess.run(
            raster_path=vrt_path_x, function=band_calculation,
            function_argument=argument_list[band],
            function_variable=variable_list[band],
            calculation_datatype=calculation_datatype[band],
            use_value_as_nodata=nodata_list[band], dummy_bands=dummy_bands,
            output_raster_path=output_list[band],
            output_data_type=output_datatype[band],
            output_nodata_value=output_nodata_list[band],
            compress=compress, compress_format=compress_format,
            n_processes=n_processes, available_ram=available_ram,
            scale=scale_list[band], offset=offset_list[band],
            progress_message='masking',
            min_progress=min_progress, max_progress=max_progress
        )
        min_progress = max_progress
        max_progress += one_progress
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; band resample: %s' % str(output_list))
    return OutputManager(paths=output_list)
