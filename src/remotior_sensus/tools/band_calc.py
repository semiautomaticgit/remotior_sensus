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
"""Band calc.

This tool allows for mathematical calculations (pixel by pixel) between
bands or single band rasters.
A new raster file is created as result of calculation.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> output = rs.band_calc(
    ... input_raster_list=['file1.tif', 'file2.tif'], output_path='output.tif',
    ... expression_string='"file1 + file2"', input_name_list=['file1', 'file2']
    ... )
"""

import datetime
import re
from typing import Optional

import numpy as np

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.core.processor_functions import band_calculation
from remotior_sensus.util import (
    dates_times, files_directories, raster_vector, shared_tools
)


def band_calc(
        expression_string: str, output_path: Optional[str] = None,
        input_raster_list: Optional[list] = None,
        input_name_list: Optional[list] = None,
        n_processes: Optional[int] = None,
        available_ram: Optional[int] = None,
        align_raster: Optional[str] = None,
        extent_raster: Optional[str] = None,
        extent_list: Optional[list] = None,
        extent_intersection: Optional[bool] = True,
        xy_resolution_list: Optional[list] = None,
        input_nodata_as_value: Optional[bool] = None,
        use_value_as_nodata: Optional[int] = None,
        output_nodata: Optional[int] = None,
        output_datatype: Optional[str] = None,
        use_scale: Optional[float] = None,
        use_offset: Optional[float] = None,
        calc_datatype: Optional[str] = None,
        any_nodata_mask: Optional[bool] = False,
        bandset_catalog: Optional[BandSetCatalog] = None,
        bandset_number: Optional[int] = None,
        input_bands: Optional[BandSet] = None
) -> OutputManager:
    """Performs band calculation.

    Calculation is defined by an expression string using variable names that 
    corresponds to input bands or rasters.
    Expression can use band alias such as "bandset1b1", spectral band alias 
    such as "#NIR#" for Near-Infrared and "#RED#" for Red,
    or expression alias such as "#NDVI#".
    Multiple expression lines can be entered for serial calculation.
    Several iteration functions are available for band sets.
    NumPy functions can also be used if the output is a single band array 
    with the same size as input raster.
    Input rasters can have different projection definitions, as the tool 
    will try to reproject input rasters on the fly based on a reference raster
    (first input raster or align_raster).

    Args:
        input_raster_list: list of input raster paths or list of lists 
            [path, name] ignoring input_name_list.
        output_path: path of output file for single expression or 
            path to a directory for multiple expression outputs.
        expression_string: expression string used for calculation; multiple 
            expressions can be entered separated by new line.
        input_name_list: list of input raster names used in expressions 
            (if name not defined in input_raster_list).
        input_bands: input BandSet for direct expression; in expressions, bands can be refferred as "b1", "b2", etc.;
            also, spectral band alias such as "#RED#" or "#NIR#"; also "b*" for using all the bands.
        n_processes: number of threads for calculation.
        available_ram: number of megabytes of RAM available to processes.
        align_raster: string path of raster used for aligning output pixels and projections.
        extent_raster: string path of raster used for extent reference.
        extent_list: list of coordinates for defining calculation extent 
            [left, top, right, bottom] in the same coordinates as the reference raster.
        extent_intersection: if True the output extent is geometric 
            intersection of input raster extents, if False the output extent 
            is the maximum extent from union of input raster extents.
        xy_resolution_list: list of [x, y] pixel resolution.
        input_nodata_as_value: if True then unmask the value of nodata pixels 
            in calculations, if False then mask nodata pixels in calculations.
        use_value_as_nodata: use integer value as nodata in calculation.
        output_nodata: integer value used as nodata in output raster.
        output_datatype: string of data type for output raster such as 
            Float64, Float32, Int32, UInt32, Int16, UInt16, or Byte.
        use_scale: float number used for scale for output.
        use_offset: float number used for offset for output.
        calc_datatype: data type used during calculation, which may differ 
            from output_datatype, such as Float64, Float32, Int32, UInt32, Int16, UInt16, or Byte.
        any_nodata_mask: if True then output nodata where any input is nodata, 
            if False then output nodata where all the inputs are nodata, 
            if None then do not apply nodata to output.
        bandset_catalog: BandSetCatalog object for using band sets in calculations.
        bandset_number: number of BandSet defined as current one.

    Returns:
        :func:`~remotior_sensus.core.output_manager.OutputManager` object with
            - paths = [output raster paths]
            
    Examples:
        Sum of two raster files
            >>> output_object = band_calc(
            ... input_raster_list=['file1.tif', 'file2.tif'], output_path='output.tif',
            ... expression_string='"file1 + file2"', input_name_list=['file1', 'file2']
            ... )
            >>> # for instance display the output path
            >>> print(output_object.paths)
            ['output.tif']
            
        Calculation setting output datatype
            >>> output_object = band_calc(
            ... input_raster_list=['file1.tif', 'file2.tif'], output_path='output.tif',
            ... expression_string='"file1 + file2"', input_name_list=['file1', 'file2'],
            ... output_datatype='Int32'
            ... )
            
        Calculation setting the output extent as the maximum extent from union of input raster extents
            >>> output_object = band_calc(
            ... input_raster_list=['file1.tif', 'file2.tif'], output_path='output.tif',
            ... expression_string='"file1 + file2"', input_name_list=['file1', 'file2'],
            ... extent_intersection=False
            ... )
    """  # noqa: E501
    if input_bands is not None:
        output = _calculate_bandset(
            input_bands=input_bands, output_path=output_path,
            expression_string=expression_string, n_processes=n_processes,
            available_ram=available_ram, align_raster=align_raster,
            extent_raster=extent_raster, extent_list=extent_list,
            extent_intersection=extent_intersection,
            xy_resolution_list=xy_resolution_list,
            input_nodata_as_value=input_nodata_as_value,
            use_value_as_nodata=use_value_as_nodata,
            output_nodata=output_nodata, output_datatype=output_datatype,
            use_scale=use_scale, use_offset=use_offset,
            calc_datatype=calc_datatype, any_nodata_mask=any_nodata_mask
        )
        return output
    else:
        cfg.logger.log.info('start')
        cfg.progress.update(
            process=__name__.split('.')[-1].replace('_', ' '),
            message='starting', start=True
        )
        cfg.logger.log.debug(
            'input_bandset_list: %s; input_name_list: %s' % (
                str(input_raster_list), str(input_name_list))
        )
        # create list of band names from band sets
        raster_variables = _band_names_alias(
            input_raster_list, input_name_list, bandset_catalog, bandset_number
        )
        # check expression
        exp_list, all_out_name_list, output_message = _check_expression(
            expression_string, raster_variables, bandset_catalog,
            bandset_number, output_path
        )
        if output_message is not None:
            cfg.logger.log.error('expression error: %s', output_message)
            return OutputManager(
                check=False, extra={'message': output_message}
            )
        output_list = []
        # process calculation
        n = 0
        min_p = 1
        max_p = int((99 - 1) / len(exp_list))
        previous_output_list = []
        for e in exp_list:
            output, out_name = _run_expression(
                expression_list=e, output_path=output_path,
                previous_output_list=previous_output_list,
                n_processes=n_processes, available_ram=available_ram,
                extent_raster=extent_raster, align_raster=align_raster,
                extent_list=extent_list,
                extent_intersection=extent_intersection,
                xy_resolution_list=xy_resolution_list,
                input_nodata_as_value=input_nodata_as_value,
                use_value_as_nodata=use_value_as_nodata,
                output_nodata=output_nodata,
                output_datatype=output_datatype, use_scale=use_scale,
                use_offset=use_offset, calc_datatype=calc_datatype,
                nodata_mask=any_nodata_mask, min_progress=min_p + max_p * n,
                max_progress=min_p + max_p * (n + 1),
                progress_message='running calculation %s' % (n + 1),
                bandset_catalog=bandset_catalog
            )
            output_list.append(output)
            previous_output_list.append([output, out_name])
            cfg.logger.log.debug('output: %s' % output)
            n += 1
        cfg.progress.update(end=True)
        cfg.logger.log.info('end; band calc: %s', output_list)
        return OutputManager(paths=output_list)


def _run_expression(
        expression_list, output_path=None, previous_output_list=None,
        n_processes: int = None, available_ram: int = None,
        extent_raster=None, align_raster=None,
        extent_list=None, extent_intersection=True, xy_resolution_list=None,
        input_nodata_as_value=None, use_value_as_nodata=None,
        output_nodata=None, output_datatype=None, use_scale=None,
        use_offset=None, calc_datatype=None, nodata_mask=None,
        min_progress=None, max_progress=None, progress_message=None,
        bandset_catalog=None
) -> tuple:
    """Run expression calculation.

    Run the expression of calculation using parallel processes.

    Args:
        expression_list: list of input expression parameters.
        output_path: output file path for single expression or directory path for multiple expressions.
        previous_output_list: list of previous output path and output name list.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        extent_raster: optional path of raster used for extent reference.
        align_raster: optional path of raster used for aligning pixels.
        extent_list: optional list of extent coordinates.
        extent_intersection: calculate extent from input raster intersection.
        xy_resolution_list: optional list of x y pixel resolution.
        input_nodata_as_value: True to consider the value of nodata pixels;
            False to ignore nodata pixels.
        use_value_as_nodata: use value as nodata.
        output_nodata: output nodata value.
        output_datatype: string of data type for output raster such as Float32 or Int16.
        use_scale: optional integer number of scale for output.
        use_offset: optional integer number of offset for output.
        calc_datatype: calculation data type.
        nodata_mask: True to apply the input nodata mask to output;
            False to not apply the input nodata mask.
        min_progress: minimum progress value.
        max_progress: maximum progress value.
        bandset_catalog: BandSetCatalog object.
        progress_message: progress message.

    Returns:
        The tuple output path and the output name.
    """  # noqa: E501
    cfg.logger.log.debug('expression_list: %s' % str(expression_list))
    if n_processes is None:
        n_processes = cfg.n_processes
    (expression, expr_function, output_name, bs_number, out_path, virtual,
     raster_variables_dict) = expression_list
    # add previous outputs to raster variables dictionary
    for x in previous_output_list:
        raster_variables_dict[x[1]] = x[0]
    # get output path
    output, out_name, virtual = _get_output_path(
        output_name, out_path, virtual, output_path
    )
    cfg.logger.log.debug('output: %s' % output)
    # convert datatype string to numpy data type
    if calc_datatype == cfg.float64_dt:
        data_type = np.float64
    elif calc_datatype == cfg.float32_dt:
        data_type = np.float32
    elif calc_datatype == cfg.int32_dt:
        data_type = np.int32
    elif calc_datatype == cfg.uint32_dt:
        data_type = np.uint32
    elif calc_datatype == cfg.int16_dt:
        data_type = np.int16
    elif calc_datatype == cfg.uint16_dt:
        data_type = np.uint16
    elif calc_datatype == cfg.byte_dt:
        data_type = np.byte
    else:
        data_type = np.float32
    # get function from expression
    expr_function, input_rasters = _expression_to_function(
        expression, raster_variables_dict
    )
    expr_function = expr_function.split(cfg.variable_output_separator)[0]
    try:
        _check_numpy_operators(expr_function, len(input_rasters))
    except Exception as err:
        cfg.logger.log.error(str(err))
        cfg.messages.error(str(err))
        return False, False
    # get input files
    prepared = shared_tools.prepare_input_list(
        list(input_rasters.values()), reference_raster_crs=extent_raster,
        n_processes=n_processes
    )
    input_raster_list = prepared['input_list']
    # get extent and resolution
    xy_resolution = xy_resolution_list
    p_x, p_y = None, None
    if extent_raster is not None:
        left, top, right, bottom, p_x, p_y, output_wkt, unit = \
            raster_vector.image_geotransformation(
                extent_raster
            )
        extent_list = [left, top, right, bottom]
    if align_raster is not None:
        xy_resolution = [p_x, p_y]
    # create virtual raster of input
    vrt_check = raster_vector.create_temporary_virtual_raster(
        input_raster_list, intersection=extent_intersection,
        box_coordinate_list=extent_list, pixel_size=xy_resolution,
        grid_reference=align_raster
    )
    cfg.logger.log.debug('vrt_check: %s' % vrt_check)
    # dummy bands for memory calculation as twice the number of input raster
    dummy_bands = len(input_raster_list) + 1
    # run calculation
    cfg.multiprocess.run(
        raster_path=vrt_check, function=band_calculation,
        function_argument=expr_function, calculation_datatype=data_type,
        use_value_as_nodata=use_value_as_nodata, n_processes=n_processes,
        available_ram=available_ram, any_nodata_mask=nodata_mask,
        output_raster_path=output, output_data_type=output_datatype,
        output_nodata_value=output_nodata, compress=cfg.raster_compression,
        scale=use_scale, offset=use_offset, dummy_bands=dummy_bands,
        input_nodata_as_value=input_nodata_as_value, virtual_raster=virtual,
        progress_message=progress_message, min_progress=min_progress,
        max_progress=max_progress
    )
    # add output to BandSet
    if bs_number is not None and bandset_catalog is not None:
        bandset_catalog.add_band_to_bandset(
            path=output, bandset_number=int(bs_number)
        )
        try:
            bandset_catalog.add_band_to_bandset(
                path=output, bandset_number=int(bs_number)
            )
        except Exception as err:
            cfg.logger.log.error(str(err))
            cfg.messages.error(str(err))
    return output, out_name


def _get_output_path(
        expression_output_name=None, expression_out_path=None, virtual=None,
        output_path=None
) -> tuple:
    """Gets output path.

    Gets output path for calculation.

    Args:
        virtual: True for virtual output or False for raster file.
        expression_output_name: output name string from expression.
        expression_out_path: output directory from expression.
        output_path: output file path for single expression or directory path
            for multiple expressions.
    """
    cfg.logger.log.debug(
        'expression_output_name: %s; expression_out_path: %s; output_path: %s'
        % (expression_output_name, expression_out_path, output_path)
    )
    # virtual
    if output_path is not None and files_directories.file_extension(
            output_path.lower()
    ) == cfg.vrt_suffix:
        virtual = True
    cfg.logger.log.debug('virtual: %s' % virtual)
    if virtual:
        out_extension = cfg.vrt_suffix
    else:
        out_extension = cfg.tif_suffix
    path = cfg.temp.temporary_file_path(name_suffix=out_extension)
    if expression_out_path is not None:
        suffix = files_directories.file_extension(expression_out_path)
        cfg.logger.log.debug('suffix: %s' % suffix)
        if files_directories.is_directory(expression_out_path):
            path = shared_tools.join_path(
                expression_out_path, '{}{}'.format(
                    expression_output_name, out_extension
                )
            ).replace('\\', '/')
        elif len(suffix) > 0:
            path = shared_tools.join_path(
                files_directories.parent_directory(
                    expression_out_path
                ), '{}{}'.format(expression_output_name, out_extension)
            ).replace('\\', '/')
    elif output_path is not None:
        suffix = files_directories.file_extension(output_path)
        cfg.logger.log.debug('suffix: %s' % suffix)
        p_name = files_directories.file_name(output_path)
        if files_directories.is_directory(output_path):
            path = shared_tools.join_path(
                output_path, '{}{}'.format(
                    expression_output_name, out_extension
                )
            ).replace('\\', '/')
        elif len(suffix) > 0:
            if suffix.lower() == cfg.vrt_suffix:
                out_extension = cfg.vrt_suffix
            else:
                out_extension = cfg.tif_suffix
            path = shared_tools.join_path(
                files_directories.parent_directory(output_path), '{}{}'.format(
                    p_name, out_extension
                )
            ).replace('\\', '/')
    # save in temporary directory
    else:
        path = cfg.temp.temporary_file_path(
            name_suffix=cfg.vrt_suffix, name=expression_output_name
        )
        virtual = True
    out_name = files_directories.file_name(path)
    cfg.logger.log.debug('path: %s' % path)
    return path, out_name, virtual


def _band_names_alias(
        input_raster_list, input_name_list, bandset_catalog: BandSetCatalog,
        bandset_number=None
) -> dict:
    """Gets band names alias.

    Gets band names alias for calculation.

    Args:
        bandset_number: optional number of BandSet as current one.
        input_raster_list: list of raster paths or list of lists path,
            name ignoring input_name_list.
        input_name_list: list of raster names for calculation.
    """
    band_names = {}
    # raster files
    if input_raster_list is not None:
        for i in range(len(input_raster_list)):
            try:
                if type(input_raster_list[i]) is list:
                    band_names['%s%s%s' % (
                        cfg.variable_band_quotes, input_raster_list[i][1],
                        cfg.variable_band_quotes)
                               ] = str(input_raster_list[i][0])
                else:
                    band_names['%s%s%s' % (
                        cfg.variable_band_quotes, input_name_list[i],
                        cfg.variable_band_quotes)
                               ] = str(input_raster_list[i])
            except Exception as err:
                cfg.logger.log.error(str(err))
                cfg.messages.error(str(err))
                break
    # BandSet bands
    if bandset_catalog is not None:
        if bandset_number is None:
            bandset_number = bandset_catalog.current_bandset
        for i in range(1, bandset_catalog.get_bandset_count() + 1):
            bands = bandset_catalog.get(i).get_band_alias()
            apaths = bandset_catalog.get(i).get_absolute_paths()
            for b in range(bandset_catalog.get_band_count(i)):
                band_names['%s%s%s%s%s' % (
                    cfg.variable_band_quotes, cfg.variable_bandset_name, i,
                    bands[b], cfg.variable_band_quotes
                )] = apaths[b]
            # current BandSet
            if i == bandset_number:
                for b in range(len(bands)):
                    band_names['%s%s%s%s%s' % (
                        cfg.variable_band_quotes, cfg.variable_bandset_name,
                        cfg.variable_current_bandset, bands[b],
                        cfg.variable_band_quotes)] = apaths[b]
    cfg.logger.log.debug('band_names: %s' % (str(band_names)))
    return band_names


def _check_expression(
        expression_string, raster_variables=None,
        bandset_catalog: BandSetCatalog = None, bandset_number=None,
        output_dir_path=None
):
    """Checks expression.

    Checks expression for calculation.

    Args:
        expression_string: string of expressions
        raster_variables: raster output_name variable dictionary
        bandset_number: optional number of BandSet as current one
        output_dir_path: optional path of output directory to use as variables
    """
    cfg.logger.log.debug('start')
    raster_variables_dict = raster_variables.copy()
    output_message = None
    # short variable names
    at = cfg.variable_output_separator
    per = cfg.variable_bandset_number_separator
    bsn = cfg.variable_bandset_name
    bn = cfg.variable_band_name
    cb = cfg.variable_current_bandset
    # output output_name list
    all_out_name_list = []
    # expressions list
    exp_list = False
    if expression_string is None:
        output_message = '0: expressions none'
        cfg.logger.log.debug('end')
        return exp_list, all_out_name_list, output_message
    else:
        bandset_list = []
        if bandset_catalog is not None:
            if bandset_number is None:
                bandset_number = bandset_catalog.current_bandset
            bandset_list.append(bandset_number)
        expressions = expression_string.rstrip().split(cfg.new_line)
        counter0 = 1
        # check if iterator forbandsinbandset
        forbandsinbandset = False
        # check iterators
        try:
            first_line = None
            cfg.logger.log.debug('expressions: %s' % str(expressions))
            # comment lines starting with # character
            for line in expressions:
                if line.strip()[0] == '#':
                    counter0 += 1
                else:
                    first_line = line
                    break
            # BandSet iteration with structure forbandsets[x1:x2]name_filter
            # or forbandsets[x1,x2,x3]name_filter or date iteration with
            # structure forbandsets[YYYY-MM-DD:YYYY-MM-DD]name_filter or
            # forbandsets[YYYY-MM-DD, YYYY-MM-DD, YYYY-MM-DD, ...]name_filter
            # with name_filter optional filter of name of first band in the 
            # BandSet
            if cfg.forbandsets in first_line:
                cfg.logger.log.debug(cfg.forbandsets)
                bandset_list, output_message = _bandsets_iterator(
                    first_line, bandset_catalog
                )
                expressions.pop(0)
            # bands in BandSet iteration with structure
            # forbandsinbandset[x1:x2]name_filter
            # or forbandsinbandset[x1,x2,x3,...]name_filter or date iteration
            # with structure
            # forbandsinbandset[YYYY-MM-DD, YYYY-MM-DD, YYYY-MM-DD, ...]filter
            # with name_filter optional filter of name of first band in the 
            # BandSet
            elif cfg.forbandsinbandset in first_line:
                cfg.logger.log.debug(cfg.forbandsinbandset)
                bandset_list, output_message = _bandsets_iterator(
                    first_line, bandset_catalog
                )
                forbandsinbandset = True
                expressions.pop(0)
                counter0 = 1
        except Exception as err:
            cfg.logger.log.error(str(err))
            output_message = '0: %s' % expressions
        # check outputs
        if output_message is None:
            if bandset_catalog is not None:
                lines = []
                # replace BandSet number in expressions for iteration
                for bandset_x in bandset_list:
                    bs_x = bandset_catalog.get_bandset(bandset_x)
                    counter = counter0
                    for line_number in range(0, len(expressions)):
                        counter = counter + counter0 + 1
                        # skip comment lines starting with # character
                        if expressions[line_number].strip()[0] != '#':
                            output_bandset_number = None
                            date_string = dates_times.get_time_string()
                            line_split = expressions[line_number].split(at)
                            calculation = str(line_split[0])
                            # replace expression alias
                            for ex_alias in cfg.expression_alias:
                                calculation = shared_tools.replace(
                                    calculation, '%s%s%s' % (
                                        cfg.variable_band_quotes, ex_alias[0],
                                        cfg.variable_band_quotes), ex_alias[1]
                                )
                            # output variables after 
                            # variable_output_separator at the end of the line
                            output_name = None
                            output_path = None
                            if len(line_split) > 0:
                                # output variables:
                                # output path after first 
                                # variable_output_separator and output name 
                                # after the second
                                if len(line_split) == 3:
                                    output_name = \
                                        expressions[line_number].split(at)[
                                            2].strip()
                                    output_path = \
                                        expressions[line_number].split(at)[1]
                                    # output variable path in the same 
                                    # directory as the first band of the 
                                    # BandSet
                                    if (cfg.variable_output_name_bandset
                                            in output_path):
                                        try:
                                            output_path = (
                                                files_directories.
                                                parent_directory(
                                                    bs_x.get_absolute_paths()[
                                                        0]
                                                )
                                            )
                                        except Exception as err:
                                            cfg.logger.log.error(str(err))
                                    # output variable path in temporary 
                                    # directory
                                    elif cfg.variable_output_temporary == \
                                            output_path.lower():
                                        output_path = cfg.temp.dir
                                # output output_name after first 
                                # variable_output_separator
                                elif len(line_split) == 2:
                                    output_path = None
                                    try:
                                        output_name = \
                                            expressions[line_number].split(at)[
                                                1].strip()
                                    except Exception as err:
                                        str(err)
                                        output_name = None
                                # output variable output_name BandSet
                                try:
                                    output_name = shared_tools.replace(
                                        output_name,
                                        cfg.variable_output_name_bandset,
                                        bs_x.name
                                    )
                                except Exception as err:
                                    str(err)
                                # output variable output_name date
                                try:
                                    output_name = shared_tools.replace(
                                        output_name,
                                        cfg.variable_output_name_date,
                                        date_string
                                    )
                                except Exception as err:
                                    str(err)
                                # add output to BandSet number defined after
                                # variable_bandset_number_separator
                                try:
                                    output_name, output_bandset_number = (
                                        output_name.split(per))
                                    output_name = output_name.strip()
                                except Exception as err:
                                    str(err)
                                cfg.logger.log.debug(
                                    'output_path: %s; output_name: %s; '
                                    'output_bandset_number: %s'
                                    % (str(output_path), str(output_name),
                                       str(output_bandset_number))
                                )
                                # input variables
                                try:
                                    calculation = shared_tools.replace(
                                        calculation,
                                        cfg.variable_output_name_bandset,
                                        bs_x.name
                                    )
                                except Exception as err:
                                    str(err)
                                # spectral bands alias
                                if (cfg.variable_blue_name in calculation
                                        or cfg.variable_green_name in
                                        calculation
                                        or cfg.variable_red_name in calculation
                                        or cfg.variable_nir_name in calculation
                                        or cfg.variable_swir1_name in
                                        calculation
                                        or cfg.variable_swir2_name in
                                        calculation):
                                    (blue_band, green_band, red_band, nir_band,
                                     swir_1_band,
                                     swir_2_band) = bs_x.spectral_range_bands()
                                    spectral_bands = [
                                        [cfg.variable_blue_name, blue_band],
                                        [cfg.variable_green_name, green_band],
                                        [cfg.variable_red_name, red_band],
                                        [cfg.variable_nir_name, nir_band],
                                        [cfg.variable_swir1_name, swir_1_band],
                                        [cfg.variable_swir2_name, swir_2_band]]
                                    for spectral_band in spectral_bands:
                                        if spectral_band[0] in calculation:
                                            try:
                                                calculation = \
                                                    shared_tools.replace(
                                                        calculation,
                                                        spectral_band[0],
                                                        '%s%s%s%s' % (
                                                            bsn,
                                                            str(bandset_x),
                                                            bn, str(
                                                                spectral_band[
                                                                    1]
                                                            )
                                                        )
                                                    )
                                            except Exception as err:
                                                cfg.logger.log.error(str(err))
                                                output_message = '%s: %s' % (
                                                    str(counter),
                                                    str(spectral_band[0]))
                                # current BandSet
                                if '%s%s%s' % (bsn, cb, bn) in calculation:
                                    calculation = shared_tools.replace(
                                        calculation, '%s%s' % (bsn, cb),
                                                     '%s%s' % (
                                                         bsn, str(bandset_x))
                                    )
                                # new line creation
                                new_line = None
                                # create band expressions in 
                                # forbandsinbandset iteration
                                if (cfg.variable_band in calculation
                                        and forbandsinbandset):
                                    for band_x in range(
                                            1, bs_x.get_band_count() + 1
                                    ):
                                        if (output_bandset_number is not None
                                                and output_path is not None
                                                and output_name is not None):
                                            calculation += \
                                                ' %s%s%s%s%s%s%s' % (
                                                    at, output_path,
                                                    at, str(band_x),
                                                    output_name, per,
                                                    output_bandset_number.
                                                    replace(cb, str(bandset_x))
                                                )
                                        elif (output_path is not None
                                              and output_name is not None):
                                            calculation += ' %s%s%s%s%s' % (
                                                at, output_path, at,
                                                str(band_x), output_name)
                                        elif (output_name is not None
                                              and output_bandset_number is not
                                              None):
                                            calculation += ' %s%s%s%s%s' % (
                                                at, str(band_x), output_name,
                                                per,
                                                output_bandset_number.replace(
                                                    cb, str(bandset_x)
                                                )
                                            )
                                        elif output_name is not None:
                                            calculation += ' %s%s%s' % (
                                                at, str(band_x), output_name)
                                        calculation = shared_tools.replace(
                                            calculation, cfg.variable_band,
                                            '%s%s%s%s' % (
                                                bsn, str(bandset_x), bn,
                                                str(band_x)
                                            )
                                        )
                                        lines.append(calculation)
                                # compose new line expression
                                else:
                                    if (output_bandset_number is not None
                                            and output_path is not None
                                            and output_name is not None):
                                        new_line = '%s %s%s%s%s%s%s' % (
                                            calculation, at, output_path, at,
                                            output_name, per,
                                            output_bandset_number)
                                    elif (output_path is not None
                                          and output_name is not None):
                                        new_line = '%s %s%s%s%s' % (
                                            calculation, at, output_path,
                                            at, output_name)
                                    elif (output_name is not None
                                          and output_bandset_number is not
                                          None):
                                        new_line = '%s %s%s%s%s' % (
                                            calculation, at, output_name,
                                            per, output_bandset_number.replace(
                                                cb, str(bandset_x)
                                            )
                                        )
                                    elif output_name is not None:
                                        new_line = '%s %s%s' % (
                                            calculation, at, output_name)
                                    else:
                                        new_line = calculation
                                if new_line is not None:
                                    lines.append(new_line)
                expressions = lines
                cfg.logger.log.debug('expressions: %s' % str(expressions))
        else:
            cfg.logger.log.error(str(output_message))
            cfg.logger.log.debug('end')
            return exp_list, all_out_name_list, output_message
        # build expression list
        if output_message is None:
            counter = counter0
            # output number counter
            output_number = 0
            # expressions list
            exp_list = []
            # build expressions for process
            for line in expressions:
                cfg.logger.log.debug('line: %s' % str(line))
                counter += counter0 + 1
                output_name = None
                out_path = None
                bs_number = None
                virtual = False
                # output path and output name after variable_output_separator
                line_split = line.split(at)
                if len(line_split) == 3:
                    try:
                        output_name = line_split[2].strip()
                        out_path = line_split[1].strip()
                    except Exception as err:
                        cfg.logger.log.error(str(err))
                        output_message = '%s: %s' % (str(counter), str(line))
                        exp_list = False
                        break
                elif len(line_split) == 2:
                    try:
                        output_name = line_split[1].strip()

                        output_ext = files_directories.file_extension(
                            output_name.lower()
                        )
                        # check extension
                        if output_ext == cfg.tif_suffix:
                            extension = None
                        elif output_ext == cfg.vrt_suffix:
                            extension = None
                            virtual = True
                        else:
                            extension = cfg.vrt_suffix
                            virtual = True
                        if output_dir_path is not None:
                            if files_directories.is_directory(output_dir_path):
                                out_path = '%s/%s%s' % (
                                    output_dir_path, output_name, extension
                                ).replace('//', '/')
                            else:
                                out_path = str('%s/%s%s' % (
                                    cfg.temp.dir, output_name, extension
                                )).replace('//', '/')
                        else:
                            out_path = str('%s/%s%s' % (
                                cfg.temp.dir, output_name, extension
                            )).replace('//', '/')
                    except Exception as err:
                        cfg.logger.log.error(str(err))
                        output_message = '%s: %s' % (str(counter), str(line))
                        exp_list = False
                        break
                # variable output output name current bandset
                if (output_name is not None
                        and cfg.variable_output_name_bandset in output_name):
                    try:
                        b_name = bandset_catalog.get(bandset_number, 'name')
                        output_name = shared_tools.replace(
                            output_name, cfg.variable_output_name_bandset,
                            b_name
                        )
                    except Exception as err:
                        cfg.logger.log.error(str(err))
                        output_message = '%s: %s' % (str(counter), str(line))
                        exp_list = False
                        break
                # variable output output_name date
                if (output_name is not None
                        and cfg.variable_output_name_date in output_name):
                    try:
                        date_string = dates_times.get_time_string()
                        output_name = shared_tools.replace(
                            output_name, cfg.variable_output_name_date,
                            date_string
                        )
                    except Exception as err:
                        cfg.logger.log.error(str(err))
                        output_message = '%s: %s' % (str(counter), str(line))
                        exp_list = False
                        break
                # check output names
                output_number += 1
                if output_name is not None:
                    try:
                        output_name, bs_number = output_name.split(per)
                        output_name = output_name.strip()
                    except Exception as err:
                        str(err)
                    # virtual
                    output_ext = files_directories.file_extension(
                        output_name.lower()
                    )
                    if output_ext == cfg.vrt_suffix:
                        virtual = True
                    # tif
                    if (output_ext == cfg.tif_suffix
                            or output_ext == cfg.vrt_suffix):
                        output_name = files_directories.file_name(
                            output_name, False
                        )
                    all_out_name_list.append(output_name)
                else:
                    # output default name
                    output_name = '%s%s' % (
                        cfg.default_output_name, str(output_number))
                if out_path is not None:
                    raster_variables_dict['%s%s%s' % (
                        cfg.variable_band_quotes, output_name,
                        cfg.variable_band_quotes)] = out_path
                # replace operators
                if bandset_catalog is None:
                    expr, errors = line, None
                else:
                    expr, errors = _replace_operator_names(
                        line, bandset_catalog, bandset_number
                    )
                cfg.logger.log.debug(
                    'output_name: %s; expr: %s'
                    % (str(output_name), str(expr))
                )
                if errors is not None:
                    cfg.logger.log.error(str(errors))
                    output_message = '%s: %s' % (str(counter), str(line))
                    exp_list = False
                    break
                # get expression function
                expr_function, input_rasters = _expression_to_function(
                    expr, raster_variables_dict
                )
                # function variables
                if cfg.calc_function_name in expr_function:
                    exp_list.append(
                        [expr, expr_function, output_name, bs_number, out_path,
                         virtual, input_rasters]
                    )
                # no variables
                elif expr == expr_function:
                    output_message = '%s: %s' % (str(counter), str(line))
                    cfg.logger.log.error(str(line))
                    exp_list = False
                    break
                # check valid function
                else:
                    try:
                        _check_numpy_operators(
                            expr_function.split(at)[
                                0], len(input_rasters)
                        )
                    except Exception as err:
                        cfg.logger.log.error(str(err))
                        output_message = '%s: %s' % (str(counter), str(line))
                        exp_list = False
                        break
                if output_message is None:
                    exp_list.append(
                        [expr, expr_function, output_name, bs_number, out_path,
                         virtual, input_rasters]
                    )
        if exp_list is not False:
            cfg.logger.log.debug(
                'end; len(exp_list): %s; all_out_name_list: %s; '
                'output_message: %s'
                % (str(len(exp_list)), str(all_out_name_list),
                   str(output_message))
            )
        return exp_list, all_out_name_list, output_message


def _bandsets_iterator(expression, bandset_catalog):
    """Band sets iterator.

    :param expression: expression string
    """
    if cfg.forbandsets in expression or cfg.forbandsinbandset in expression:
        expression = shared_tools.replace(expression, cfg.forbandsets, '')
        expression = shared_tools.replace(
            expression, cfg.forbandsinbandset,
            ''
        )
        date_list = []
        date_range_list = []
        bandset_list = []
        bandset_list_arg = []
        first_line = ''
        output_message = None
        # find BandSet name filter
        try:
            first_line_split = expression.split(']')
            filter_bandset_name = first_line_split[1].strip()
            if len(filter_bandset_name) == 0:
                first_line = first_line_split[0]
                bandset_filter = None
            else:
                first_line = first_line_split[0]
                bandset_filter = filter_bandset_name.split(',')
        except Exception as err:
            str(err)
            bandset_filter = None
        bandsets_arg = first_line.replace('[', '').replace(']', '')
        split_bandsets_arg = bandsets_arg.split(',')
        for x in split_bandsets_arg:
            split_x = x.split(':')
            # list of ranges of dates
            if len(split_x) > 1:
                try:
                    # range of dates
                    date_range_list.append(
                        [
                            np.datetime64(
                                str(
                                    datetime.datetime.strptime(
                                        split_x[0].strip(),
                                        cfg.calc_date_format
                                    ).date()
                                )
                            ),
                            np.datetime64(
                                str(
                                    datetime.datetime.strptime(
                                        split_x[1].strip(),
                                        cfg.calc_date_format
                                    ).date()
                                )
                            )]
                    )
                except Exception as err:
                    str(err)
                    try:
                        # range of band sets
                        bandset_list_arg.extend(
                            list(range(int(split_x[0]), int(split_x[1]) + 1))
                        )
                    except Exception as err:
                        str(err)
                        output_message = '1: %s' % str(bandsets_arg)
            # list
            else:
                try:
                    # list of dates
                    datetime.datetime.strptime(
                        split_x[0].strip(), cfg.calc_date_format
                    )
                    date_list.append(split_x[0].strip())
                except Exception as err:
                    str(err)
                    # list of band sets
                    try:
                        bandset_list_arg.append(int(split_x[0]))
                    except Exception as err:
                        str(err)
                        output_message = '1: %s' % str(bandsets_arg)
        # list of band sets
        if bandset_filter is None:
            bandset_list = bandset_list_arg.copy()
        else:
            for j in bandset_list_arg:
                band_name = bandset_catalog.bandsets_table['bandset_name'][
                    bandset_catalog.bandsets_table['bandset_number'] == j]
                if len(band_name) > 0:
                    try:
                        for filter_bs in bandset_filter:
                            if filter_bs.lower() in band_name[0].lower():
                                bandset_list.append(j)
                                break
                    except Exception as err:
                        str(err)
        # list of dates
        if len(date_list) > 0:
            for date_x in date_list:
                bandset_number = \
                    bandset_catalog.bandsets_table['bandset_number'][
                        bandset_catalog.bandsets_table[
                            'date'] == datetime.datetime.strptime(
                            date_x, cfg.calc_date_format
                        ).date()]
                if len(bandset_number) > 0:
                    if bandset_filter is None:
                        bandset_list.extend(bandset_number)
                    else:
                        for bandset_n in bandset_number:
                            bandset_n_name = \
                                bandset_catalog.bandsets_table['bandset_name'][
                                    bandset_catalog.bandsets_table[
                                        'bandset_number'] == bandset_n]
                            for filter_x in bandset_filter:
                                if filter_x.lower() in \
                                        bandset_n_name[0].lower():
                                    bandset_list.append(bandset_n)
                                    break
        if len(date_range_list) > 0:
            # range of dates
            for date_range_x in date_range_list:
                bandset_number = \
                    bandset_catalog.bandsets_table['bandset_number'][
                        (bandset_catalog.bandsets_table['date'] >=
                         date_range_x[
                             0]) & (
                                bandset_catalog.bandsets_table['date'] <=
                                date_range_x[1])]
                if len(bandset_number) > 0:
                    if bandset_filter is None:
                        bandset_list.extend(bandset_number)
                    else:
                        for bandset_n in bandset_number:
                            bandset_n_name = \
                                bandset_catalog.bandsets_table['bandset_name'][
                                    bandset_catalog.bandsets_table[
                                        'bandset_number'] == bandset_n]
                            for filter_x in bandset_filter:
                                if filter_x.lower() \
                                        in bandset_n_name[0].lower():
                                    bandset_list.append(bandset_n)
                                    break
        cfg.logger.log.debug(
            'bandset_filter: %s; bandsets_arg: %s; bandset_list: %s'
            % (str(bandset_filter), str(bandsets_arg), str(bandset_list))
        )
        return list(set(bandset_list)), output_message


def _expression_to_function(expression, raster_variables: dict):
    """Converts string to function and replace input variables.

    :param expression: expression string
    """
    expr_func = expression
    # find nodata values
    if 'nodata(' in expr_func:
        # find all non-greedy expression
        nodata_names = re.findall(r'nodata\(#?(.*?)#?\)', expr_func)
        for name in nodata_names:
            try:
                path = raster_variables[name]
                nd = raster_vector.raster_nodata_value(path)
            except Exception as err:
                str(err)
                nd = np.nan
            expr_func = shared_tools.replace(
                expr_func, 'nodata(%s)' % name, str(nd)
            )
            cfg.logger.log.debug('nd: %s' % str(nd))
    # dictionary of actual input paths
    input_rasters = {}
    layer_number = 0
    for k in raster_variables:
        if k in expr_func:
            input_rasters[k] = raster_variables[k]
            expr_func = expr_func.replace(
                k, ' %s[::, ::, %s] ' % (
                    cfg.array_function_placeholder, str(layer_number))
            )
            layer_number += 1
    cfg.logger.log.debug(
        'expr_func: %s; input_rasters: %s' % (
            str(expr_func), str(input_rasters))
    )
    return expr_func, input_rasters


def _replace_operator_names(
        expression, bandset_catalog: BandSetCatalog, bandset_number=None
):
    """Replaces operators for expressions.

    :param expression: expression string
    :param bandset_number: optional number of BandSet as current one
    """
    output_message = None
    if bandset_number is None:
        bandset_number = bandset_catalog.current_bandset
    # variable all bands in current BandSet e.g. bandset#b*
    var_current_bandset = '%s%s%s%s%s%s' % (
        cfg.variable_band_quotes, cfg.variable_bandset_name,
        cfg.variable_current_bandset, cfg.variable_band_name, cfg.variable_all,
        cfg.variable_band_quotes)
    if var_current_bandset in expression:
        band_list = '['
        for band in range(
                1, bandset_catalog.get_band_count(bandset_number) + 1
        ):
            band_list += '%s%s%s%s%s%s, ' % (
                cfg.variable_band_quotes, cfg.variable_bandset_name,
                str(bandset_number),
                cfg.variable_band_name, str(band), cfg.variable_band_quotes)
        # percentile
        percentiles = re.findall(
            r'percentile\(#?(.*?)#?\)', expression.replace(' ', '')
        )
        for percentile_x in percentiles:
            if var_current_bandset in percentile_x:
                per_x_split = percentile_x.split(',')
                try:
                    expression = expression.replace(
                        percentile_x, '%s],%s, axis = 0' % (
                            band_list[:-2], per_x_split[1])
                    )
                except Exception as err:
                    str(err)
        band_list = '%s], axis = 0' % band_list[:-2]
        expression = shared_tools.replace(
            expression, var_current_bandset, band_list
        )
        cfg.logger.log.debug('expression: %s' % str(expression))
    # variable band number in all band sets e.g. bandset*b1
    elif '%s%s%s%s' % (
            cfg.variable_band_quotes, cfg.variable_bandset_name,
            cfg.variable_all,
            cfg.variable_band_name) in expression:
        band_numbers = re.findall(
            cfg.variable_bandset_name + r'\*' + cfg.variable_band_name +
            '#?(.*?)#?"',
            expression
        )
        for parts in band_numbers:
            try:
                num_b = int(parts)
            except Exception as err:
                cfg.logger.log.error(str(err))
                output_message = str(err)
                break
            band_list = '['
            for n in range(1, bandset_catalog.get_bandset_count() + 1):
                # if band in BandSet
                if num_b <= bandset_catalog.get_band_count(n):
                    band_list += '%s%s%s%s%s%s, ' % (
                        cfg.variable_band_quotes, cfg.variable_bandset_name,
                        str(n), cfg.variable_band_name,
                        str(num_b), cfg.variable_band_quotes)
            # percentile
            percentiles = re.findall(r'percentile\(#?(.*?)#?\)', expression)
            for percentile_x in percentiles:
                if '%s%s%s%s' % (
                        cfg.variable_band_quotes, cfg.variable_bandset_name,
                        cfg.variable_all,
                        cfg.variable_band_name) in percentile_x:
                    per_x_split = percentile_x.split(',')
                    try:
                        expression = expression.replace(
                            percentile_x, '%s],%s, axis = 0' % (
                                band_list[:-2], per_x_split[1])
                        )
                    except Exception as err:
                        str(err)
            band_list = '%s], axis = 0' % band_list[:-2]
            expression = shared_tools.replace(
                expression, '%s%s%s%s%s%s' % (
                    cfg.variable_band_quotes, cfg.variable_bandset_name,
                    cfg.variable_all, cfg.variable_band_name,
                    str(num_b), cfg.variable_band_quotes), band_list
            )
        cfg.logger.log.debug('expression: %s' % str(expression))
    # variable band number in BandSet list or range e.g. bandset{1,2,3}b1
    elif '%s%s{' % (
            cfg.variable_band_quotes, cfg.variable_bandset_name) in expression:
        band_numbers = re.findall(
            cfg.variable_bandset_name + '{(.*?)\"', expression
        )
        band_set_list = []
        for parts in band_numbers:
            part_split = parts.split('}b')
            try:
                num_b = int(part_split[1])
            except Exception as err:
                cfg.logger.log.error(str(err))
                output_message = str(err)
                break
            bandset_arg = part_split[0]
            date_list = []
            date_range_list = []
            bandset_number_list = []
            if ':' in bandset_arg and ',' in bandset_arg:
                # list of ranges of dates
                try:
                    lrg = bandset_arg.split(',')
                    for g in lrg:
                        try:
                            # range of numbers
                            rg = g.split(':')
                            for r in range(
                                    int(rg[0].strip()), int(rg[1].strip()) + 1
                            ):
                                bandset_number_list.append(r)
                        except Exception as err:
                            str(err)
                            try:
                                # range of dates
                                rg = g.split(':')
                                date_range_list.append(
                                    [
                                        np.datetime64(
                                            str(
                                                datetime.datetime.strptime(
                                                    rg[0].strip(),
                                                    cfg.calc_date_format
                                                )
                                            )
                                        ),
                                        np.datetime64(
                                            str(
                                                datetime.datetime.strptime(
                                                    rg[1].strip(),
                                                    cfg.calc_date_format
                                                )
                                            )
                                        )
                                    ]
                                )
                            except Exception as err:
                                str(err)
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    output_message = str(err)
                    break
            else:
                try:
                    try:
                        # range of numbers
                        rg = bandset_arg.split(':')
                        for r in range(
                                int(rg[0].strip()), int(rg[1].strip()) + 1
                        ):
                            bandset_number_list.append(r)
                    except Exception as err:
                        str(err)
                        # range of dates
                        rg = bandset_arg.split(':')
                        date_range_list.append(
                            [
                                np.datetime64(
                                    str(
                                        datetime.datetime.strptime(
                                            rg[0].strip(), cfg.calc_date_format
                                        ).date()
                                    )
                                ),
                                np.datetime64(
                                    str(
                                        datetime.datetime.strptime(
                                            rg[1].strip(), cfg.calc_date_format
                                        ).date()
                                    )
                                )]
                        )
                except Exception as err:
                    str(err)
                    # list of dates
                    try:
                        rg = bandset_arg.split(',')
                        for r in rg:
                            try:
                                # number of BandSet
                                bandset_number_list.append(int(r.strip()))
                            except Exception as err:
                                str(err)
                                # date of band sets
                                date_list.append(r.strip())
                    except Exception as err:
                        cfg.logger.log.error(str(err))
                        output_message = str(err)
                        break
            # number of BandSet
            if len(bandset_number_list) > 0:
                band_set_list = bandset_number_list
            # date of BandSet
            elif len(date_list) > 0:
                for j in range(bandset_catalog.get_bandset_count()):
                    if bandset_catalog.get_date(j + 1) in date_list:
                        band_set_list.append(j + 1)
            # range of dates of BandSet
            else:
                for j in range(bandset_catalog.get_bandset_count()):
                    try:
                        b_date = bandset_catalog.bandsets_table['date'][
                            bandset_catalog.bandsets_table[
                                'bandset_number'] == (j + 1)][0]
                        for dStr in date_range_list:
                            if (b_date >= dStr[0]) & (b_date <= dStr[1]):
                                band_set_list.append(j + 1)
                                break
                    except Exception as err:
                        str(err)
            band_list = '['
            for n in band_set_list:
                try:
                    # if band in BandSet
                    if num_b <= bandset_catalog.get_band_count(n):
                        band_list += '%s%s%s%s%s%s, ' % (
                            cfg.variable_band_quotes,
                            cfg.variable_bandset_name, str(n),
                            cfg.variable_band_name, str(num_b),
                            cfg.variable_band_quotes)
                except Exception as err:
                    str(err)
            # percentile
            percentiles = re.findall(r'percentile\((.*?)\)', expression)
            for percentile_x in percentiles:
                if '%s%s{%s' % (
                        cfg.variable_band_quotes, cfg.variable_bandset_name,
                        parts) in percentile_x:
                    per_x_split = percentile_x.split('",')
                    try:
                        expression = expression.replace(
                            percentile_x,
                            '%s],%s, axis = 0' % (
                                band_list[:-2], per_x_split[1])
                        )
                    except Exception as err:
                        str(err)
            band_list = '%s], axis = 0' % band_list[:-2]
            expression = shared_tools.replace(
                expression, '%s%s{%s%s' % (
                    cfg.variable_band_quotes, cfg.variable_bandset_name, parts,
                    cfg.variable_band_quotes),
                band_list
            )
        cfg.logger.log.debug('expression: %s' % str(expression))
    # variable all bands in a BandSet e.g. bandset1b*
    elif cfg.variable_bandset_name in expression and '%s%s%s' % (
            cfg.variable_band_name, cfg.variable_all,
            cfg.variable_band_quotes) in expression:
        for n in range(1, bandset_catalog.get_bandset_count() + 1):
            band_list = '['
            if '%s%s%s%s%s%s' % (
                    cfg.variable_band_quotes, cfg.variable_bandset_name,
                    str(n), cfg.variable_band_name,
                    cfg.variable_all, cfg.variable_band_quotes) in expression:
                for band in range(1, bandset_catalog.get_band_count(n) + 1):
                    band_list += '%s%s%s%s%s%s, ' % (
                        cfg.variable_band_quotes, cfg.variable_bandset_name,
                        str(n), cfg.variable_band_name,
                        str(band), cfg.variable_band_quotes)
                # percentile
                percentiles = re.findall(
                    r'percentile\(#?(.*?)#?\)', expression
                )
                for percentile_x in percentiles:
                    if '%s%s' % (
                            cfg.variable_band_quotes,
                            cfg.variable_bandset_name) in percentile_x and \
                            '%s%s%s' % (
                            cfg.variable_band_name, cfg.variable_all,
                            cfg.variable_band_quotes) in percentile_x:
                        per_x_split = percentile_x.split(',')
                        try:
                            expression = expression.replace(
                                percentile_x,
                                '%s],%s, axis = 0' % (
                                    band_list[:-2], per_x_split[1])
                            )
                        except Exception as err:
                            str(err)
                band_list = '%s], axis = 0' % band_list[:-2]
                expression = shared_tools.replace(
                    expression, '%s%s%s%s%s%s' % (
                        cfg.variable_band_quotes, cfg.variable_bandset_name,
                        str(n), cfg.variable_band_name,
                        cfg.variable_all, cfg.variable_band_quotes), band_list
                )
        cfg.logger.log.debug('expression: %s' % str(expression))
    return expression, output_message


def _check_numpy_operators(expression, layer_number: int):
    """Checks expression using numpy operators.

    :param expression: expression string
    """
    # expose numpy functions
    log = np.log
    _log = log
    log10 = np.log10
    _log10 = log10
    sqrt = np.sqrt
    _sqrt = sqrt
    cos = np.cos
    _cos = cos
    arccos = np.arccos
    _arccos = arccos
    sin = np.sin
    _sin = sin
    arcsin = np.arcsin
    _arcsin = arcsin
    tan = np.tan
    _tan = tan
    arctan = np.arctan
    _arctan = arctan
    exp = np.exp
    _exp = exp
    min = np.nanmin
    _min = min
    max = np.nanmax
    _max = max
    sum = np.nansum
    _sum = sum
    percentile = np.nanpercentile
    _percentile = percentile
    median = np.nanmedian
    _median = median
    mean = np.nanmean
    _mean = mean
    std = np.nanstd
    _std = std
    where = np.where
    _where = where
    nan = np.nan
    _nan = nan
    # check expression
    expression = expression.replace(
        cfg.array_function_placeholder, '_array_function_placeholder'
    )
    cfg.logger.log.debug('layer_number: %s' % str(layer_number))
    size = layer_number * 5 * 5
    _array_function_placeholder = np.arange(size).reshape((5, 5, layer_number))
    eval(expression)


def _calculate_bandset(
        input_bands: BandSet, expression_string: str, output_path: str,
        n_processes: Optional[int] = None,
        available_ram: Optional[int] = None,
        align_raster: Optional[str] = None,
        extent_raster: Optional[str] = None,
        extent_list: Optional[list] = None,
        extent_intersection: Optional[bool] = True,
        xy_resolution_list: Optional[list] = None,
        input_nodata_as_value: Optional[bool] = None,
        use_value_as_nodata: Optional[int] = None,
        output_nodata: Optional[int] = None,
        output_datatype: Optional[str] = None,
        use_scale: Optional[float] = None,
        use_offset: Optional[float] = None,
        calc_datatype: Optional[str] = None,
        any_nodata_mask: Optional[bool] = False
) -> OutputManager:
    """Performs band calculation using BandSet as input.

    Calculation is defined by an expression string using variable names referred to BandSet.

    Args:
        input_bands: BandSet object.
        output_path: path of output file for single expression or 
            path to a directory for multiple expression outputs.
        expression_string: expression string used for calculation; multiple 
            expressions can be entered separated by new line.
        n_processes: number of threads for calculation.
        available_ram: number of megabytes of RAM available to processes.
        align_raster: string path of raster used for aligning output pixels and projections.
        extent_raster: string path of raster used for extent reference.
        extent_list: list of coordinates for defining calculation extent 
            [left, top, right, bottom] in the same coordinates as the reference raster.
        extent_intersection: if True the output extent is geometric 
            intersection of input raster extents, if False the output extent 
            is the maximum extent from union of input raster extents.
        xy_resolution_list: list of [x, y] pixel resolution.
        input_nodata_as_value: if True then unmask the value of nodata pixels 
            in calculations, if False then mask nodata pixels in calculations.
        use_value_as_nodata: use integer value as nodata in calculation.
        output_nodata: integer value used as nodata in output raster.
        output_datatype: string of data type for output raster such as 
            Float64, Float32, Int32, UInt32, Int16, UInt16, or Byte.
        use_scale: float number used for scale for output.
        use_offset: float number used for offset for output.
        calc_datatype: data type used during calculation, which may differ 
            from output_datatype, such as Float64, Float32, Int32, UInt32, Int16, UInt16, or Byte.
        any_nodata_mask: if True then output nodata where any input is nodata, 
            if False then output nodata where all the inputs are nodata, 
            if None then do not apply nodata to output.

    Returns:
        :func:`~remotior_sensus.core.output_manager.OutputManager` object with
            - paths = [output raster paths]
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=True
    )
    cfg.logger.log.debug('input_bands: %s' % (str(input_bands)))
    # create list of band names from band sets
    raster_variables = _bandset_names_alias(bandset=input_bands)
    # check expression
    exp_list, all_out_name_list, output_message = _check_expression_bandset(
        expression_string, raster_variables, input_bands
    )
    if output_message is not None:
        cfg.logger.log.error('expression error: %s', output_message)
        return OutputManager(check=False, extra={'message': output_message})
    output_list = []
    # process calculation
    n = 0
    min_p = 1
    max_p = int((99 - 1) / len(exp_list))
    previous_output_list = []
    for e in exp_list:
        output, out_name = _run_expression(
            expression_list=e, output_path=output_path,
            previous_output_list=previous_output_list, n_processes=n_processes,
            available_ram=available_ram,
            extent_raster=extent_raster, align_raster=align_raster,
            extent_list=extent_list, extent_intersection=extent_intersection,
            xy_resolution_list=xy_resolution_list,
            input_nodata_as_value=input_nodata_as_value,
            use_value_as_nodata=use_value_as_nodata,
            output_nodata=output_nodata,
            output_datatype=output_datatype, use_scale=use_scale,
            use_offset=use_offset, calc_datatype=calc_datatype,
            nodata_mask=any_nodata_mask, min_progress=min_p + max_p * n,
            max_progress=min_p + max_p * (n + 1),
            progress_message='running calculation %s' % (n + 1)
        )
        output_list.append(output)
        previous_output_list.append([output, out_name])
        cfg.logger.log.debug('output: %s' % output)
        n += 1
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; band calc: %s', output_list)
    return OutputManager(paths=output_list)


def _bandset_names_alias(bandset: BandSet) -> dict:
    """Gets band names alias.

    Gets band names alias for calculation.

    Args:
        bandset: BandSet object
    """
    band_names = {}
    # BandSet bands
    bands = bandset.get_band_alias()
    apaths = bandset.get_absolute_paths()
    # current BandSet
    for b in range(len(bands)):
        band_names['%s%s%s' % (
            cfg.variable_band_quotes, bands[b], cfg.variable_band_quotes
        )] = apaths[b]
    (blue_band, green_band, red_band, nir_band, swir_1_band,
     swir_2_band) = bandset.spectral_range_bands(output_as_number=False)
    spectral_bands = [
        [cfg.variable_blue_name, blue_band],
        [cfg.variable_green_name, green_band],
        [cfg.variable_red_name, red_band], [cfg.variable_nir_name, nir_band],
        [cfg.variable_swir1_name, swir_1_band],
        [cfg.variable_swir2_name, swir_2_band]]
    for spectral_band in spectral_bands:
        try:
            band_names['%s%s%s' % (
                cfg.variable_band_quotes, spectral_band[0],
                cfg.variable_band_quotes
            )] = spectral_band[1].absolute_path
        except Exception as err:
            str(err)
    cfg.logger.log.debug('band_names: %s' % (str(band_names)))
    return band_names


def _check_expression_bandset(
        expression_string, raster_variables_dict, bandset: BandSet
):
    """Checks expression.

    Checks expression for calculation.

    Args:
        expression_string: string of expressions
        bandset: BandSet object
    """
    cfg.logger.log.debug('start')
    output_message = None
    # short variable names
    at = cfg.variable_output_separator
    per = cfg.variable_bandset_number_separator
    # output output_name list
    all_out_name_list = []
    # expressions list
    exp_list = False
    if expression_string is None:
        output_message = '0: expressions none'
        cfg.logger.log.debug('end')
        return exp_list, all_out_name_list, output_message
    else:
        cfg.logger.log.debug('expression_string: %s' % str(expression_string))
        output_bandset_number = output_name = output_path = new_line = None
        date_string = dates_times.get_time_string()
        line_split = expression_string.split(at)
        calculation = str(line_split[0])
        # output variables after variable_output_separator
        # at the end of the line
        if len(line_split) > 0:
            # output variables: output path after first
            # variable_output_separator and output name after the second
            if len(line_split) == 3:
                output_name = expression_string.split(at)[2].strip()
                output_path = expression_string.split(at)[1]
                # output variable path in the same directory as the first
                # band of the BandSet
                if cfg.variable_output_name_bandset in output_path:
                    try:
                        output_path = (files_directories.parent_directory(
                            bandset.get_absolute_paths()[0]
                        ))
                    except Exception as err:
                        cfg.logger.log.error(str(err))
                # output variable path in temporary directory
                elif cfg.variable_output_temporary == output_path.lower():
                    output_path = cfg.temp.dir
            # output output_name after first variable_output_separator
            elif len(line_split) == 2:
                output_path = None
                try:
                    output_name = expression_string.split(at)[1].strip()
                except Exception as err:
                    str(err)
                    output_name = None
            # output variable output_name BandSet
            try:
                output_name = shared_tools.replace(
                    output_name, cfg.variable_output_name_bandset, bandset.name
                )
            except Exception as err:
                str(err)
            # output variable output_name date
            try:
                output_name = shared_tools.replace(
                    output_name, cfg.variable_output_name_date, date_string
                )
            except Exception as err:
                str(err)
            # add output to BandSet number defined after
            # variable_bandset_number_separator
            try:
                output_name, output_bandset_number = (output_name.split(per))
                output_name = output_name.strip()
            except Exception as err:
                str(err)
            cfg.logger.log.debug(
                'output_path: %s; output_name: %s; output_bandset_number: %s'
                % (str(output_path), str(output_name),
                   str(output_bandset_number))
            )
            # input variables
            try:
                calculation = shared_tools.replace(
                    calculation, cfg.variable_output_name_bandset, bandset.name
                )
            except Exception as err:
                str(err)
            # compose new line expression
            if output_path is not None and output_name is not None:
                new_line = '%s %s%s%s%s' % (
                    calculation, at, output_path, at, output_name)
            elif output_name is not None:
                new_line = '%s %s%s' % (calculation, at, output_name)
            else:
                new_line = calculation
        cfg.logger.log.debug('new_line: %s' % str(new_line))
        # build expression list
        if new_line is not None:
            cfg.logger.log.debug('new_line: %s' % str(new_line))
            # output number counter
            output_number = 0
            # expressions list
            exp_list = []
            # replace expression alias
            for ex_alias in cfg.expression_alias:
                new_line = shared_tools.replace(
                    new_line, '%s%s%s' % (
                        cfg.variable_band_quotes, ex_alias[0],
                        cfg.variable_band_quotes), ex_alias[1]
                )
            # spectral bands alias
            if (cfg.variable_blue_name in new_line
                    or cfg.variable_green_name in new_line
                    or cfg.variable_red_name in new_line
                    or cfg.variable_nir_name in new_line
                    or cfg.variable_swir1_name in new_line
                    or cfg.variable_swir2_name in new_line):
                (blue_band, green_band, red_band, nir_band,
                 swir_1_band, swir_2_band) = bandset.spectral_range_bands()
                spectral_bands = [
                    [cfg.variable_blue_name, blue_band],
                    [cfg.variable_green_name, green_band],
                    [cfg.variable_red_name, red_band],
                    [cfg.variable_nir_name, nir_band],
                    [cfg.variable_swir1_name, swir_1_band],
                    [cfg.variable_swir2_name, swir_2_band]]
                for spectral_band in spectral_bands:
                    if spectral_band[0] in new_line:
                        try:
                            new_line = shared_tools.replace(
                                new_line, spectral_band[0],
                                '%s%s' % (cfg.variable_band_name,
                                          str(spectral_band[1])
                                          )
                            )
                        except Exception as err:
                            cfg.logger.log.error(str(err))
                            output_message = '%s' % str(spectral_band[0])
            output_name = out_path = bs_number = None
            # output path and output name after variable_output_separator
            line_split = new_line.split(at)
            if len(line_split) == 3:
                try:
                    output_name = line_split[2].strip()
                    out_path = line_split[1].strip()
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    output_message = '%s' % (str(new_line))
                    exp_list = False
            elif len(line_split) == 2:
                try:
                    output_name = line_split[1].strip()
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    output_message = '%s' % (str(new_line))
                    exp_list = False
            # variable output name current bandset
            if (output_name is not None
                    and cfg.variable_output_name_bandset in output_name):
                try:
                    output_name = shared_tools.replace(
                        output_name, cfg.variable_output_name_bandset,
                        bandset.name
                    )
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    output_message = '%s' % (str(new_line))
                    exp_list = False
            # variable output output_name date
            if (output_name is not None
                    and cfg.variable_output_name_date in output_name):
                try:
                    date_string = dates_times.get_time_string()
                    output_name = shared_tools.replace(
                        output_name, cfg.variable_output_name_date, date_string
                    )
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    output_message = '%s' % (str(new_line))
                    exp_list = False
            # check output names
            virtual = False
            output_number += 1
            if output_name is not None:
                try:
                    output_name, bs_number = output_name.split(per)
                    output_name = output_name.strip()
                except Exception as err:
                    str(err)
                # virtual
                output_ext = files_directories.file_extension(
                    output_name.lower()
                )
                if output_ext == cfg.vrt_suffix:
                    virtual = True
                # tif
                if (output_ext == cfg.tif_suffix
                        or output_ext == cfg.vrt_suffix):
                    output_name = files_directories.file_name(
                        output_name, False
                    )
                all_out_name_list.append(output_name)
            else:
                # output default name
                output_name = '%s%s' % (
                    cfg.default_output_name, str(output_number))
            # replace operators
            expr, errors = _replace_bandset_operator_names(new_line, bandset)
            cfg.logger.log.debug(
                'output_name: %s; expr: %s' % (str(output_name), str(expr))
            )
            if errors is not None:
                cfg.logger.log.error(str(errors))
                output_message = '%s' % (str(new_line))
                exp_list = False
            # get expression function
            expr_function, input_rasters = _expression_to_function(
                expr, raster_variables_dict
            )
            # function variables
            if cfg.calc_function_name in expr_function:
                exp_list.append(
                    [expr, expr_function, output_name, bs_number, out_path,
                     virtual, input_rasters]
                )
            # no variables
            elif expr == expr_function:
                output_message = '%s' % (str(new_line))
                cfg.logger.log.error(str(new_line))
                exp_list = False
            # check valid function
            else:
                try:
                    _check_numpy_operators(
                        expr_function.split(at)[
                            0], len(input_rasters)
                    )
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    output_message = '%s' % (str(new_line))
                    exp_list = False
            if output_message is None:
                exp_list.append(
                    [expr, expr_function, output_name, bs_number, out_path,
                     virtual, input_rasters]
                )
        if exp_list is not False:
            cfg.logger.log.debug(
                'end; len(exp_list): %s; all_out_name_list: %s; '
                'output_message: %s'
                % (str(len(exp_list)), str(all_out_name_list),
                   str(output_message))
            )
        return exp_list, all_out_name_list, output_message


def _replace_bandset_operator_names(expression, bandset: BandSet):
    """Replaces operators for expressions.

    :param expression: expression string
    :param bandset: BandSet object
    """
    output_message = None
    # variable all bands in BandSet e.g. "b*"
    if '%s%s%s%s' % (
            cfg.variable_band_quotes, cfg.variable_band_name, cfg.variable_all,
            cfg.variable_band_quotes) in expression:
        band_list = '['
        for band in range(1, bandset.get_band_count() + 1):
            band_list += '%s%s%s%s, ' % (
                cfg.variable_band_quotes, cfg.variable_band_name,
                str(band), cfg.variable_band_quotes)
        # percentile
        percentiles = re.findall(
            r'percentile\(#?(.*?)#?\)', expression
        )
        for percentile_x in percentiles:
            if '%s%s%s%s' % (cfg.variable_band_quotes,
                             cfg.variable_band_name, cfg.variable_all,
                             cfg.variable_band_quotes) in percentile_x:
                per_x_split = percentile_x.split(',')
                try:
                    expression = expression.replace(
                        percentile_x,
                        '%s],%s, axis = 0' % (
                            band_list[:-2], per_x_split[1])
                    )
                except Exception as err:
                    str(err)
        band_list = '%s], axis = 0' % band_list[:-2]
        expression = shared_tools.replace(
            expression,
            '%s%s%s%s' % (cfg.variable_band_quotes, cfg.variable_band_name,
                          cfg.variable_all, cfg.variable_band_quotes),
            band_list
        )
        cfg.logger.log.debug('expression: %s' % str(expression))
    return expression, output_message
