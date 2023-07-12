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
"""Band PCA.

This tool allows for the calculation of Principal Components Analysis on
input bands, producing rasters corresponding to the principal components.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> catalog = rs.bandset_catalog()
    >>> # create a BandSets
    >>> file_list_1 = ['file1_b1.tif', 'file1_b2.tif', 'file1_b3.tif']
    >>> catalog.create_bandset(file_list_1, bandset_number=1)
    >>> # start the process
    >>> output = rs.band_pca(input_bands=catalog.get_bandset(1),
    ... output_path='directory_path')
"""

import io
from typing import Union, Optional

import numpy as np

from remotior_sensus.core import configurations as cfg, table_manager as tm
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.core.processor_functions import (
    bands_covariance, calculate_pca, raster_pixel_count
)
from remotior_sensus.util import (
    shared_tools, files_directories, read_write_files
)


def band_pca(
        input_bands: Union[list, int, BandSet],
        output_path: Union[list, str] = None,
        overwrite: Optional[bool] = False,
        nodata_value: Optional[int] = None, extent_list: Optional[list] = None,
        n_processes: Optional[int] = None, available_ram: int = None,
        bandset_catalog: Optional[BandSetCatalog] = None,
        number_components: Optional[int] = None,
        progress_message: Optional[bool] = True
) -> OutputManager:
    """Calculation of Principal Components Analysis.

    This tool allows for the calculation of Principal Components Analysis of
    raster bands obtaining the principal components.
    A new raster file is created for each component. In addition, a table containing the Principal Components statistics is created.

    Args:
        input_bands: input of type BandSet or list of paths or
            integer number of BandSet.
        output_path: string of output path directory or list of paths.
        overwrite: if True, output overwrites existing files.
        nodata_value: value to be considered as nodata.
        extent_list: list of boundary coordinates left top right bottom.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        bandset_catalog: optional type BandSetCatalog for BandSet number.
        number_components: defines the maximum number of components calculated.
        progress_message: if True then start progress message, if False does not start the progress message (useful if launched from other tools).

    Returns:
        Object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - paths = output list
            - extra = {'table': table path string}

    Examples:
        Perform the PCA on the first BandSet
            >>> # import Remotior Sensus and start the session
            >>> import remotior_sensus
            >>> rs = remotior_sensus.Session()
            >>> catalog = rs.bandset_catalog()
            >>> # create a BandSets
            >>> file_list_1 = ['file1_b1.tif', 'file1_b2.tif', 'file1_b3.tif']
            >>> catalog.create_bandset(file_list_1, bandset_number=1)
            >>> # start the process
            >>> pca = rs.band_pca(input_bands=catalog.get_bandset(1),output_path='directory_path')
    """  # noqa: E501
    if progress_message:
        cfg.logger.log.info('start')
        cfg.progress.update(
            process=__name__.split('.')[-1].replace('_', ' '),
            message='starting', start=True
        )
    # prepare process files
    prepared = shared_tools.prepare_process_files(
        input_bands=input_bands, output_path=output_path, overwrite=overwrite,
        n_processes=n_processes, bandset_catalog=bandset_catalog,
        box_coordinate_list=extent_list
    )
    input_raster_list = prepared['input_raster_list']
    out_path = prepared['output_path']
    vrt_r = prepared['virtual_output']
    vrt_path = prepared['temporary_virtual_raster']
    n_processes = prepared['n_processes']
    if number_components is None or number_components > len(input_raster_list):
        number_components = len(input_raster_list)
    # list of band order
    band_order = list(range(len(input_raster_list)))
    cfg.multiprocess.run_separated(
        raster_path_list=input_raster_list, function=raster_pixel_count,
        function_argument=band_order, use_value_as_nodata=nodata_value,
        any_nodata_mask=True, n_processes=n_processes,
        available_ram=available_ram, keep_output_argument=True,
        progress_message='pixel count', min_progress=1, max_progress=20
    )
    cfg.multiprocess.get_dictionary_sum()
    band_dict = cfg.multiprocess.output
    # calculate mean
    max_pixel_count = 0
    try:
        for band_number in band_order:
            band_dict[
                'mean_%s' % band_number] = \
                band_dict['sum_%s' % band_number] / band_dict[
                    'count_%s' % band_number]
            max_pixel_count = max(
                max_pixel_count, band_dict['count_%s' % band_number]
            )
    except Exception as err:
        cfg.logger.log.error(str(err))
        cfg.messages.error(str(err))
        return OutputManager(check=False)
    # calculate covariance
    # dummy bands for memory calculation
    dummy_bands = 2
    cfg.multiprocess.run(
        raster_path=vrt_path, function=bands_covariance,
        function_argument=band_order,
        function_variable=band_dict, use_value_as_nodata=nodata_value,
        any_nodata_mask=True, dummy_bands=dummy_bands,
        keep_output_argument=True, virtual_raster=vrt_r,
        progress_message='calculate covariance',
        min_progress=21, max_progress=40
    )
    cfg.multiprocess.get_dictionary_sum()
    cov_dict = cfg.multiprocess.output
    # calculate covariance matrix
    cov_matrix = np.zeros((len(input_raster_list), len(input_raster_list)))
    # iterate bands
    for _x in band_order:
        for _y in band_order:
            # calculate covariance as [SUM((x - Mean_x) * (y - Mean_y))] / (
            # count - 1)
            cov_matrix[_x, _y] = cov_dict['cov_%s-%s' % (_x, _y)] / (
                    max_pixel_count - 1)
    # calculate sorted eigenvalues and eigenvectors
    eigenvalues, vect = np.linalg.eigh(cov_matrix)
    sorted_args = np.argsort(eigenvalues)
    components = []
    total_variance = []
    cumulative_total_variance = []
    sorted_eigenvalues = []
    for i in reversed(sorted_args):
        if len(components) > number_components:
            break
        sorted_eigenvalues.append(eigenvalues[i])
        components.append(vect[:, i])
        total_variance.append(eigenvalues[i] / eigenvalues.sum() * 100)
        cumulative_total_variance.append(np.sum(total_variance))
    # calculate correlation matrix
    correlation_matrix = np.corrcoef(cov_matrix)
    # create array of means for band normalization
    mean_array = np.zeros((len(input_raster_list)))
    for band_number in band_order:
        try:
            mean_array[band_number] = float(
                band_dict['mean_%s' % str(band_number)]
            )
        except Exception as err:
            cfg.logger.log.error(str(err))
    # run calculation for each component
    output_raster_path_list = []
    min_p = 41
    max_p = int((99 - 1) / number_components)
    for c in range(number_components):
        # check output path
        if vrt_r:
            extension = cfg.vrt_suffix
        else:
            extension = cfg.tif_suffix
        output = shared_tools.join_path(
            files_directories.parent_directory(out_path), '{}_pc{}{}'.format(
                files_directories.file_name(out_path, suffix=False), (c + 1),
                extension
            )
        ).replace('\\', '/')
        cfg.multiprocess.run(
            raster_path=vrt_path, function=calculate_pca,
            function_argument=components[c],
            function_variable=mean_array, calculation_datatype=np.float32,
            use_value_as_nodata=nodata_value, any_nodata_mask=True,
            output_raster_path=output,
            output_data_type=cfg.float32_dt, compress=cfg.raster_compression,
            virtual_raster=vrt_r,
            progress_message='calculating raster component %s' % str(c + 1),
            min_progress=min_p + max_p * c,
            max_progress=min_p + max_p * (c + 1)
        )
        output_raster_path_list.append(output)
    # create principal component table
    try:
        table = _pca_table(
            covariance_matrix=cov_matrix,
            correlation_matrix=correlation_matrix, eigenvectors=components,
            eigenvalues=sorted_eigenvalues, total_variance=total_variance,
            cumulative_total_variance=cumulative_total_variance
        )
    except Exception as err:
        cfg.logger.log.error(str(err))
        cfg.messages.error(str(err))
        return OutputManager(check=False)
    # save principal component details to table
    tbl_out = shared_tools.join_path(
        files_directories.parent_directory(out_path), '{}{}'.format(
            files_directories.file_name(out_path, suffix=False), cfg.csv_suffix
        )
    ).replace('\\', '/')
    read_write_files.write_file(table, tbl_out)
    if len(output_raster_path_list) == 0:
        cfg.logger.log.error('unable to process files')
        cfg.messages.error('unable to process files')
        return OutputManager(check=False)
    else:
        for i in output_raster_path_list:
            if not files_directories.is_file(i):
                cfg.logger.log.error('unable to process file: %s' % str(i))
                cfg.messages.error('unable to process file: %s' % str(i))
                return OutputManager(check=False)
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; pca: %s' % str(output_raster_path_list))
    return OutputManager(paths=output_raster_path_list, extra={'table': tbl_out})


def _pca_table(
        covariance_matrix, correlation_matrix, eigenvectors, eigenvalues,
        total_variance,
        cumulative_total_variance
):
    """Creates text for table."""
    cv = cfg.comma_delimiter
    nl = cfg.new_line
    text = ['Principal Components Analysis', nl, nl]
    # prepare fields
    dtype_list = []
    fields = ['Bands']
    # column names
    columns = ['Bands', cv]
    vector_columns = ['Bands', cv]
    band_numbers = list(range(1, covariance_matrix.shape[0] + 1))
    for i in band_numbers:
        dtype_list.append((str(i), 'float64'))
        fields.append(str(i))
        columns.append(str(i))
        columns.append(cv)
        vector_columns.append('Vector_%s' % str(i))
        vector_columns.append(cv)
    # remove last cv
    columns.pop(-1)
    vector_columns.pop(-1)
    columns.append(nl)
    vector_columns.append(nl)
    # prepare column text
    columns_text = ''.join(columns)
    vector_columns_text = ''.join(vector_columns)
    # get covariance
    covariance_table = np.rec.fromarrays(covariance_matrix, dtype=dtype_list)
    covariance_table = tm.append_field(
        covariance_table, 'Bands', np.array(band_numbers), 'int8'
    )
    covariance_table = tm.redefine_matrix_columns(
        matrix=covariance_table, input_field_names=fields,
        output_field_names=fields
    )
    # export table creating stream handler
    stream = io.StringIO()
    np.savetxt(stream, covariance_table, delimiter=cv, fmt='%1.4f')
    covariance_text = stream.getvalue()
    text.append('Covariance matrix')
    text.append(nl)
    text.append(columns_text)
    text.append(covariance_text)
    # get correlation
    correlation_table = np.rec.fromarrays(correlation_matrix, dtype=dtype_list)
    correlation_table = tm.append_field(
        correlation_table, 'Bands', np.array(band_numbers), 'int8'
    )
    correlation_table = tm.redefine_matrix_columns(
        matrix=correlation_table, input_field_names=fields,
        output_field_names=fields
    )
    # export table creating stream handler
    stream_2 = io.StringIO()
    np.savetxt(stream_2, correlation_table, delimiter=cv, fmt='%1.4f')
    correlation_text = stream_2.getvalue()
    text.append(nl)
    text.append('Correlation matrix')
    text.append(nl)
    text.append(columns_text)
    text.append(correlation_text)
    # get eigenvectors
    eigenvectors_array = np.zeros(covariance_matrix.shape)
    for v in range(len(eigenvectors)):
        eigenvectors_array[v, ::] = eigenvectors[v]
    eigenvectors_table = np.rec.fromarrays(
        eigenvectors_array, dtype=dtype_list
    )
    eigenvectors_table = tm.append_field(
        eigenvectors_table, 'Bands', np.array(band_numbers), 'int8'
    )
    eigenvectors_table = tm.redefine_matrix_columns(
        matrix=eigenvectors_table, input_field_names=fields,
        output_field_names=fields
    )
    # export table creating stream handler
    stream_3 = io.StringIO()
    np.savetxt(stream_3, eigenvectors_table, delimiter=cv, fmt='%1.4f')
    eigenvectors_text = stream_3.getvalue()
    text.append(nl)
    text.append('Eigenvectors')
    text.append(nl)
    text.append(vector_columns_text)
    text.append(eigenvectors_text)
    # get eigenvalues
    eigenvalues_text = []
    for e in range(len(eigenvalues)):
        eigenvalues_text.append(
            '%1.4f%s%1.4f%s%1.4f%s'
            % (eigenvalues[e], cv, total_variance[e], cv,
               cumulative_total_variance[e], nl)
        )
    text.append(nl)
    text.append('Eigenvalues')
    text.append(cv)
    text.append('Accounted variance')
    text.append(cv)
    text.append('Cumulative variance')
    text.append(nl)
    text.append(''.join(eigenvalues_text))
    joined_text = ''.join(text)
    return joined_text.replace('.0000', '')
