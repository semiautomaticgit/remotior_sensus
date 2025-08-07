# Remotior Sensus , software to process remote sensing and GIS data.
# Copyright (C) 2022-2025 Luca Congedo.
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
"""Band clustering.

This tool allows for the automatic clustering of a BandSet or a list of files.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> catalog = rs.bandset_catalog()
    >>> # create a BandSets
    >>> file_list_1 = ['file1_b1.tif', 'file1_b2.tif', 'file1_b3.tif']
    >>> catalog.create_bandset(file_list_1, bandset_number=1)
    >>> # start the process
    >>> output = rs.band_clustering(
    ... input_bands=catalog.get_bandset(1), output_raster_path='output.tif', 
    ... algorithm_name='minimum distance', class_number=10, 
    ... seed_signatures='random pixel'
    ... )
"""

from typing import Union, Optional

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.spectral_signatures import SpectralSignaturesCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.tools import band_classification
from remotior_sensus.core.processor_functions import (
    raster_class_unique_values_with_sum, raster_unique_values_with_sum,
    raster_point_values
)
from remotior_sensus.util import (shared_tools, files_directories)


def band_clustering(
        input_bands: Union[list, int, BandSet],
        output_raster_path: str,
        algorithm_name: str,
        class_number: int,
        seed_signatures: Optional[
            Union[str, SpectralSignaturesCatalog]] = None,
        threshold: Optional[float] = None,
        max_iter: Optional[int] = None,
        overwrite: Optional[bool] = False,
        nodata_value: Optional[int] = None, extent_list: Optional[list] = None,
        n_processes: Optional[int] = None, available_ram: int = None,
        bandset_catalog: Optional[BandSetCatalog] = None,
        progress_message: Optional[bool] = True
) -> OutputManager:
    """Calculation of Band Clustering.

    This tool allows for the calculation of Band Clustering.
    A classification raster file is created. 
    In addition, the spectral signatures of the final iteration are saved 
    as .scpx file.

    Args:
        input_bands: input of type BandSet or list of paths or
            integer number of BandSet.
        output_raster_path: path of output file.
        algorithm_name: algorithm name selected from 
            cfg.classification_algorithms 
            between minimum distance and spectral angle maping.
        class_number: the output number of classes (default = 10).
        seed_signatures: the type of seed signatures (random pixel or band 
            mean) or a :func:`~remotior_sensus.core.spectral_signatures.SpectralSignaturesCatalog` 
            containing spectral signatures, or path of spectral signature file.
        threshold: if None, classification without threshold; if float, use 
            this value as threshold for all the signatures.
        max_iter: sets the maximum number of iterations (default = 10).
        overwrite: if True, output overwrites existing files.
        nodata_value: value to be considered as nodata.
        extent_list: list of boundary coordinates left top right bottom.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        bandset_catalog: optional type BandSetCatalog for BandSet number.
        progress_message: if True then start progress message, if False does 
            not start the progress message (useful if launched from other tools).

    Returns:
        Object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - path = [output path]
            - extra = {'signature_path': output spectral signature .scpx file path}

    Examples:
        Perform the clustering on a list of files
            >>> # import Remotior Sensus and start the session
            >>> import remotior_sensus
            >>> rs = remotior_sensus.Session()
            >>> # input file list
            >>> file_list_2 = ['file1_b1.tif', 'file1_b2.tif', 'file1_b3.tif']
            >>> # start the process
            >>> output = rs.band_clustering(
            ... input_bands=file_list_2, output_raster_path='output.tif', 
            ... algorithm_name='minimum distance', class_number=10, 
            ... seed_signatures='band mean'
            ... )
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '),
        message='starting', start=progress_message
    )
    if type(input_bands) is BandSet:
        input_bandset = input_bands
    elif type(input_bands) is int and type(bandset_catalog) is BandSetCatalog:
        input_bandset = bandset_catalog.get(input_bands)
    else:
        # create BandSet Catalog
        new_catalog = BandSetCatalog()
        input_bandset = BandSet.create(input_bands, catalog=new_catalog)
    # prepare process files
    prepared = shared_tools.prepare_process_files(
        input_bands=input_bandset, output_path=output_raster_path,
        overwrite=overwrite, n_processes=n_processes,
        bandset_catalog=bandset_catalog, box_coordinate_list=extent_list
    )
    input_raster_list = prepared['input_raster_list']
    out_path = prepared['output_path']
    n_processes = prepared['n_processes']
    if nodata_value is None:
        nodata_value = prepared['nodata_list']
    # set algorithm
    if (algorithm_name == cfg.spectral_angle_mapping
            or algorithm_name == cfg.spectral_angle_mapping_a):
        algorithm = cfg.spectral_angle_mapping_a
    else:
        algorithm = cfg.minimum_distance_a
    if max_iter is None:
        max_iter = 10
    if max_iter < 0:
        max_iter = 1
    # calculate band mean signatures
    if seed_signatures is None:
        seed_signatures = cfg.band_mean
    if seed_signatures == cfg.band_mean:
        # dummy bands for memory calculation
        dummy_bands = 2
        cfg.multiprocess.run_separated(
            raster_path_list=input_raster_list,
            function=raster_unique_values_with_sum, dummy_bands=dummy_bands,
            use_value_as_nodata=nodata_value, n_processes=n_processes,
            available_ram=available_ram, keep_output_argument=True,
            progress_message='unique values', min_progress=1, max_progress=30
        )
        cfg.multiprocess.find_minimum_maximum()
        if cfg.multiprocess.output is False:
            cfg.logger.log.error('unable to calculate')
            cfg.messages.error('unable to calculate')
            cfg.progress.update(failed=True)
            return OutputManager(check=False)
        [minimum_list, maximum_list] = cfg.multiprocess.output
        signature_catalog = SpectralSignaturesCatalog(
            bandset=input_bandset
        )
        for c in range(0, class_number):
            signature = [
                minimum_list[i] + ((maximum_list[i] - minimum_list[
                    i]) / class_number) * (c + 0.5)
                for i in range(len(input_raster_list))
            ]
            # add signature in SpectralCatalog
            signature_catalog.add_spectral_signature(
                value_list=signature, macroclass_id=c, class_id=c,
                macroclass_name=str(c), class_name=str(c), signature=1
            )
    # get random pixel signatures
    elif seed_signatures == cfg.random_pixel:
        signature_catalog = _seed_signatures(
            input_bandset=input_bandset, class_number=class_number,
            nodata_value=nodata_value, n_processes=n_processes
        )
    # get spectral signatures
    else:
        if type(seed_signatures) is SpectralSignaturesCatalog:
            signature_catalog = seed_signatures
        else:
            signature_catalog = SpectralSignaturesCatalog(
                bandset=input_bandset
            )
            signature_catalog.load(file_path=seed_signatures)
    _k_means_iter(input_bandset, input_raster_list, signature_catalog,
                  algorithm, output_raster_path, class_number, max_iter,
                  threshold, nodata_value, n_processes, available_ram)
    if files_directories.is_file(out_path):
        signature_path = shared_tools.join_path(
            files_directories.parent_directory(out_path), '{}{}'.format(
                files_directories.file_name(out_path, suffix=False),
                cfg.scpx_suffix
            )
        )
        signature_catalog.save(output_path=signature_path)
        cfg.progress.update(end=True)
        cfg.logger.log.info('end; clustering: %s' % str(out_path))
        return OutputManager(
            path=out_path, extra={'signature_path': signature_path}
        )
    else:
        cfg.logger.log.error('classification failed')
        cfg.messages.error('classification failed')
        cfg.progress.update(failed=True)
        return OutputManager(check=False)


def _k_means_iter(
        input_bandset, input_raster_list, _signature_catalog, algorithm_name,
        output_raster, class_number=None, max_iterations=None, threshold=None,
        nodata_value=None, n_processes=None, available_ram=None
):
    """Calculate iteration."""
    # for potential use
    _class_number = class_number
    for iteration in range(1, max_iterations + 1):
        if cfg.action:
            output_path = None
            # remove previously calculated raster
            try:
                if files_directories.is_file(output_path):
                    files_directories.remove_file(output_path)
            except Exception as err:
                str(err)
            output_path = cfg.temp.temporary_file_path(
                name='cluster', name_suffix=cfg.tif_suffix
            )
            # last iteration
            if iteration == max_iterations:
                output_path = output_raster
            band_classification.band_classification(
                input_bands=input_bandset, output_path=output_path,
                spectral_signatures=_signature_catalog, macroclass=True,
                algorithm_name=algorithm_name, signature_raster=False
            )
            # last iteration
            if iteration == max_iterations:
                return output_path
            # list of band order
            band_order = list(range(len(input_raster_list)))
            # add classification raster
            input_raster_list.append(output_path)
            if _signature_catalog is None:
                cfg.logger.log.error('unable to calculate')
                cfg.messages.error('unable to calculate')
                return None
            class_list = _signature_catalog.macroclasses.keys()
            # calculate new spectral signatures
            cfg.multiprocess.run(
                raster_path=input_raster_list,
                function=raster_class_unique_values_with_sum,
                function_argument=band_order, function_variable=class_list,
                use_value_as_nodata=nodata_value, n_processes=n_processes,
                available_ram=available_ram, keep_output_argument=True,
                progress_message='unique values', min_progress=1,
                max_progress=30
            )
            cfg.multiprocess.get_dictionary_sum()
            if cfg.multiprocess.output is False:
                cfg.logger.log.error('unable to calculate')
                cfg.messages.error('unable to calculate')
                return None
            band_dict = cfg.multiprocess.output
            # new spectral signature catalog
            signature_catalog_new = SpectralSignaturesCatalog(
                bandset=input_bandset
            )
            # iterate classes
            for c in class_list:
                signature = []
                for _x in band_order:
                    try:
                        signature.append(
                            band_dict['sum_%i_%i' % (c, _x)]
                            / band_dict['count_%i_%i' % (c, _x)]
                        )
                        # add signature in SpectralCatalog
                        signature_catalog_new.add_spectral_signature(
                            value_list=signature, macroclass_id=c,
                            class_id=c, macroclass_name=str(c),
                            class_name=str(c), signature=1
                        )
                    except Exception as err:
                        str(err)
            # check distance
            distances = []
            dist = None
            ids = signature_catalog_new.table.signature_id.tolist()
            for id_1 in ids:
                for id_2 in ids:
                    if algorithm_name == cfg.minimum_distance_a:
                        # calculate Euclidean Distance
                        dist = (
                            signature_catalog_new.calculate_euclidean_distance(
                                signature_id_x=id_1, signature_id_y=id_2
                            )
                        )
                    elif algorithm_name == cfg.spectral_angle_mapping_a:
                        # calculate spectral angle
                        dist = signature_catalog_new.calculate_spectral_angle(
                            signature_id_x=id_1, signature_id_y=id_2
                        )
                    distances.append(dist)
                values = signature_catalog_new.signatures[id_1].value
                # check nan values
                check = shared_tools.check_nan(values)
                # replace nan signatures
                if check:
                    mc = signature_catalog_new.table[
                        signature_catalog_new.table.signature_id
                        == id_1].macroclass_id[0]
                    old_signature_id = _signature_catalog.table[
                        _signature_catalog.table.macroclass_id
                        == mc].signature_id[0]
                    signature_catalog_new.signatures[id_1].value = (
                        _signature_catalog.signatures[old_signature_id].value
                    )
            if threshold is not None and max(distances) > threshold:
                # copy raster
                files_directories.copy_file(output_path, output_raster)
                return output_raster
            # replace signatures
            _signature_catalog = signature_catalog_new
        else:
            return None


def _seed_signatures(
        input_bandset, class_number, nodata_value, n_processes
):
    """Creates seed signatures."""
    left = min(input_bandset.get_band_attributes('left'))
    top = max(input_bandset.get_band_attributes('top'))
    right = max(input_bandset.get_band_attributes('right'))
    bottom = min(input_bandset.get_band_attributes('bottom'))
    input_raster_list = input_bandset.get_absolute_paths()
    # get 2 x class number points
    points = shared_tools.random_points_grid(
        sample_number=class_number * 2, x_min=left, x_max=right, y_min=bottom,
        y_max=top
    )
    # create signature catalog
    signature_catalog = SpectralSignaturesCatalog(bandset=input_bandset)
    # class value
    c = 1
    for point in points:
        # build function argument list of dictionaries
        argument_list = []
        function_list = []
        for raster, input_raster_i in enumerate(input_raster_list):
            if nodata_value is None:
                nd = None
            else:
                nd = nodata_value[raster]
            argument_list.append(
                {
                    'input_raster': input_raster_i,
                    'point_coordinate': point,
                    'output_no_data': nd,
                    'gdal_path': cfg.gdal_path,
                }
            )
            function_list.append(raster_point_values)
        # get pixel values
        cfg.multiprocess.run_iterative_process(
            function_list=function_list, argument_list=argument_list,
            n_processes=n_processes
        )
        if cfg.multiprocess.output is False:
            cfg.logger.log.error('unable to calculate')
            cfg.messages.error('unable to calculate')
            return None
        signature = []
        for r in cfg.multiprocess.output:
            signature.extend(r)
        if None not in signature:
            # add signature in SpectralCatalog
            signature_catalog.add_spectral_signature(
                value_list=signature, macroclass_id=c, class_id=c,
                macroclass_name=str(c), class_name=str(c), signature=1
            )
            if c == class_number:
                break
            else:
                c += 1
    return signature_catalog
