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
"""Band mosaic.

This tool performs the mosaic of single bands, or multiple BandSets.
Corresponding bands in two or more BandSet are merged into a single raster
for each band, covering the extent of input bands.
For instance, this is useful to mosaic several multiband images togheter,
obtaining a multispectral mosaic of bands (e.g., the first band of the mosaic
corresponds to all the first bands of input images).

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> file_list = ['file1_b1.tif', 'file2_b1.tif']
    >>> mosaic_bands = rs.mosaic(input_bands=file_list,output_path='output_directory')
"""  # noqa: E501

from typing import Union, Optional

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.util import files_directories, raster_vector, shared_tools


def mosaic(
        input_bands: Union[list, int, BandSet],
        output_path: Optional[str] = None, overwrite: Optional[bool] = False,
        prefix: Optional[str] = '', nodata_value: Optional[int] = None,
        n_processes: Optional[int] = None,
        available_ram: Optional[int] = None,
        virtual_output: Optional[bool] = False,
        output_name: Optional[str] = None,
        reference_raster_crs: Optional[str] = None,
        bandset_catalog: Optional[BandSetCatalog] = None
) -> OutputManager:
    """Mosaic bands.

    This tool performs the mosaic of corresponding bands from multiple
    BandSets.
    A new raster is created for each band, named as output_name or first band name,
    and followed by band number.

    Args:
        input_bands: list of paths of input rasters, or number of BandSet, or BandSet object.
        output_path: string of output path directory.
        overwrite: if True, output overwrites existing files.
        prefix: optional string for output name prefix.
        nodata_value: value to be considered as nodata.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        virtual_output: if True (and output_path is directory), save output
            as virtual raster of multiprocess parts.
        output_name: string used as general name for all output bands.
        reference_raster_crs: path to a raster to be used as crs reference; if None, the first band is used as reference.
        bandset_catalog: optional type BandSetCatalog for BandSet number.

    Returns:
        Object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - paths = output list

    Examples:
        Perform the mosaic of three BandSets
            >>> # import Remotior Sensus and start the session
            >>> import remotior_sensus
            >>> rs = remotior_sensus.Session()
            >>> catalog = rs.bandset_catalog()
            >>> # create three BandSets
            >>> file_list_1 = ['file1_b1.tif', 'file1_b2.tif', 'file1_b3.tif']
            >>> file_list_2 = ['file2_b1.tif', 'file2_b2.tif', 'file2_b3.tif']
            >>> file_list_3 = ['file3_b1.tif', 'file3_b2.tif', 'file3_b3.tif']
            >>> catalog.create_bandset(file_list_1, bandset_number=1)
            >>> catalog.create_bandset(file_list_2, bandset_number=2)
            >>> catalog.create_bandset(file_list_3, bandset_number=3)
            >>> # start the process
            >>> mosaic_bands = rs.mosaic(input_bands=[1, 2, 3],
            ... output_path='output_directory', bandset_catalog=catalog)
    """  # noqa: E501
    cfg.logger.log.info('start')
    cfg.progress.update(
        process=__name__.split('.')[-1].replace('_', ' '), message='starting',
        start=True
    )
    cfg.logger.log.debug('input_bands: %s' % str(input_bands))
    if n_processes is None:
        n_processes = cfg.n_processes
    # list of band lists to mosaic
    band_list_list = []
    if len(input_bands) == 1:
        # single BandSet of bands to mosaic
        if type(input_bands[0]) is BandSet or type(
                input_bands[0]
        ) is int:
            # get input list
            band_list = BandSetCatalog.get_band_list(
                input_bands[0], bandset_catalog
            )
            band_list_list.append(band_list)
        else:
            cfg.logger.log.error('band list')
            return OutputManager(check=False)
    else:
        combination_band_list = []
        raster_list = []
        for i in input_bands:
            # list of band sets
            if type(i) is BandSet or type(i) is int:
                # get input list
                band_list = BandSetCatalog.get_band_list(i, bandset_catalog)
                combination_band_list.append(band_list)
            # list of raster paths
            else:
                raster_list.append(i)
        if len(raster_list) > 0:
            print('raster_list', raster_list)
            if type(raster_list[0]) is list:
                band_list_list = raster_list
            else:
                band_list_list.append(raster_list)
        elif len(combination_band_list) > 0:
            # combine corresponding bands
            try:
                size = len(combination_band_list[0])
                for s in range(size):
                    new_list = []
                    for c in combination_band_list:
                        new_list.append(c[s])
                    band_list_list.append(new_list)
            except Exception as err:
                cfg.logger.log.error(str(err))
                cfg.messages.error(str(err))
                return OutputManager(check=False)
    # mosaic every list of bands
    n = 0
    output_list = []
    for mosaic_list in band_list_list:
        cfg.logger.log.debug('mosaic_list: %s' % str(mosaic_list))
        cfg.progress.update(
            step=n, steps=len(band_list_list), minimum=10, maximum=99,
            message='processing list %s' % str(n + 1),
            percentage=n / len(band_list_list)
        )
        # list of inputs
        prepared = shared_tools.prepare_input_list(
            mosaic_list, reference_raster_crs, n_processes=n_processes
        )
        input_raster_list = prepared['input_list']
        raster_info = prepared['information_list']
        nodata_list = prepared['nodata_list']
        warped = prepared['warped']
        try:
            output_data_type = raster_info[0][8]
        except Exception as err:
            str(err)
            output_data_type = None
        if nodata_value is None:
            nodata_value = nodata_list[0]
        if warped:
            virtual_output = False
        # check output path
        if output_name is None:
            p = shared_tools.join_path(
                output_path, '{}{}'.format(
                    prefix, files_directories.file_name(input_raster_list[0])
                )
            ).replace('\\', '/')
        else:
            p = shared_tools.join_path(
                output_path, '{}{}{}'.format(prefix, output_name, n)
            ).replace('\\', '/')
        out_path, vrt_r = files_directories.raster_output_path(
            p, overwrite=overwrite
        )
        output_list.append(out_path)
        if virtual_output:
            try:
                # create virtual raster
                raster_vector.create_virtual_raster_2_mosaic(
                    input_raster_list=input_raster_list, output=out_path,
                    dst_nodata=nodata_value, data_type=output_data_type
                )
            except Exception as err:
                cfg.logger.log.error(str(err))
                cfg.messages.error(str(err))
                return OutputManager(check=False)
        else:
            vrt_file = cfg.temp.temporary_raster_path(extension=cfg.vrt_suffix)
            try:
                # create virtual raster
                raster_vector.create_virtual_raster_2_mosaic(
                    input_raster_list=input_raster_list, output=vrt_file,
                    dst_nodata=nodata_value, data_type=output_data_type
                )
                # copy raster
                cfg.multiprocess.gdal_copy_raster(
                    vrt_file, out_path, 'GTiff', cfg.raster_compression, 'LZW',
                    additional_params='-ot %s' % str(output_data_type),
                    n_processes=n_processes, available_ram=available_ram
                )
            except Exception as err:
                cfg.logger.log.error(str(err))
                cfg.messages.error(str(err))
                return OutputManager(check=False)
        n = n + 1
    cfg.progress.update(end=True)
    cfg.logger.log.info('end; mosaic: %s' % str(output_list))
    return OutputManager(paths=output_list)
