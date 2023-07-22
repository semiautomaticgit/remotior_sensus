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
Perform the preprocessing of products.
"""

import os
from xml.dom import minidom

import numpy as np

from remotior_sensus.core import configurations as cfg, table_manager as tm
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.core.processor_functions import (
    band_calculation, raster_unique_values_with_sum
)
from remotior_sensus.util import files_directories


# create product table and preprocess
def preprocess(
        input_path, output_path, metadata_file_path=None,
        product=None, nodata_value=None, sensor=None, acquisition_date=None,
        dos1_correction=False, output_prefix='', n_processes: int = None,
        available_ram: int = None, progress_message=True
):
    table = create_product_table(
        input_path=input_path, metadata_file_path=metadata_file_path,
        product=product, nodata_value=nodata_value, sensor=sensor,
        acquisition_date=acquisition_date
    )
    output = perform_preprocess(
        product_table=table, output_path=output_path,
        dos1_correction=dos1_correction, output_prefix=output_prefix,
        n_processes=n_processes, available_ram=available_ram,
        progress_message=progress_message
        )
    return output


# preprocess products
def perform_preprocess(
        product_table, output_path, dos1_correction=False,
        output_prefix='', n_processes: int = None, available_ram: int = None,
        progress_message=True
) -> OutputManager:
    """Preprocess products.

    Perform image conversion to reflectance of several products.

    Can calculate DOS1 corrected reflectance (Sobrino, J. et al., 2004. Land
    surface temperature retrieval from LANDSAT TM 5. Remote Sensing of
    Environment, Elsevier, 90, 434-440)  approximating path radiance
    to path reflectance for level 1 data:
    TOA reflectance = DN * reflectance_scale + reflectance_offset
    path reflectance p = DNm - Dark Object reflectance = DNm * reflectance_scale + reflectance_offset - 0.01
    land surface reflectance = TOA reflectance - p = (DN * reflectance_scale) - (DNm * reflectance_scale - 0.01)

    Landsat's data Collection 1 and 2
    Level 1T
    Landsat 8-9 TOA reflectance proportional to exo-atmospheric solar
    irradiance in each band and the Earth-Sun distance
    (USGS, 2021. Landsat 8-9 Calibration and Validation (Cal/Val) Algorithm
    Description Document (ADD). Version 4.0. Department of the Interior
    U.S. Geological Survey, South Dakota)
    TOA reflectance with correction for the sun angle =
    DN * Reflectance multiplicative scaling factor + Reflectance additive
    scaling factor / sin(Sun elevation)
    Level 2S
    Surface reflectance = DN * Reflectance multiplicative scaling factor +
    Reflectance additive scaling factor

    Sentinel-2 data
    Level 1C
    TOA reflectance = DN / QUANTIFICATION VALUE + OFFSET
    Level 2S
    Surface reflectance = DN / QUANTIFICATION VALUE + OFFSET

    Args:
        product_table:
        output_path:
        dos1_correction:
        output_prefix:
        n_processes:
        available_ram: number of megabytes of RAM available to processes.
        progress_message:

    Returns:
        object :func:`~remotior_sensus.core.output_manager.OutputManager`
    """  # noqa: E501
    if progress_message:
        cfg.logger.log.info('start')
        cfg.progress.update(
            process=__name__.split('.')[-1].replace('_', ' '),
            message='starting', start=True
        )
    cfg.logger.log.debug('product_table: %s' % str(product_table))
    if n_processes is None:
        n_processes = cfg.n_processes
    input_list = []
    input_dos1_list = []
    dos1_nodata_list = []
    nodata_list = []
    calculation_datatype = []
    scale_list = []
    offset_list = []
    output_nodata = []
    output_datatype = []
    output_raster_path_list = []
    expressions = []
    dos1_expressions = []
    # create process string list
    # Sentinel-2
    sentinel_product = product_table[product_table.product == cfg.sentinel2]
    # Landsat
    landsat_product = product_table[product_table.product == cfg.landsat]
    # Sentinel-2
    if len(sentinel_product) > 0:
        if dos1_correction:
            # exclude Level-2A
            sentinel_product_2a = sentinel_product[
                sentinel_product.processing_level != 'level-2a']
            # calculate DOS1 corrected reflectance approximating path
            # radiance to path reflectance
            # land surface reflectance = TOA reflectance - p =
            # (DN * reflectance_scale) - (DNm * reflectance_scale - 0.01)
            # raster and dnm are variables in the calculation
            string_1 = np.char.add(
                'np.clip(( %s * ' % cfg.array_function_placeholder,
                sentinel_product_2a.scale.astype('<U16')
            )
            string_2 = np.char.add(string_1, ' - (')
            string_3 = np.char.add(
                string_2, sentinel_product_2a.scale.astype('<U16')
            )
            dos1_expressions.extend(
                np.char.add(string_3, ' * dnm - 0.01)), 0, 1)').tolist()
            )
            input_dos1_list.extend(sentinel_product_2a.product_path.tolist())
            # output raster list
            output_string_1 = np.char.add(
                '%s/%s' % (output_path, output_prefix),
                sentinel_product_2a.band_name
            )
            output_raster_path_list.extend(
                np.char.add(output_string_1, cfg.tif_suffix).tolist()
            )
            nodata_list.extend(sentinel_product_2a.nodata.tolist())
            dos1_nodata_list.extend(sentinel_product_2a.nodata.tolist())
            calculation_datatype.extend(
                [np.float32] * len(sentinel_product_2a)
            )
            output_datatype.extend([cfg.uint16_dt] * len(sentinel_product_2a))
            scale_list.extend([0.0001] * len(sentinel_product_2a))
            offset_list.extend([0] * len(sentinel_product_2a))
            output_nodata.extend(
                [cfg.nodata_val_UInt16] * len(sentinel_product_2a)
            )
        else:
            # calculate reflectance = DN / quantificationValue = DN * scale
            # raster is interpreted as variable in the calculation
            string_1 = np.char.add(
                'np.clip( ( %s * ' % cfg.array_function_placeholder,
                sentinel_product.scale.astype('<U16')
            )
            expressions.extend(np.char.add(string_1, ') , 0, 1)').tolist())
            input_list.extend(sentinel_product.product_path.tolist())
            # output raster list
            output_string_1 = np.char.add(
                '%s/%s' % (output_path, output_prefix),
                sentinel_product.band_name
            )
            output_raster_path_list.extend(
                np.char.add(output_string_1, cfg.tif_suffix).tolist()
            )
            nodata_list.extend(sentinel_product.nodata.tolist())
            calculation_datatype.extend([np.float32] * len(sentinel_product))
            output_datatype.extend([cfg.uint16_dt] * len(sentinel_product))
            scale_list.extend([0.0001] * len(sentinel_product))
            offset_list.extend([0] * len(sentinel_product))
            output_nodata.extend(
                [cfg.nodata_val_UInt16] * len(sentinel_product)
            )
    # Landsat
    elif len(landsat_product) > 0:
        # temperature
        landsat_temperature_product = landsat_product[landsat_product.k1 != 0]
        string_0 = np.char.add(
            landsat_temperature_product.k2.astype('<U16'), ' / ( log( 1 + '
        )
        string_1 = np.char.add(
            string_0, landsat_temperature_product.k1.astype('<U16')
        )
        string_2 = np.char.add(
            string_1, ' / (%s * ' % cfg.array_function_placeholder
        )
        string_3 = np.char.add(
            string_2, landsat_temperature_product.scale.astype('<U16')
        )
        string_4 = np.char.add(string_3, ' + ')
        string_5 = np.char.add(
            string_4, landsat_temperature_product.offset.astype('<U16')
        )
        expressions.extend(np.char.add(string_5, ') ) )').tolist())
        input_list.extend(landsat_temperature_product.product_path.tolist())
        calculation_datatype.extend(
            [np.float32] * len(landsat_temperature_product)
        )
        output_datatype.extend(
            [cfg.float32_dt] * len(landsat_temperature_product)
        )
        scale_list.extend([1] * len(landsat_temperature_product))
        offset_list.extend([0] * len(landsat_temperature_product))
        output_nodata.extend(
            [cfg.nodata_val_Float32] * len(landsat_temperature_product)
        )
        # output raster list
        output_string_temperature_1 = np.char.add(
            '%s/%s' % (output_path, output_prefix),
            landsat_temperature_product.band_name
        )
        output_raster_path_list.extend(
            np.char.add(output_string_temperature_1, cfg.tif_suffix).tolist()
        )
        nodata_list.extend(landsat_temperature_product.nodata.tolist())
        landsat_temperature_product_10 = landsat_product[
            landsat_product.band_number == '10']
        string_1 = np.char.add(
            '%s * ' % cfg.array_function_placeholder,
            landsat_temperature_product_10.scale.astype('<U16')
        )
        string_2 = np.char.add(string_1, ' + ')
        string_3 = np.char.add(
            string_2, landsat_temperature_product_10.offset.astype('<U16')
        )
        expressions.extend(string_3.tolist())
        input_list.extend(landsat_temperature_product_10.product_path.tolist())
        calculation_datatype.extend(
            [np.float32] * len(landsat_temperature_product_10)
        )
        output_datatype.extend(
            [cfg.float32_dt] * len(landsat_temperature_product_10)
        )
        scale_list.extend([1] * len(landsat_temperature_product_10))
        offset_list.extend([0] * len(landsat_temperature_product_10))
        output_nodata.extend(
            [cfg.nodata_val_Float32] * len(landsat_temperature_product_10)
        )
        # output raster list
        output_string_temperature_10 = np.char.add(
            '%s/%s' % (output_path, output_prefix),
            landsat_temperature_product_10.band_name
        )
        output_raster_path_list.extend(
            np.char.add(output_string_temperature_10, cfg.tif_suffix).tolist()
        )
        nodata_list.extend(landsat_temperature_product_10.nodata.tolist())
        if dos1_correction:
            # exclude level 2 products and temperature
            landsat_product_l1 = landsat_product[
                (landsat_product.processing_level == 'l1tp') & (
                        landsat_product.band_number != '10') &
                (landsat_product.k1 == 0)]
            # calculate DOS1 corrected reflectance approximating path
            # radiance to path reflectance
            # land surface reflectance = TOA reflectance - p =
            # (DN * reflectance_scale) - (DNm * reflectance_scale - 0.01)
            # raster and dnm are variables in the calculation
            string_1 = np.char.add(
                'np.clip(( %s * ' % cfg.array_function_placeholder,
                landsat_product_l1.scale.astype('<U16')
            )
            string_2 = np.char.add(string_1, ' - (')
            string_3 = np.char.add(
                string_2, landsat_product_l1.scale.astype('<U16')
            )
            dos1_expressions.extend(
                np.char.add(string_3, ' * dnm - 0.01)), 0, 1)').tolist()
            )
            input_dos1_list.extend(landsat_product_l1.product_path.tolist())
            # output raster list
            output_string_1 = np.char.add(
                '%s/%s' % (output_path, output_prefix),
                landsat_product_l1.band_name
            )
            output_raster_path_list.extend(
                np.char.add(output_string_1, cfg.tif_suffix).tolist()
            )
            nodata_list.extend(landsat_product_l1.nodata.tolist())
            dos1_nodata_list.extend(landsat_product_l1.nodata.tolist())
            calculation_datatype.extend([np.float32] * len(landsat_product_l1))
            output_datatype.extend([cfg.uint16_dt] * len(landsat_product_l1))
            scale_list.extend([0.0001] * len(landsat_product_l1))
            offset_list.extend([0] * len(landsat_product_l1))
            output_nodata.extend(
                [cfg.nodata_val_UInt16] * len(landsat_product_l1)
            )
        else:
            # level 1 products
            landsat_1_product = landsat_product[
                (landsat_product.processing_level == 'l1tp') & (
                        landsat_product.band_number != '10')
                & (landsat_product.k1 == 0)]
            # calculate reflectance = (raster * scale
            #  + offset) / sin(Sun elevation)
            # raster is interpreted as variable in the calculation
            string_1 = np.char.add(
                'np.clip( ( ( %s * ' % cfg.array_function_placeholder,
                landsat_1_product.scale.astype('<U16')
            )
            string_2 = np.char.add(string_1, ' + ')
            string_3 = np.char.add(
                string_2, landsat_1_product.offset.astype('<U16')
            )
            string_4 = np.char.add(string_3, ') / sin(')
            string_5 = np.char.add(
                string_4, landsat_1_product.sun_elevation.astype(
                    '<U16'
                )
            )
            expressions.extend(np.char.add(string_5, ') ) , 0, 1)').tolist())
            input_list.extend(landsat_1_product.product_path.tolist())
            # output raster list
            output_string_1 = np.char.add(
                '%s/%s' % (output_path, output_prefix),
                landsat_1_product.band_name
            )
            output_raster_path_list.extend(
                np.char.add(output_string_1, cfg.tif_suffix).tolist()
            )
            nodata_list.extend(landsat_1_product.nodata.tolist())
            calculation_datatype.extend([np.float32] * len(landsat_1_product))
            output_datatype.extend([cfg.uint16_dt] * len(landsat_1_product))
            scale_list.extend([0.0001] * len(landsat_1_product))
            offset_list.extend([0] * len(landsat_1_product))
            output_nodata.extend(
                [cfg.nodata_val_UInt16] * len(landsat_1_product)
            )
            # level 2 products
            landsat_2_product = landsat_product[
                (landsat_product.processing_level == 'l2sp') & (
                        landsat_product.band_number != '10')]
            # calculate reflectance = (raster * scale
            #  + offset) / sin(Sun elevation)
            # raster is interpreted as variable in the calculation
            string_1 = np.char.add(
                'np.clip( ( %s * ' % cfg.array_function_placeholder,
                landsat_2_product.scale.astype('<U16')
            )
            string_2 = np.char.add(string_1, ' + ')
            string_3 = np.char.add(
                string_2, landsat_2_product.offset.astype('<U16')
            )
            expressions.extend(np.char.add(string_3, ') , 0, 1)').tolist())
            input_list.extend(landsat_2_product.product_path.tolist())
            # output raster list
            output_string_2 = np.char.add(
                '%s/%s' % (output_path, output_prefix),
                landsat_2_product.band_name
            )
            output_raster_path_list.extend(
                np.char.add(output_string_2, cfg.tif_suffix).tolist()
            )
            nodata_list.extend(landsat_2_product.nodata.tolist())
            calculation_datatype.extend([np.float32] * len(landsat_product))
            output_datatype.extend([cfg.uint16_dt] * len(landsat_product))
            scale_list.extend([0.0001] * len(landsat_product))
            offset_list.extend([0] * len(landsat_product))
            output_nodata.extend(
                [cfg.nodata_val_UInt16] * len(landsat_product)
            )
    files_directories.create_directory(output_path)
    # dummy bands for memory calculation
    dummy_bands = 2
    # conversion
    if dos1_correction:
        # get min dn values
        cfg.multiprocess.run_separated(
            raster_path_list=input_dos1_list,
            function=raster_unique_values_with_sum, dummy_bands=dummy_bands,
            use_value_as_nodata=dos1_nodata_list, n_processes=n_processes,
            available_ram=available_ram, keep_output_argument=True,
            progress_message='unique values', min_progress=1, max_progress=30
        )
        cfg.multiprocess.find_minimum_dn()
        min_dn = cfg.multiprocess.output
        for i in range(len(dos1_expressions)):
            expressions.append(
                dos1_expressions[i].replace('dnm', str(min_dn[i]))
            )
            input_list.append(input_dos1_list[i])
    # dummy bands for memory calculation
    dummy_bands = 2
    # run calculation
    cfg.multiprocess.run_separated(
        raster_path_list=input_list, function=band_calculation,
        function_argument=expressions,
        calculation_datatype=calculation_datatype,
        use_value_as_nodata=nodata_list, dummy_bands=dummy_bands,
        output_raster_list=output_raster_path_list,
        output_data_type=output_datatype, output_nodata_value=output_nodata,
        compress=cfg.raster_compression, n_processes=n_processes,
        available_ram=available_ram, scale=scale_list, offset=offset_list,
        progress_message='processing', min_progress=30, max_progress=99
    )
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
    cfg.logger.log.info(
        'end; preprocess products: %s' % str(output_raster_path_list)
    )
    return OutputManager(paths=output_raster_path_list)


# create product table
def create_product_table(
        input_path, metadata_file_path=None, product=None, nodata_value=None,
        sensor=None, acquisition_date=None
):
    band_names = []
    band_number_list = []
    product_path_list = []
    product_name_list = []
    scale_value_list = []
    scale_offset_dict = {}
    k_dict = {}
    offset_value_list = []
    k2_list = []
    k1_list = []
    e_sun_list = []
    sun_elevation_list = []
    earth_sun_distance_list = []
    spacecraft_list = []
    metadata = metadata_doc = metadata_type = product_date = product_name = \
        processing_level = sun_elevation = None
    earth_sun_distance = None
    if product == cfg.sentinel2:
        product_name = cfg.sentinel2
    elif product == cfg.landsat:
        product_name = cfg.landsat
    # get metadata
    if metadata_file_path is None:
        for f in os.listdir(input_path):
            # Sentinel-2 metadata
            if f.lower().endswith('.xml') and (
                    'mtd_msil1c' in f.lower() or 'mtd_safl1c' in f.lower()
                    or 'mtd_msil2a' in f.lower()):
                metadata = '%s/%s' % (input_path, str(f))
                product_name = cfg.sentinel2
                metadata_type = 'xml'
            # Landsat metadata
            elif f[0].lower() == 'l' and f.lower().endswith(
                    '.xml'
            ) and 'mtl' in f.lower():
                metadata = '%s/%s' % (input_path, str(f))
                product_name = cfg.landsat
                metadata_type = 'xml'
    else:
        metadata = metadata_file_path
        if files_directories.file_extension(metadata) == cfg.xml_suffix:
            metadata_type = 'xml'
    if metadata_type == 'xml':
        # open metadata
        try:
            metadata_doc = minidom.parse(metadata)
            # Sentinel-2
            try:
                spacecraft_name = \
                    metadata_doc.getElementsByTagName('SPACECRAFT_NAME')[
                        0].firstChild.data
                if spacecraft_name:
                    product_name = cfg.sentinel2
            except Exception as err:
                str(err)
            # Landsat
            try:
                spacecraft_id = \
                    metadata_doc.getElementsByTagName('SPACECRAFT_ID')[
                        0].firstChild.data
                if spacecraft_id:
                    product_name = cfg.landsat
            except Exception as err:
                str(err)
        except Exception as err:
            cfg.messages.error('unable to open metadata')
            cfg.logger.log.error(str(err))
            return OutputManager(check=False)
    # Sentinel-2
    if product_name == cfg.sentinel2:
        cfg.logger.log.debug(cfg.sentinel2)
        scale_value = 1 / 10000
        offset_value = 0
        sentinel2_bands = cfg.satellites[cfg.satSentinel2][2]
        # open metadata
        if metadata_doc:
            try:
                # get date in the format YYYY-MM-DD
                product_date = \
                    metadata_doc.getElementsByTagName('PRODUCT_START_TIME')[
                        0].firstChild.data.split('T')[0]
                processing_level = \
                    metadata_doc.getElementsByTagName('PROCESSING_LEVEL')[
                        0].firstChild.data
                # L2A products
                if '2a' in processing_level.lower():
                    scale_value = 1 / int(
                        metadata_doc.getElementsByTagName(
                            'BOA_QUANTIFICATION_VALUE'
                        )[0].firstChild.data
                    )
                    offset = metadata_doc.getElementsByTagName(
                        'BOA_ADD_OFFSET'
                    )
                # L1C products
                else:
                    scale_value = 1 / int(
                        metadata_doc.getElementsByTagName(
                            'QUANTIFICATION_VALUE'
                        )[0].firstChild.data
                    )
                    offset = metadata_doc.getElementsByTagName(
                        'RADIO_ADD_OFFSET'
                    )
                for n in range(len(sentinel2_bands)):
                    if offset_value:
                        offset_value_list.append(offset[n].firstChild.data)
                cfg.logger.log.debug('metadata')
            except Exception as err:
                cfg.logger.log.error(str(err))
                cfg.messages.error(str(err))
        # use default values
        else:
            scale_value = 1 / 10000
            offset_value_list = [0] * len(sentinel2_bands)
            cfg.messages.warning('using default values without metadata')
            cfg.logger.log.debug('no metadata')
        # get bands
        file_list = files_directories.files_in_directory(
            input_path, sort_files=True, suffix_filter=cfg.tif_suffix
        )
        file_list.extend(
            files_directories.files_in_directory(
                input_path, sort_files=True, suffix_filter='.jp2'
            )
        )
        for f in file_list:
            # check band number
            if f[-6:-4].lower() in sentinel2_bands:
                band_names.append(files_directories.file_name(f))
                product_path_list.append(f)
                band_number_list.append(f[-6:-4].lower())
        product_name_list = [product_name] * len(band_names)
        spacecraft_list = [product_name] * len(band_names)
        scale_value_list = [scale_value] * len(band_names)
        if len(offset_value_list) == 0:
            offset_value_list = [0] * len(band_names)
    elif product_name == cfg.landsat:
        cfg.logger.log.debug(cfg.landsat)
        # open metadata
        if metadata_doc:
            sensor_id = metadata_doc.getElementsByTagName('SENSOR_ID')[
                0].firstChild.data
            processing_level = \
                metadata_doc.getElementsByTagName('PROCESSING_LEVEL')[
                    0].firstChild.data.lower()
            spacecraft_id = metadata_doc.getElementsByTagName('SPACECRAFT_ID')[
                0].firstChild.data.lower()
            # get date in the format YYYY-MM-DD
            product_date = metadata_doc.getElementsByTagName('DATE_ACQUIRED')[
                0].firstChild.data
            sun_elevation = metadata_doc.getElementsByTagName('SUN_ELEVATION')[
                0].firstChild.data
            earth_sun_distance = \
                metadata_doc.getElementsByTagName('EARTH_SUN_DISTANCE')[
                    0].firstChild.data
            if sensor_id.lower() == cfg.sensor_oli:
                band_list = cfg.satellites[cfg.satLandsat8][2]
            elif sensor_id.lower() == cfg.sensor_etm:
                band_list = cfg.satellites[cfg.satLandsat7][2]
            elif sensor_id.lower() == cfg.sensor_tm:
                band_list = cfg.satellites[cfg.satLandsat45][2]
            elif sensor_id.lower() == cfg.sensor_mss:
                band_list = cfg.satellites[cfg.satLandsat13][2]
            else:
                band_list = []
            for b in band_list:
                if processing_level == 'l2sp':
                    reflectance_tag = metadata_doc.getElementsByTagName(
                        'LEVEL2_SURFACE_REFLECTANCE_PARAMETERS'
                    )[0]
                else:
                    reflectance_tag = metadata_doc.getElementsByTagName(
                        'LEVEL1_RADIOMETRIC_RESCALING'
                    )[0]
                try:
                    reflectance_mult = reflectance_tag.getElementsByTagName(
                        'REFLECTANCE_MULT_BAND_%s' % str(b)
                    )[0].firstChild.data
                    reflectance_add = reflectance_tag.getElementsByTagName(
                        'REFLECTANCE_ADD_BAND_%s' % str(b)
                    )[0].firstChild.data
                    scale_offset_dict[str(b)] = [float(reflectance_mult),
                                                 float(reflectance_add)]
                except Exception as err:
                    str(err)
                    scale_offset_dict[str(b)] = [1, 0]
            # temperature
            # Landsat 8-9
            if sensor_id.lower() == cfg.sensor_oli and processing_level == \
                    'l2sp':
                scale_offset_dict['10'] = [float(
                    metadata_doc.getElementsByTagName(
                        'TEMPERATURE_MULT_BAND_ST_B10'
                    )[0].firstChild.data
                ), float(
                    metadata_doc.getElementsByTagName(
                        'TEMPERATURE_ADD_BAND_ST_B10'
                    )[0].firstChild.data
                )]
            # Landsat 7 level 1
            elif sensor_id.lower() == cfg.sensor_etm and processing_level == \
                    'l1tp':
                k_dict['6_VCID_1'] = [float(
                    metadata_doc.getElementsByTagName(
                        'K1_CONSTANT_BAND_6_VCID_1'
                    )[0].firstChild.data
                ), float(
                    metadata_doc.getElementsByTagName(
                        'K2_CONSTANT_BAND_6_VCID_1'
                    )[0].firstChild.data
                ), float(
                    metadata_doc.getElementsByTagName(
                        'RADIANCE_MULT_BAND_6_VCID_1'
                    )[0].firstChild.data
                ), float(
                    metadata_doc.getElementsByTagName(
                        'RADIANCE_ADD_BAND_6_VCID_1'
                    )[0].firstChild.data
                )]
                k_dict['6_VCID_2'] = [float(
                    metadata_doc.getElementsByTagName(
                        'K1_CONSTANT_BAND_6_VCID_2'
                    )[0].firstChild.data
                ), float(
                    metadata_doc.getElementsByTagName(
                        'K2_CONSTANT_BAND_6_VCID_2'
                    )[0].firstChild.data
                ), float(
                    metadata_doc.getElementsByTagName(
                        'RADIANCE_MULT_BAND_6_VCID_2'
                    )[0].firstChild.data
                ), float(
                    metadata_doc.getElementsByTagName(
                        'RADIANCE_ADD_BAND_6_VCID_2'
                    )[0].firstChild.data
                )]
            # Landsat 7 level 2
            elif sensor_id.lower() == cfg.sensor_tm and processing_level == \
                    'l2sp':
                scale_offset_dict['6'] = [float(
                    metadata_doc.getElementsByTagName(
                        'TEMPERATURE_MULT_BAND_ST_B6'
                    )[0].firstChild.data
                ), float(
                    metadata_doc.getElementsByTagName(
                        'TEMPERATURE_ADD_BAND_ST_B6'
                    )[0].firstChild.data
                )]
            # Landsat 5 level 1
            elif sensor_id.lower() == cfg.sensor_tm and processing_level == \
                    'l1tp':
                k_dict['6'] = [float(
                    metadata_doc.getElementsByTagName('K1_CONSTANT_BAND_6')[
                        0].firstChild.data
                ), float(
                    metadata_doc.getElementsByTagName('K2_CONSTANT_BAND_6')[
                        0].firstChild.data
                ), float(
                    metadata_doc.getElementsByTagName('RADIANCE_MULT_BAND_6')[
                        0].firstChild.data
                ), float(
                    metadata_doc.getElementsByTagName('RADIANCE_ADD_BAND_6')[
                        0].firstChild.data
                )]
            # Landsat 5 level 2
            elif sensor_id.lower() == cfg.sensor_tm and processing_level == \
                    'l2sp':
                scale_offset_dict['6'] = [float(
                    metadata_doc.getElementsByTagName(
                        'TEMPERATURE_MULT_BAND_ST_B6'
                    )[0].firstChild.data
                ), float(
                    metadata_doc.getElementsByTagName(
                        'TEMPERATURE_ADD_BAND_ST_B6'
                    )[0].firstChild.data
                )]

        # use default values
        else:
            spacecraft_id = cfg.landsat
            if sensor:
                sensor_id = sensor
            else:
                sensor_id = 'oli_tirs'
                processing_level = 'l2sp'
            if acquisition_date:
                product_date = acquisition_date
            else:
                product_date = '2000-01-01'
            cfg.messages.warning('using default values without metadata')
            cfg.logger.log.debug('no metadata')
        cfg.logger.log.debug('sensor_id: %s' % sensor_id)
        # get bands
        if sensor_id.lower() == cfg.sensor_oli:
            landsat_bands = cfg.satellites[cfg.satLandsat8][2]
        elif sensor_id.lower() == cfg.sensor_etm:
            landsat_bands = cfg.satellites[cfg.satLandsat7][2]
        elif sensor_id.lower() == cfg.sensor_tm:
            landsat_bands = cfg.satellites[cfg.satLandsat45][2]
        elif sensor_id.lower() == cfg.sensor_mss:
            landsat_bands = cfg.satellites[cfg.satLandsat13][2]
        else:
            landsat_bands = []
        file_list = files_directories.files_in_directory(
            input_path, sort_files=True, suffix_filter=cfg.tif_suffix
        )
        for f in file_list:
            # check band number for multispectral bands
            if f[-5:-4] in landsat_bands:
                band_names.append(files_directories.file_name(f))
                band_number_list.append(f[-5:-4])
                product_path_list.append(f)
                sun_elevation_list.append(sun_elevation)
                earth_sun_distance_list.append(earth_sun_distance)
                if f[-5:-4] in scale_offset_dict:
                    scale_value_list.append(scale_offset_dict[f[-5:-4]][0])
                    offset_value_list.append(scale_offset_dict[f[-5:-4]][1])
                k1_list.append(0)
                k2_list.append(0)
            # temperature bands Landsat 5 band 6
            elif sensor_id.lower() == cfg.sensor_tm:
                if processing_level == 'l2sp' and f[-5:-4] == '6':
                    band_names.append(files_directories.file_name(f))
                    band_number_list.append('6')
                    product_path_list.append(f)
                    sun_elevation_list.append(sun_elevation)
                    earth_sun_distance_list.append(earth_sun_distance)
                    if '6' in scale_offset_dict:
                        scale_value_list.append(scale_offset_dict['6'][0])
                        offset_value_list.append(scale_offset_dict['6'][1])
                    k1_list.append(0)
                    k2_list.append(0)
                elif processing_level == 'l1tp' and f[-5:-4] == '6':
                    band_names.append(files_directories.file_name(f))
                    band_number_list.append('6')
                    product_path_list.append(f)
                    sun_elevation_list.append(sun_elevation)
                    earth_sun_distance_list.append(earth_sun_distance)
                    if '6' in k_dict:
                        k1_list.append(k_dict['6'][0])
                        k2_list.append(k_dict['6'][1])
                        scale_value_list.append(k_dict['6'][2])
                        offset_value_list.append(k_dict['6'][3])
            # temperature bands Landsat 7 band 6
            elif sensor_id.lower() == cfg.sensor_etm:
                if processing_level == 'l2sp' and f[-5:-4] == '6':
                    band_names.append(files_directories.file_name(f))
                    band_number_list.append('6')
                    product_path_list.append(f)
                    sun_elevation_list.append(sun_elevation)
                    earth_sun_distance_list.append(earth_sun_distance)
                    if '6' in scale_offset_dict:
                        scale_value_list.append(scale_offset_dict['6'][0])
                        offset_value_list.append(scale_offset_dict['6'][1])
                    k1_list.append(0)
                    k2_list.append(0)
                elif processing_level == 'l1tp' and f[-12:-4] == '6_VCID_1':
                    band_names.append(files_directories.file_name(f))
                    band_number_list.append('6_VCID_1')
                    product_path_list.append(f)
                    sun_elevation_list.append(sun_elevation)
                    earth_sun_distance_list.append(earth_sun_distance)
                    if '6_VCID_1' in k_dict:
                        k1_list.append(k_dict['6_VCID_1'][0])
                        k2_list.append(k_dict['6_VCID_1'][1])
                    scale_value_list.append(1)
                    offset_value_list.append(0)
                elif processing_level == 'l1tp' and f[-12:-4] == '6_VCID_2':
                    band_names.append(files_directories.file_name(f))
                    band_number_list.append('6_VCID_2')
                    product_path_list.append(f)
                    sun_elevation_list.append(sun_elevation)
                    earth_sun_distance_list.append(earth_sun_distance)
                    if '6_VCID_2' in k_dict:
                        k1_list.append(k_dict['6_VCID_2'][0])
                        k2_list.append(k_dict['6_VCID_2'][1])
                    scale_value_list.append(1)
                    offset_value_list.append(0)
            # temperature band 10 Landsat 8-9
            elif sensor_id.lower() == cfg.sensor_oli and f[-6:-4] == '10':
                band_names.append(files_directories.file_name(f))
                band_number_list.append('10')
                product_path_list.append(f)
                sun_elevation_list.append(sun_elevation)
                earth_sun_distance_list.append(earth_sun_distance)
                if '10' in scale_offset_dict:
                    scale_value_list.append(scale_offset_dict['10'][0])
                    offset_value_list.append(scale_offset_dict['10'][1])
                k1_list.append(0)
                k2_list.append(0)
        product_name_list = [cfg.landsat] * len(band_names)
        spacecraft_list = [spacecraft_id] * len(band_names)
    if len(scale_value_list) == 0:
        scale_value_list = [0.0000275] * len(band_names)
    if len(offset_value_list) == 0:
        offset_value_list = [-0.2] * len(band_names)
    if len(k2_list) == 0:
        k2_list = [0] * len(band_names)
    if len(k1_list) == 0:
        k1_list = [0] * len(band_names)
    if len(e_sun_list) == 0:
        e_sun_list = [0] * len(band_names)
    if len(sun_elevation_list) == 0:
        sun_elevation_list = [0] * len(band_names)
    if len(earth_sun_distance_list) == 0:
        earth_sun_distance_list = [0] * len(band_names)
    processing_level_list = [processing_level] * len(band_names)
    product_date_list = [product_date] * len(band_names)
    if len(offset_value_list) == 0:
        offset_value_list = [0] * len(band_names)
    if nodata_value:
        nodata_value_list = [nodata_value] * len(band_names)
    else:
        nodata_value_list = [np.nan] * len(band_names)
    product_table = tm.add_product_to_preprocess(
        product_list=product_name_list, spacecraft_list=spacecraft_list,
        processing_level=processing_level_list,
        band_name_list=band_names, product_path_list=product_path_list,
        scale_list=scale_value_list,
        offset_list=offset_value_list, nodata_list=nodata_value_list,
        date_list=product_date_list,
        k1_list=k1_list, k2_list=k2_list, band_number_list=band_number_list,
        e_sun_list=e_sun_list,
        sun_elevation_list=sun_elevation_list,
        earth_sun_distance_list=earth_sun_distance_list
    )
    return product_table
