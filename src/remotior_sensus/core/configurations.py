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

"""Configuration module.
Module containing shared variables and parameters across tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from remotior_sensus.core.multiprocess_manager import Multiprocess

root_name = 'remotior_sensus'
version = None
# type hint for Multiprocess
multiprocess: Multiprocess
# shared classes
logger = messages = band_calc = band_classification = band_combination = None
band_dilation = band_erosion = band_neighbor_pixels = band_pca = None
band_sieve = None
# shared Temporary class
temp = None
# variable to stop processes
action = True
# variables used in Progress class
progress = None
process = root_name
message = 'starting'
refresh_time = 1.0
# operating system information
sys_64bit = None
file_sys_encoding = None
sys_name = None
# notification options
sound_notification = None
smtp_notification = None
smtp_server = ''
smtp_user = ''
smtp_password = ''
smtp_recipients = ''
# optional GDAL path
gdal_path = None
# variables used in BandSet class
band_name_suf = '#b'
date_auto = 'auto'
# memory units used in Multiprocess class for calculating block size
memory_unit_array_12 = 0.000016
memory_unit_array_8 = 0.000010
memory_unit_array_4 = 0.000006
# number of parallel processes. used for Multiprocess calculations
n_processes = 2
# available RAM that should be used by processes
available_ram = 2048
# parameters for raster files
raster_data_type = 'Float32'
raster_compression = True
raster_compression_format = 'LZW'
# nodata values for data types
nodata_val = -32768
nodata_val_UInt16 = 65535
nodata_val_Int32 = 2147483647
nodata_val_Int64 = -9223372036854775808
nodata_val_Float32 = -3.4028235e+38
nodata_val_UInt32 = 4294967295
nodata_val_UInt64 = 2 ** 64 - 1
nodata_val_Byte = 255
# predefined suffixes
csv_suffix = '.csv'
dbf_suffix = '.dbf'
tif_suffix = '.tif'
vrt_suffix = '.vrt'
shp_suffix = '.shp'
gpkg_suffix = '.gpkg'
txt_suffix = '.txt'
xml_suffix = '.xml'
rsmo_suffix = '.rsmo'
# text delimiters
comma_delimiter = ','
tab_delimiter = '\t'
new_line = '\n'
# product variables used for download and preprocessing
sentinel2 = 'Sentinel-2'
landsat = 'Landsat'
sensor_oli = 'oli_tirs'
sensor_etm = 'etm'
sensor_tm = 'tm'
sensor_mss = 'mss'
# NASA CMR Search
# https://cmr.earthdata.nasa.gov/search/site/search_api_docs.html
landsat_hls = 'Landsat_HLS'
landsat_hls_collection = 'C2021957657-LPCLOUD'
sentinel2_hls = 'Sentinel-2_HLS'
sentinel2_hls_collection = 'C2021957295-LPCLOUD'
product_list = [sentinel2, landsat_hls, sentinel2_hls]
# satellites bands for center wavelength definition
no_satellite = 'Band order'
satGeoEye1 = 'GeoEye-1 [bands 1, 2, 3, 4]'
satGOES = 'GOES [bands 1, 2, 3, 4, 5, 6]'
satLandsat9 = 'Landsat 9 OLI [bands 1, 2, 3, 4, 5, 6, 7]'
satLandsat8 = 'Landsat 8 OLI [bands 1, 2, 3, 4, 5, 6, 7]'
satLandsat7 = 'Landsat 7 ETM+ [bands 1, 2, 3, 4, 5, 7]'
satLandsat45 = 'Landsat 4-5 TM [bands 1, 2, 3, 4, 5, 7]'
satLandsat13 = 'Landsat 1-3 MSS [bands 4, 5, 6, 7]'
satRapidEye = 'RapidEye [bands 1, 2, 3, 4, 5]'
satSentinel1 = 'Sentinel-1 [bands VV, VH]'
satSentinel2 = 'Sentinel-2 [bands 1, 2, 3, 4, 5, 6, 7, 8, 8A, 9, 10, 11, 12]'
satSentinel3 = 'Sentinel-3 [bands 1, 2, 3, 4, 5, 6, 7, 8, 9, ' \
               '10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]'
satASTER = 'ASTER [bands 1, 2, 3N, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]'
satMODIS = 'MODIS [bands 3, 4, 1, 2, 5, 6, 7]'
satMODIS2 = 'MODIS [bands 1, 2]'
satSPOT4 = 'SPOT 4 [bands 1, 2, 3, 4]'
satSPOT5 = 'SPOT 5 [bands 1, 2, 3, 4]'
satSPOT6 = 'SPOT 6 [bands 1, 2, 3, 4]'
satPleiades = 'Pleiades [bands 1, 2, 3, 4]'
satQuickBird = 'QuickBird [bands 1, 2, 3, 4]'
satWorldView23 = 'WorldView-2 -3 Multispectral [bands 1, 2, 3, 4, 5, 6, 7, 8]'
# satellite list used in BandSet class
satWlList = [
    no_satellite, satASTER, satGeoEye1, satGOES, satLandsat8, satLandsat7,
    satLandsat45, satLandsat13, satMODIS, satMODIS2, satPleiades, satQuickBird,
    satRapidEye, satSentinel2, satSentinel3, satSPOT4, satSPOT5, satSPOT6,
    satWorldView23
]
# units used for center wavelength
no_unit = 'band number'
wl_micro = 'µm (1 E-6m)'
wl_nano = 'nm (1 E-9m)'
# list of units
unit_list = [no_unit, wl_micro, wl_nano]
unit_nano = 'E-9m'
unit_micro = 'E-6m'
# wavelength center and thresholds in micrometers used in BandSet class
blue_center = 0.475
blue_threshold = 0.2
green_center = 0.56
green_threshold = 0.03
red_center = 0.65
red_threshold = 0.04
nir_center = 0.85
nir_threshold = 0.15
swir_1_center = 1.61
swir_1_threshold = 0.2
swir_2_center = 2.2
swir_2_threshold = 0.2
# dictionary of satellite bands center wavelengths
satellites = {
    # ASTER center wavelength calculated from USGS, 2015.
    # Advanced Spaceborne Thermal Emission and Reflection
    # Radiometer (ASTER) Level 1 Precision Terrain Corrected Registered
    # At-Sensor Radiance Product (AST_L1T)
    satASTER: [
        [0.560, 0.660, 0.810, 1.650, 2.165, 2.205, 2.260, 2.330, 2.395,
         8.300, 8.650, 9.100, 10.600, 11.300],
        wl_micro,
        ['01', '02', '3N', '04', '05', '06', '07', '08', '09', '10',
         '11', '12', '13', '14']],
    # Landsat center wavelength calculated from
    # http://landsat.usgs.gov/band_designations_landsat_satellites.php
    satLandsat8: [[0.44, 0.48, 0.56, 0.655, 0.865, 1.61, 2.2],
                  wl_micro, ['1', '2', '3', '4', '5', '6', '7']],
    satLandsat7: [[0.485, 0.56, 0.66, 0.835, 1.65, 2.22],
                  wl_micro, ['1', '2', '3', '4', '5', '7']],
    satLandsat45: [[0.485, 0.56, 0.66, 0.83, 1.65, 2.215],
                   wl_micro, ['1', '2', '3', '4', '5', '7']],
    satLandsat13: [[0.55, 0.65, 0.75, 0.95], wl_micro, ['4', '5', '6', '7']],
    # MODIS center wavelength calculated from
    # https://lpdaac.usgs.gov/dataset_discovery/modis
    satMODIS: [[0.469, 0.555, 0.645, 0.858, 1.24, 1.64, 2.13],
               wl_micro, ['03', '04', '01', '02', '05', '06', '07']],
    satMODIS2: [[0.645, 0.858], wl_micro, ['01', '02']],
    # RapidEye center wavelength calculated from
    # http://www.blackbridge.com/rapideye/products/ortho.htm
    satRapidEye: [[0.475, 0.555, 0.6575, 0.71, 0.805],
                  wl_micro, ['01', '02', '03', '04', '05']],
    # SPOT center wavelength calculated from
    # http://www.astrium-geo.com/en/194-resolution-and-spectral-bands
    satSPOT4: [[0.545, 0.645, 0.835, 1.665], wl_micro,
               ['01', '02', '03', '04']],
    satSPOT5: [[0.545, 0.645, 0.835, 1.665], wl_micro,
               ['01', '02', '03', '04']],
    satSPOT6: [[0.485, 0.56, 0.66, 0.825], wl_micro, ['01', '02', '03', '04']],
    # Pleiades center wavelength calculated from
    # http://www.astrium-geo.com/en/3027-pleiades-50-cm-resolution-products
    satPleiades: [[0.49, 0.56, 0.65, 0.84], wl_micro,
                  ['01', '02', '03', '04']],
    # QuickBird center wavelength calculated from
    # http://www.digitalglobe.com/resources/satellite-information
    satQuickBird: [[0.4875, 0.543, 0.65, 0.8165], wl_micro,
                   ['01', '02', '03', '04']],
    # WorldView-2 center wavelength calculated from
    # http://www.digitalglobe.com/resources/satellite-information
    satWorldView23: [
        [0.425, 0.48, 0.545, 0.605, 0.66, 0.725, 0.8325, 0.95],
        wl_micro, ['01', '02', '03', '04', '05', '06', '07', '08']],
    # GeoEye-1 center wavelength calculated from
    # http://www.digitalglobe.com/resources/satellite-information
    satGeoEye1: [[0.48, 0.545, 0.6725, 0.85], wl_micro,
                 ['01', '02', '03', '04']],
    # Sentinel-1
    satSentinel1: [[1, 2], no_unit, ['1', '2']],
    # Sentinel-2 center wavelength from
    # https://sentinel.esa.int/documents/247904/685211/Sentinel-2A
    # +MSI+Spectral+Responses
    satSentinel2: [
        [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865,
         0.945, 1.375, 1.610, 2.190], wl_micro,
        ['01', '02', '03', '04', '05', '06', '07', '08', '8a', '09',
         '10', '11', '12']],
    # Sentinel-3 center wavelength from Sentinel-3 xfdumanifest.xml
    satSentinel3: [
        [0.400, 0.4125, 0.4425, 0.490, 0.510, 0.560, 0.620, 0.665,
         0.67375, 0.68125, 0.70875, 0.75375, 0.76125,
         0.764375, 0.7675, 0.77875, 0.865, 0.885, 0.900, 0.940, 1.020],
        wl_micro,
        ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
         '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
         '21']],
    # GOES center wavelength from GOES-R, 2017.PRODUCT DEFINITION
    # AND USER’S GUIDE (PUG) VOLUME 3: LEVEL 1B PRODUCTS
    satGOES: [[0.47, 0.64, 0.87, 1.38, 1.61, 2.25], wl_micro,
              ['01', '02', '03', '04', '05', '06']]
}
# variable used for array name placeholder in expressions for calculations
array_function_placeholder = '_array_function_placeholder'
# reclassification name variables
old_value = 'old_value'
new_value = 'new_value'
variable_raster_name = 'raster'
# calculation band name alias
variable_bandset_name = 'bandset'
variable_band_name = 'b'
variable_band_quotes = '"'
variable_band = '#BAND#'
variable_current_bandset = '#'
variable_output_separator = '@'
variable_bandset_number_separator = '%'
variable_all = '*'
variable_blue_name = '#BLUE#'
variable_green_name = '#GREEN#'
variable_red_name = '#RED#'
variable_nir_name = '#NIR#'
variable_swir1_name = '#SWIR1#'
variable_swir2_name = '#SWIR2#'
variable_ndvi_name = '#NDVI#'
variable_ndvi_expression = '("#NIR#" - "#RED#") / ("#NIR#" + "#RED#")'
expression_alias = [[variable_ndvi_name, variable_ndvi_expression]]
variable_output_name_bandset = '#BANDSET#'
variable_output_name_date = '#DATE#'
variable_output_temporary = 'temp'
forbandsinbandset = 'forbandsinbandset'
forbandsets = 'forbandsets'
calc_function_name = '!function!'
calc_date_format = '%Y-%m-%d'
default_output_name = 'output'
stat_percentile = '@stat_percentile@'
statistics_list = [
    ['Count', 'np.count_nonzero(~np.isnan(array))'],
    ['Max', 'np.nanmax(array)'], ['Mean', 'np.nanmean(array)'],
    ['Median', 'np.nanmedian(array)'], ['Min', 'np.nanmin(array)'],
    ['Percentile', 'np.nanpercentile(array, %s)' % stat_percentile],
    ['StandardDeviation', 'np.nanstd(array)'], ['Sum', 'np.nansum(array)']
]
# calculation data types used in calculations
float64_dt = 'Float64'
float32_dt = 'Float32'
int32_dt = 'Int32'
uint32_dt = 'UInt32'
int16_dt = 'Int16'
uint16_dt = 'UInt16'
byte_dt = 'Byte'
datatype_list = [float64_dt, float32_dt, int32_dt, uint32_dt, int16_dt,
                 uint16_dt, byte_dt]
# variables used in spectral signatures
uid_field_name = 'roi_id'
macroclass_field_name = 'macroclass_id'
class_field_name = 'class_id'
macroclass_default = 'macroclass'
# input normalization for classification
z_score = 'z score'
linear_scaling = 'linear scaling'
# classification frameworks
classification_framework = 'classification_framework'
scikit_framework = 'scikit'
pytorch_framework = 'pytorch'
spectral_signatures_framework = 'spectral_signatures'
model_classifier_framework = 'model_classifier'
normalization_values_framework = 'normalization_values'
covariance_matrices_framework = 'covariance_matrices'
algorithm_name_framework = 'algorithm_name'
input_normalization_framework = 'input_normalization'
# classification algorithm names
minimum_distance = 'minimum distance'
maximum_likelihood = 'maximum likelihood'
spectral_angle_mapping = 'spectral angle mapping'
random_forest = 'random forest'
random_forest_ovr = 'random forest ovr'
support_vector_machine = 'support vector machine'
multi_layer_perceptron = 'multi-layer perceptron'
pytorch_multi_layer_perceptron = 'pytorch multi-layer perceptron'
classification_algorithms = [
    minimum_distance, maximum_likelihood, spectral_angle_mapping,
    random_forest, random_forest_ovr, support_vector_machine,
    multi_layer_perceptron, pytorch_multi_layer_perceptron
]
# name used in raster conversion to vector for area field
area_field_name = 'area'
