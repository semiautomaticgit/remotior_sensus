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
"""Download products.

This tool allows for downloading products such as Landsat
and Sentinel-2 datasets.
"""

import datetime
import json
import requests
from xml.dom import minidom
from xml.etree import cElementTree

from remotior_sensus.core import configurations as cfg, table_manager as tm
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.util import (
    download_tools, raster_vector, files_directories, dates_times,
    read_write_files
)


def search(
        product, date_from, date_to, max_cloud_cover=100, result_number=50,
        name_filter=None, coordinate_list=None, progress_message=True,
        proxy_host=None, proxy_port=None, proxy_user=None, proxy_password=None,
        copernicus_user=None, copernicus_password=None
) -> OutputManager:
    """Perform the query of image databases.

    It allows for the search of image products, currently Landsat and Sentinel-2.
    """  # noqa: E501
    result = OutputManager(check=False)
    if product == cfg.sentinel2:
        result = query_sentinel_2_database(
            date_from=date_from, date_to=date_to,
            max_cloud_cover=max_cloud_cover,
            result_number=result_number, name_filter=name_filter,
            coordinate_list=coordinate_list, progress_message=progress_message,
            proxy_host=proxy_host, proxy_port=proxy_port,
            proxy_user=proxy_user, proxy_password=proxy_password,
            copernicus_user=copernicus_user,
            copernicus_password=copernicus_password
        )
    elif product == cfg.landsat_hls or product == cfg.sentinel2_hls:
        result = query_nasa_cmr(
            product=product, date_from=date_from, date_to=date_to,
            max_cloud_cover=max_cloud_cover, result_number=result_number,
            name_filter=name_filter, coordinate_list=coordinate_list,
            progress_message=progress_message, proxy_host=proxy_host,
            proxy_port=proxy_port, proxy_user=proxy_user,
            proxy_password=proxy_password
        )
    return result


def query_sentinel_2_database(
        date_from, date_to, max_cloud_cover=100, result_number=50,
        name_filter=None, coordinate_list=None, progress_message=True,
        copernicus_user=None, copernicus_password=None,
        proxy_host=None, proxy_port=None, proxy_user=None, proxy_password=None

) -> OutputManager:
    """Perform the query of Sentinel-2 database.

    This tool performs the query of Sentinel-2 database.

    Query using Copernicus Data Space Ecosystem API
    https://documentation.dataspace.copernicus.eu/#/APIs/OData
    (from https://documentation.dataspace.copernicus.eu:
    'Copernicus Data Space Ecosystem represents an overall and comprehensive
    data ecosystem accessible via web portal, applications and APIs.
    ...
    The Copernicus Data Space Ecosystem provides the essential data
    and service offering for everyone to use,
    for commercial and non-commercial purposes').

    Sentinel-2 metadata are downloaded through the following Google service:
    https://storage.googleapis.com/gcp-public-data-sentinel-2 .

    Args:
        date_from: date defining the starting period of the query.
        date_to:
        max_cloud_cover:
        result_number:
        name_filter:
        coordinate_list: list [left, top, right, bottom] WGS84 coordinates.
        progress_message: progress message
        copernicus_user:
        copernicus_password:
        proxy_host: proxy host.
        proxy_port: proxy port.
        proxy_user: proxy user.
        proxy_password: proxy password.

    Returns:
        object :func:`~remotior_sensus.core.output_manager.OutputManager`

    """
    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(process='search', message='starting', start=True)
    cfg.logger.log.debug(
        'date_from: %s, date_to: %s; max_cloud_cover: %s; result_number: %s; '
        'name_filter: %s; coordinate_list: %s'
        % (str(date_from), str(date_to), str(max_cloud_cover),
           str(result_number), str(name_filter), str(coordinate_list))
    )
    image_find_list = []
    # without filtering the results
    if name_filter is None:
        image_find_list.append('s2')
        final_query = 'S2'
    # filter the results based on a string
    else:
        name_filter_split = name_filter.replace(' ', '').split(',')
        for f in name_filter_split:
            image_find_list.append(f)
        # if one string then include it in the query
        if len(image_find_list) == 1:
            final_query = image_find_list[0]
        else:
            final_query = 'S2'
    # coordinate list left, top, right, bottom
    if coordinate_list is not None:
        if abs(coordinate_list[0] - coordinate_list[2]) > 10 or abs(
                coordinate_list[1] - coordinate_list[3]
        ) > 10:
            cfg.logger.log.warning('search area extent beyond limits')
            cfg.messages.warning('search area extent beyond limits')
    if copernicus_password is None:
        base_url = 'https://storage.googleapis.com/gcp-public-data-sentinel-2'
        copernicus = False
        access_token = session_state = None
    else:
        base_url = 'https://catalogue.dataspace.copernicus.eu/odata/v1'
        copernicus = True
        access_token, session_state = get_copernicus_token(
            user=copernicus_user, password=copernicus_password,
            proxy_host=proxy_host, proxy_port=proxy_port,
            proxy_user=proxy_user, proxy_password=proxy_password
        )
    product_table_list = []
    # loop the results
    e = 0
    max_result_number = 50
    if max_result_number > result_number:
        max_result_number = result_number
    for _results in range(0, result_number, max_result_number):
        if coordinate_list is None:
            # get level 2A
            url = ''.join(
                ['https://catalogue.dataspace.copernicus.eu/odata/v1',
                 '/Products?$filter=contains(Name,%27', str(final_query),
                 '%27)%20', 'and%20Attributes/OData.CSC.StringAttribute/',
                 "any(att:att/Name%20eq%20'productType'%20"
                 "and%20att/OData.CSC.StringAttribute/Value%20eq%20'S2MSI2A')",
                 '%20and%20ContentDate/Start%20gt%20', str(date_from),
                 'T00:00:00.000Z%20and%20ContentDate/Start%20lt%20',
                 str(date_to), 'T21:42:55.721Z',
                 '%20and%20Attributes/OData.CSC.DoubleAttribute/any(att:att/',
                 'Name%20eq%20%27cloudCover%27%20and%20att',
                 '/OData.CSC.DoubleAttribute/Value%20le%20',
                 str(max_cloud_cover), ')&$orderby=',
                 'ContentDate/Start%20asc&$expand=Attributes&$count=True&',
                 '$top=', str(max_result_number), '&$skip=', str(_results)]
            )
        else:
            url = ''.join(
                ['https://catalogue.dataspace.copernicus.eu/odata/v1',
                 "/Products?$filter=Collection/Name%20eq%20'SENTINEL-2'%20",
                 'and%20ContentDate/Start%20gt%20', str(date_from),
                 'T00:00:00.000Z%20and%20ContentDate/Start%20lt%20',
                 str(date_to), 'T21:42:55.721Z',
                 '%20and%20Attributes/OData.CSC.DoubleAttribute/any(att:att/',
                 'Name%20eq%20%27cloudCover%27%20and%20att',
                 '/OData.CSC.DoubleAttribute/Value%20le%20',
                 str(max_cloud_cover), ')',
                 '%20and%20OData.CSC.Intersects(area=geography%27SRID=4326;',
                 'POLYGON%20((',
                 str(coordinate_list[0]), '%20', str(coordinate_list[1]), ',',
                 str(coordinate_list[0]), '%20', str(coordinate_list[3]), ',',
                 str(coordinate_list[2]), '%20', str(coordinate_list[3]), ',',
                 str(coordinate_list[2]), '%20', str(coordinate_list[1]), ',',
                 str(coordinate_list[0]), '%20', str(coordinate_list[1]),
                 '))%27)',
                 '&$orderby=',
                 'ContentDate/Start%20asc&$expand=Attributes&$count=True&',
                 '$top=', str(max_result_number), '&$skip=', str(_results)]
            )
        # download json
        json_file = cfg.temp.temporary_file_path(name_suffix='.json')
        check = cfg.multiprocess.multi_download_file(
            url_list=[url], output_path_list=[json_file],
            message='submitting request', progress=False, timeout=60
        )
        if check is not False:
            try:
                with open(json_file) as json_search:
                    doc = json.load(json_search)
            except Exception as err:
                cfg.logger.log.error(str(err))
                cfg.messages.error(str(err))
                return OutputManager(check=False)
            entries = doc['value']
            if len(entries) == 0:
                break
            for entry in entries:
                e += 1
                cfg.progress.update(
                    message='search in progress', step=e, percentage=0,
                    steps=result_number, minimum=10, maximum=90, ping=True
                )
                product_name = entry['Name'].replace('.SAFE', '')
                uid = entry['Id']
                # online
                _online = entry['Online']
                # relative path
                _path = entry['S3Path']
                img_acquisition_date = entry['ContentDate']['Start']
                if 'MSIL2A' in product_name:
                    product_type = 'L2A'
                else:
                    product_type = 'L1C'
                cloud_cover_percentage = None
                for att in entry['Attributes']:
                    if att['Name'] == 'cloudCover':
                        cloud_cover_percentage = att['Value']
                        break
                # generally available depending on version
                footprint_coord = entry['Footprint'].split('((')[1].replace(
                    ')', ''
                ).replace("'", '')
                x_list = []
                y_list = []
                for pair in footprint_coord.split(','):
                    c = pair.lstrip().split(' ')
                    x_list.append(
                        float(c[0].replace('(', '').replace(')', ''))
                    )
                    y_list.append(
                        float(c[1].replace('(', '').replace(')', ''))
                    )
                min_lon = min(x_list)
                max_lon = max(x_list)
                min_lat = min(y_list)
                max_lat = max(y_list)
                # download Sentinel metadata
                if product_type == 'L1C':
                    if copernicus is True:
                        url_2 = (
                                '%s/Products(%s)/Nodes(%s.SAFE)'
                                '/Nodes(MTD_MSIL1C.xml)/$value'
                                % (base_url, uid, product_name)
                        )
                    else:
                        url_2 = ''.join(
                            [base_url, '/tiles/', product_name[39:41], '/',
                             product_name[41], '/', product_name[42:44], '/',
                             product_name, '.SAFE/MTD_MSIL1C.xml']
                        )
                else:
                    if copernicus is True:
                        url_2 = (
                                '%s/Products(%s)/Nodes(%s.SAFE)'
                                '/Nodes(MTD_MSIL2A.xml)/$value'
                                % (base_url, uid, product_name)
                        )
                    else:
                        url_2 = ''.join(
                            [base_url, '/L2/tiles/', product_name[39:41], '/',
                             product_name[41], '/', product_name[42:44], '/',
                             product_name, '.SAFE/MTD_MSIL2A.xml']
                        )
                # download metadata xml
                xml_file = cfg.temp.temporary_file_path(name_suffix='.xml')
                check_2 = cfg.multiprocess.multi_download_file(
                    url_list=[url_2], output_path_list=[xml_file],
                    progress=False, timeout=2, copernicus=copernicus,
                    access_token=access_token, proxy_host=proxy_host,
                    proxy_port=proxy_port, proxy_user=proxy_user,
                    proxy_password=proxy_password
                )
                if check_2:
                    try:
                        xml_parse = minidom.parse(xml_file)
                        try:
                            image_name_tag = xml_parse.getElementsByTagName(
                                'IMAGE_FILE'
                            )[0]
                        except Exception as err:
                            str(err)
                            image_name_tag = xml_parse.getElementsByTagName(
                                'IMAGE_FILE_2A'
                            )[0]
                        image_name = image_name_tag.firstChild.data.split('/')[
                            1]
                        # search images
                        for f in image_find_list:
                            if (f.lower() in product_name.lower()
                                    or f.lower() in image_name.lower()):
                                image_zone = image_name.split('_')[1][1:]
                                if product_type == 'L1C':
                                    img_name_3 = (
                                        image_name_tag.firstChild.data.split(
                                            '/'
                                        )[3].split('_'))
                                    pvi_name = '%s_%s_PVI.jp2' % (
                                        img_name_3[0], img_name_3[1])
                                    img_preview = ''.join(
                                        [base_url, '/tiles/',
                                         product_name[39:41], '/',
                                         product_name[41], '/',
                                         product_name[42:44], '/',
                                         product_name, '.SAFE/GRANULE/',
                                         image_name,
                                         '/QI_DATA/', pvi_name]
                                    )
                                else:
                                    img_name_3 = (
                                        image_name_tag.firstChild.data.split(
                                            '/'
                                        )[4].split('_'))
                                    pvi_name = '%s_%s_PVI.jp2' % (
                                        img_name_3[0], img_name_3[1])
                                    img_preview = ''.join(
                                        [base_url, '/L2/tiles/',
                                         product_name[39:41], '/',
                                         product_name[41], '/',
                                         product_name[42:44], '/',
                                         product_name, '.SAFE/GRANULE/',
                                         image_name, '/QI_DATA/', pvi_name]
                                    )
                                product_table_list.append(
                                    tm.create_product_table(
                                        product=cfg.sentinel2,
                                        product_id=product_name,
                                        acquisition_date=img_acquisition_date,
                                        cloud_cover=float(
                                            cloud_cover_percentage
                                        ), zone_path=image_zone, row=None,
                                        min_lat=float(min_lat),
                                        min_lon=float(min_lon),
                                        max_lat=float(max_lat),
                                        max_lon=float(max_lon),
                                        collection=None, size=None,
                                        preview=img_preview, uid=uid,
                                        image=image_name
                                    )
                                )
                                break
                    except Exception as err:
                        str(err)
                        for f in image_find_list:
                            if f.lower() in product_name.lower():
                                product_table_list.append(
                                    tm.create_product_table(
                                        product=cfg.sentinel2,
                                        product_id=product_name,
                                        acquisition_date=img_acquisition_date,
                                        cloud_cover=float(
                                            cloud_cover_percentage
                                        ), zone_path=None, row=None,
                                        min_lat=float(min_lat),
                                        min_lon=float(min_lon),
                                        max_lat=float(max_lat),
                                        max_lon=float(max_lon),
                                        collection=None, size=None,
                                        preview=None, uid=uid,
                                        image=product_name
                                    )
                                )
                                cfg.logger.log.warning(
                                    'failed to get metadata: %s' % url_2
                                )
                                cfg.messages.warning(
                                    'warning: failed to get metadata %s'
                                    % url_2
                                )
                                break
        else:
            cfg.logger.log.error('error: search failed')
            cfg.messages.error('error: search failed')
            return OutputManager(check=False)
    if access_token is not None:
        delete_copernicus_token(
            access_token, session_state, proxy_host=proxy_host,
            proxy_port=proxy_port, proxy_user=proxy_user,
            proxy_password=proxy_password
        )
    cfg.progress.update(end=True)
    cfg.logger.log.info('end')
    return OutputManager(
        extra={
            'product_table': tm.stack_product_table(
                product_list=product_table_list
            )
        }
    )


def get_copernicus_token(
        user, password, authentication_uri=None, proxy_host=None,
        proxy_port=None, proxy_user=None, proxy_password=None
):
    if authentication_uri is None:
        authentication_uri = (
            'https://identity.dataspace.copernicus.eu/auth/realms/CDSE'
            '/protocol/openid-connect/token'
        )
    # noinspection SpellCheckingInspection
    data = {
        'client_id': 'cdse-public', 'grant_type': 'password',
        'username': user, 'password': password,
    }
    if proxy_host is None:
        proxies = None
    else:
        proxies = {
            'https': 'https://%s:%s@%s:%s' % (
                proxy_user, proxy_password, proxy_host, proxy_port
            )
        }
    response = requests.post(
        authentication_uri, data=data, verify=True, allow_redirects=False,
        proxies=proxies
        )
    response_text = json.loads(response.text)
    access_token = response_text['access_token']
    session_state = response_text['session_state']
    return access_token, session_state


def delete_copernicus_token(
        access_token, session_state, proxy_host=None,
        proxy_port=None, proxy_user=None, proxy_password=None
):
    if proxy_host is None:
        proxies = None
    else:
        proxies = {
            'https': 'https://%s:%s@%s:%s' % (
                proxy_user, proxy_password, proxy_host, proxy_port
            )
        }
    session = requests.Session()
    session.headers['Authorization'] = 'Bearer %s' % access_token
    if proxies is not None:
        session.proxies = proxies
    session_url = (
            'https://identity.dataspace.copernicus.eu/auth/realms/CDSE'
            '/account/sessions/%s' % session_state
    )
    session.headers['Authorization'] = 'Bearer %s' % access_token
    session.headers['Content-Type'] = 'application/json'
    session.headers['Accept'] = '*/*'
    session.headers['Sec-Fetch-Mode'] = 'cors'
    session.delete(session_url)
    return True


def download(
        product_table, output_path, exporter=False, band_list=None,
        virtual_download=False, extent_coordinate_list=None, proxy_host=None,
        proxy_port=None, proxy_user=None, proxy_password=None,
        authentication_uri=None, nasa_user=None, nasa_password=None,
        copernicus_user=None, copernicus_password=None,
        progress_message=True
) -> OutputManager:
    """Download products.

    This tool downloads product.
    Downloads Sentinel-2 images using the following Google service:
    https://storage.googleapis.com/gcp-public-data-sentinel-2 or
    https://catalogue.dataspace.copernicus.eu 
    if copernicus_user and copernicus_password are provided.
    
    Downloads HLS images from:
    https://cmr.earthdata.nasa.gov/search/site/search_api_docs.html.

    Args:
        product_table: product table object.
        output_path: string of output path.
        exporter: if True, export download urls.
        band_list: list of band numbers to be downloaded.
        virtual_download: if True create a virtual raster of the linked image.
        extent_coordinate_list: list of coordinates for defining a subset region [left, top, right, bottom] .
        proxy_host: proxy host.
        proxy_port: proxy port.
        proxy_user: proxy user.
        proxy_password: proxy password.
        authentication_uri: authentication uri.
        nasa_user: user for NASA authentication.
        nasa_password: password for NASA authentication.
        copernicus_user: user for Copernicus authentication.
        copernicus_password: password for Copernicus authentication.
        progress_message: progress message.

    Returns:
        object :func:`~remotior_sensus.core.output_manager.OutputManager` with
            - paths = output file list
            - extra={'directory_paths': list of output directory paths}

    """  # noqa: E501
    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(
            process=__name__.split('.')[-1].replace('_', ' '),
            message='starting', start=True
        )
    if band_list is None:
        band_list = cfg.satellites[cfg.satSentinel2][2]
    # list of output files
    output_file_list = []
    # list of output directories
    output_directory_list = []
    total_products = product_table.shape[0]
    try:
        progress_step = 100 / (len(band_list) * total_products)
    except Exception as err:
        progress_step = 1
        str(err)
    min_progress = 0
    max_progress = min_progress + progress_step
    if copernicus_password is not None:
        access_token, session_state = get_copernicus_token(
            user=copernicus_user, password=copernicus_password,
            proxy_host=proxy_host, proxy_port=proxy_port,
            proxy_user=proxy_user, proxy_password=proxy_password
        )
    else:
        access_token = session_state = None
    for i in range(total_products):
        cloud_mask_gml = None
        if (product_table['product'][i] == cfg.sentinel2
                and copernicus_password is not None):
            top_url = (
                'https://catalogue.dataspace.copernicus.eu/odata/v1'
            )
            product_name = product_table['product_id'][i]
            acquisition_date = product_table['acquisition_date'][i]
            image_name = product_table['image'][i]
            uid = product_table['uid'][i]
            # download ancillary data MSI, TL and cloud mask GML
            if image_name[0:4] == 'L1C_':
                base_output_dir = '%s/%s_%s' % (
                    output_path, image_name, str(acquisition_date))
                metadata_msi = base_output_dir + '/MTD_MSIL1C.xml'
                metadata_msi_url = (
                        '%s/Products(%s)/Nodes(%s.SAFE)/Nodes(MTD_MSIL1C.xml)'
                        '/$value' % (top_url, uid, product_name)
                )
            else:
                base_output_dir = '%s/%s_%s' % (
                    output_path, image_name, str(acquisition_date))
                metadata_msi = base_output_dir + '/MTD_MSIL2A.xml'
                metadata_msi_url = (
                        '%s/Products(%s)/Nodes(%s.SAFE)/Nodes(MTD_MSIL2A.xml)'
                        '/$value' % (top_url, uid, product_name)
                )
            output_directory_list.append(base_output_dir)
            # check connection downloading metadata xml
            temp_file = cfg.temp.temporary_file_path(name_suffix='.xml')
            check = cfg.multiprocess.multi_download_file(
                url_list=[metadata_msi_url], output_path_list=[temp_file],
                proxy_host=proxy_host, proxy_port=proxy_port,
                proxy_user=proxy_user, proxy_password=proxy_password,
                access_token=access_token, copernicus=True, progress=False,
                timeout=2
            )
            if exporter:
                output_file_list.extend(
                    [metadata_msi_url]
                )
            else:
                if check:
                    files_directories.move_file(
                        in_path=temp_file, out_path=metadata_msi
                    )
            # download bands
            for band in band_list:
                _check_sentinel_2_bands(
                    band_number=band, product_name=product_name,
                    image_name=image_name, output_path=base_output_dir,
                    output_list=output_file_list, exporter=exporter,
                    progress=progress_message, uid=uid,
                    virtual_download=virtual_download,
                    extent_coordinate_list=extent_coordinate_list,
                    proxy_host=proxy_host, proxy_port=proxy_port,
                    proxy_user=proxy_user, proxy_password=proxy_password,
                    min_progress=min_progress, max_progress=max_progress,
                    access_token=access_token
                )
                min_progress += progress_step
                max_progress += progress_step
        elif product_table['product'][i] == cfg.sentinel2:
            top_url = (
                'https://storage.googleapis.com/gcp-public-data-sentinel-2'
            )
            product_name = product_table['product_id'][i]
            acquisition_date = product_table['acquisition_date'][i]
            image_name = product_table['image'][i]
            # download ancillary data MSI, TL and cloud mask GML
            if image_name[0:4] == 'L1C_':
                base_output_dir = '%s/%s_%s' % (
                    output_path, image_name, str(acquisition_date))
                metadata_msi = base_output_dir + '/MTD_MSIL1C.xml'
                metadata_tl = base_output_dir + '/MTD_TL.xml'
                cloud_mask_gml = base_output_dir + '/MSK_CLOUDS_B00.gml'
                base_url = ''.join(
                    [top_url, '/tiles/', product_name[39:41], '/',
                     product_name[41], '/',
                     product_name[42:44], '/', product_name]
                )
                metadata_msi_url = base_url + '.SAFE/MTD_MSIL1C.xml'
                metadata_tl_url = '%s.SAFE/GRANULE/%s/MTD_TL.xml' % (
                    base_url, image_name)
                cloud_mask_gml_url = \
                    '%s.SAFE/GRANULE/%s/QI_DATA/MSK_CLOUDS_B00.gml' % (
                        base_url, image_name)
            elif image_name[0:4] == 'L2A_':
                base_output_dir = '%s/%s_%s' % (
                    output_path, image_name, str(acquisition_date))
                metadata_msi = base_output_dir + '/MTD_MSIL2A.xml'
                metadata_tl = base_output_dir + '/MTD_TL.xml'
                # cloud_mask_gml = base_output_dir + '/MSK_CLOUDS_B00.gml'
                base_url = ''.join(
                    [top_url, '/L2/tiles/', product_name[39:41], '/',
                     product_name[41], '/',
                     product_name[42:44], '/', product_name]
                )
                metadata_msi_url = base_url + '.SAFE/MTD_MSIL2A.xml'
                metadata_tl_url = '%s.SAFE/GRANULE/%s/MTD_TL.xml' % (
                    base_url, image_name)
                cloud_mask_gml_url = \
                    '%s.SAFE/GRANULE/%s/QI_DATA/MSK_CLOUDS_B00.gml' % (
                        base_url, image_name)
            # old structure version
            else:
                base_output_dir = '%s/%s_%s' % (
                    output_path, image_name[0:-7], str(acquisition_date))
                metadata_msi = base_output_dir + '/MTD_SAFL1C.xml'
                metadata_tl = base_output_dir + '_MTD_L1C.xml'
                cloud_mask_gml = base_output_dir + 'MSK_CLOUDS_B00.gml'
                base_url = ''.join(
                    [top_url, '/tiles/', product_name[39:41], '/',
                     product_name[41], '/',
                     product_name[42:44], '/', product_name, '.SAFE/']
                )
                metadata_msi_url = '%s%s.xml' % (
                    base_url,
                    product_name.replace('_PRD_MSIL1C_', '_MTD_SAFL1C_'))
                metadata_tl_url = '%s%s/GRANULE/%s.xml' % (
                    base_url, image_name,
                    image_name[0:-7].replace('_MSI_L1C_', '_MTD_L1C_'))
                cloud_mask_gml_url = \
                    '%s%s/GRANULE/QI_DATA/%s_B00_MSIL1C.gml' % (
                        base_url, image_name,
                        image_name[0:-7].replace(
                            '_MSI_L1C_TL_', '_MSK_CLOUDS_'
                        ))
            output_directory_list.append(base_output_dir)
            # check connection downloading metadata xml
            temp_file = cfg.temp.temporary_file_path(name_suffix='.xml')
            check = cfg.multiprocess.multi_download_file(
                url_list=[metadata_msi_url], output_path_list=[temp_file],
                proxy_host=proxy_host,
                proxy_port=proxy_port, proxy_user=proxy_user,
                proxy_password=proxy_password, progress=False, timeout=1
            )
            if exporter:
                output_file_list.extend(
                    [metadata_msi_url, metadata_tl_url, cloud_mask_gml_url]
                )
            else:
                if check:
                    files_directories.move_file(
                        in_path=temp_file, out_path=metadata_msi
                    )
                    cfg.multiprocess.multi_download_file(
                        url_list=[metadata_tl_url],
                        output_path_list=[metadata_tl],
                        proxy_host=proxy_host,
                        proxy_port=proxy_port, proxy_user=proxy_user,
                        proxy_password=proxy_password, progress=False,
                        timeout=2
                    )
                    if cloud_mask_gml:
                        cfg.multiprocess.multi_download_file(
                            url_list=[cloud_mask_gml_url],
                            output_path_list=[cloud_mask_gml],
                            proxy_host=proxy_host, proxy_port=proxy_port,
                            proxy_user=proxy_user,
                            proxy_password=proxy_password, progress=False,
                            timeout=2
                        )
            # download bands
            for band in band_list:
                _check_sentinel_2_bands(
                    band_number=band, product_name=product_name,
                    image_name=image_name, output_path=base_output_dir,
                    output_list=output_file_list, exporter=exporter,
                    progress=progress_message,
                    virtual_download=virtual_download,
                    extent_coordinate_list=extent_coordinate_list,
                    proxy_host=proxy_host, proxy_port=proxy_port,
                    proxy_user=proxy_user, proxy_password=proxy_password,
                    min_progress=min_progress, max_progress=max_progress
                )
                min_progress += progress_step
                max_progress += progress_step
        elif (product_table['product'][i] == cfg.sentinel2_hls
              or product_table['product'][i] == cfg.landsat_hls):
            product_url = None
            if authentication_uri is None:
                authentication_uri = 'urs.earthdata.nasa.gov'
            product_name = product_table['product_id'][i]
            image_name = product_table['image'][i]
            acquisition_date = product_table['acquisition_date'][i]
            if product_table['product'][i] == cfg.sentinel2_hls:
                top_url = (
                    'https://data.lpdaac.earthdatacloud.nasa.gov'
                    '/lp-prod-protected'
                )
                # noinspection SpellCheckingInspection
                product_url = '%s/HLSS30.020/%s/%s' % (
                    top_url, image_name, image_name)
            elif product_table['product'][i] == cfg.landsat_hls:
                top_url = (
                    'https://data.lpdaac.earthdatacloud.nasa.gov'
                    '/lp-prod-protected'
                )
                product_url = '%s/HLSL30.020/%s/%s' % (
                    top_url, image_name, image_name)
            base_output_dir = '%s/%s_%s' % (
                output_path, product_name.replace('.', '_'),
                str(acquisition_date))
            output_directory_list.append(base_output_dir)
            # download bands
            for band in band_list:
                if product_table['product'][i] == cfg.landsat_hls:
                    if ('8A' in str(band) or '8' in str(band)
                            or '12' in str(band)):
                        band = None
                if band is not None:
                    url = '%s.B%s%s' % (
                        product_url, str(band).upper().zfill(2),
                        cfg.tif_suffix)
                    if exporter:
                        output_file_list.append(url)
                    else:
                        output_file = '%s/%s_B%s%s' % (
                            base_output_dir, product_name.replace('.', '_'),
                            str(band).zfill(2), cfg.tif_suffix)
                        cfg.multiprocess.multi_download_file(
                            url_list=[url],
                            output_path_list=[output_file],
                            authentication_uri=authentication_uri,
                            user=nasa_user, password=nasa_password,
                            proxy_host=proxy_host, proxy_port=proxy_port,
                            proxy_user=proxy_user,
                            proxy_password=proxy_password,
                            progress=progress_message,
                            message='downloading band %s' % str(band),
                            min_progress=min_progress,
                            max_progress=max_progress, timeout=2
                        )
                        if files_directories.is_file(output_file):
                            output_file_list.append(output_file)
                            cfg.logger.log.debug(
                                'downloaded file %s' % output_file
                            )
                        else:
                            cfg.messages.error(
                                'failed download %s_B%s'
                                % (image_name[0:-7], band)
                            )
                            cfg.logger.log.error(
                                'failed download %s_B%s'
                                % (image_name[0:-7], band)
                            )
                        if not files_directories.is_file(
                                '%s/metadata.txt' % base_output_dir
                        ):
                            try:
                                with open(
                                        '%s/metadata.txt' % base_output_dir,
                                        'w'
                                ) as file:
                                    file.write(
                                        str(product_table['product'][i])
                                        + cfg.new_line
                                        + str(
                                            product_table[
                                                'acquisition_date'
                                            ][i]
                                            )
                                    )
                            except Exception as err:
                                str(err)
                min_progress += progress_step
                max_progress += progress_step
    if access_token is not None:
        delete_copernicus_token(
            access_token, session_state, proxy_host=proxy_host,
            proxy_port=proxy_port, proxy_user=proxy_user,
            proxy_password=proxy_password
        )
    if exporter:
        output_csv_file = '%s/links%s%s' % (
            output_path, dates_times.get_time_string(), cfg.csv_suffix)
        text = cfg.new_line.join(output_file_list)
        read_write_files.write_file(data=text, output_path=output_csv_file)
        cfg.progress.update(end=True)
        cfg.logger.log.info('end')
        return OutputManager(path=output_csv_file)
    else:
        if len(output_file_list) > 0:
            cfg.progress.update(end=True)
            cfg.logger.log.info('end')
            return OutputManager(
                paths=output_file_list,
                extra={'directory_paths': output_directory_list}
            )
        else:
            return OutputManager(check=False)


def _check_sentinel_2_bands(
        band_number, product_name, image_name, output_path, uid=None,
        output_list=None, exporter=False, progress=True,
        virtual_download=False, extent_coordinate_list=None,
        proxy_host=None, proxy_port=None, proxy_user=None, proxy_password=None,
        min_progress=0, max_progress=100, access_token=None
):
    """Checks and download Sentinel-2 bands.

    Checks and download Sentinel-2 bands using the service https://storage.googleapis.com/gcp-public-data-sentinel-2 .

        Args:
        extent_coordinate_list: list of coordinates for defining a subset region [left, top, right, bottom]
    """  # noqa: E501
    if access_token is None:
        top_url = 'https://storage.googleapis.com/gcp-public-data-sentinel-2'
        copernicus = False
    else:
        top_url = 'https://catalogue.dataspace.copernicus.eu/odata/v1'
        copernicus = True
    band_url = ''
    output_file = ''
    if image_name[0:4] == 'L1C_':
        if copernicus is False:
            band_url = ''.join(
                [top_url, '/tiles/', product_name[39:41], '/',
                 product_name[41], '/', product_name[42:44], '/',
                 product_name, '.SAFE', '/GRANULE/', image_name, '/IMG_DATA/',
                 image_name.split('_')[1], '_',
                 product_name.split('_')[2], '_B', band_number, '.jp2']
            )
        else:
            band_url = ''.join(
                [top_url, '/Products(', uid, ')/Nodes(', product_name,
                 '.SAFE)/Nodes(GRANULE)/Nodes(', image_name,
                 ')/Nodes(IMG_DATA)/Nodes(', image_name.split('_')[1], '_',
                 product_name.split('_')[2], '_B', band_number, '.jp2)/$value']
            )
        output_file = '%s/L1C_%s_B%s' % (
            output_path, image_name[4:], band_number)
    elif image_name[0:4] == 'L2A_':
        if band_number in ['02', '03', '04', '08']:
            if copernicus is False:
                band_url = ''.join(
                    [top_url, '/L2/tiles/', product_name[39:41], '/',
                     product_name[41], '/', product_name[42:44], '/',
                     product_name, '.SAFE', '/GRANULE/', image_name,
                     '/IMG_DATA/R10m/', image_name.split('_')[1], '_',
                     product_name.split('_')[2], '_B', band_number, '_10m.jp2']
                )
            else:
                band_url = ''.join(
                    [top_url, '/Products(', uid, ')/Nodes(', product_name,
                     '.SAFE)/Nodes(GRANULE)/Nodes(', image_name,
                     ')/Nodes(IMG_DATA)/Nodes(R10m)/Nodes(',
                     image_name.split('_')[1], '_',
                     product_name.split('_')[2], '_B', band_number,
                     '_10m.jp2)/$value']
                )
            output_file = '%s/%s_B%s' % (
                output_path, image_name[4:], band_number)
        elif band_number in ['05', '06', '07', '11', '12', '8A']:
            if copernicus is False:
                band_url = ''.join(
                    [top_url, '/L2/tiles/', product_name[39:41], '/',
                     product_name[41], '/', product_name[42:44], '/',
                     product_name, '.SAFE', '/GRANULE/', image_name,
                     '/IMG_DATA/R20m/', image_name.split('_')[1], '_',
                     product_name.split('_')[2], '_B', band_number, '_20m.jp2']
                )
            else:
                band_url = ''.join(
                    [top_url, '/Products(', uid, ')/Nodes(', product_name,
                     '.SAFE)/Nodes(GRANULE)/Nodes(', image_name,
                     ')/Nodes(IMG_DATA)/Nodes(R20m)/Nodes(',
                     image_name.split('_')[1], '_',
                     product_name.split('_')[2], '_B', band_number,
                     '_20m.jp2)/$value']
                )
            output_file = '%s/%s_B%s' % (
                output_path, image_name[4:], band_number)
        elif band_number in ['01', '09']:
            if copernicus is False:
                band_url = ''.join(
                    [top_url, '/L2/tiles/', product_name[39:41], '/',
                     product_name[41], '/', product_name[42:44], '/',
                     product_name, '.SAFE', '/GRANULE/', image_name,
                     '/IMG_DATA/R60m/', image_name.split('_')[1], '_',
                     product_name.split('_')[2], '_B', band_number, '_60m.jp2']
                )
            else:
                band_url = ''.join(
                    [top_url, '/Products(', uid, ')/Nodes(', product_name,
                     '.SAFE)/Nodes(GRANULE)/Nodes(', image_name,
                     ')/Nodes(IMG_DATA)/Nodes(R60m)/Nodes(',
                     image_name.split('_')[1], '_',
                     product_name.split('_')[2], '_B', band_number,
                     '_60m.jp2)/$value']
                )
            output_file = '%s/%s_B%s' % (
                output_path, image_name[4:], band_number)
    else:
        # old product format
        band_url = ''.join(
            [top_url, '/tiles/', product_name[39:41], '/', product_name[41],
             '/', product_name[42:44], '/',
             product_name, '.SAFE', '/GRANULE/', image_name, '/IMG_DATA/',
             image_name.split('_')[1], '_',
             product_name.split('_')[2], '_B', band_number, '.jp2']
        )
        output_file = '%s/%s_B%s' % (
            output_path, image_name[0:-7], band_number)
    if exporter:
        output_list.append(band_url)
    else:
        if virtual_download:
            output_file += '.vrt'
            _download_virtual_image(
                url=band_url, output_path=output_file,
                extent_list=extent_coordinate_list
            )
        elif extent_coordinate_list is not None:
            vrt_file = cfg.temp.temporary_file_path(name_suffix='.vrt')
            _download_virtual_image(
                url=band_url, output_path=vrt_file,
                extent_list=extent_coordinate_list
            )
            output_file += '.tif'
            cfg.multiprocess.gdal_copy_raster(
                vrt_file, output_file, 'GTiff', cfg.raster_compression, 'LZW'
            )
        else:
            output_file += '.jp2'
            cfg.multiprocess.multi_download_file(
                url_list=[band_url], output_path_list=[output_file],
                proxy_host=proxy_host,
                proxy_port=proxy_port, proxy_user=proxy_user,
                proxy_password=proxy_password, timeout=2,
                progress=progress, message='downloading band %s' % band_number,
                min_progress=min_progress, max_progress=max_progress,
                copernicus=copernicus, access_token=access_token
            )
        if files_directories.is_file(output_file):
            output_list.append(output_file)
            cfg.logger.log.debug('downloaded file %s' % output_file)
        else:
            cfg.messages.error(
                'failed download %s_B%s' % (image_name[0:-7], band_number)
            )
            cfg.logger.log.error(
                'error: download failed %s_B%s' % (
                    image_name[0:-7], band_number)
            )


def _download_virtual_image(url, output_path, extent_list=None):
    """Downloads virtual image."""
    cfg.logger.log.debug('url: %s' % str(url))
    try:
        # noinspection SpellCheckingInspection
        raster_vector.create_virtual_raster(
            input_raster_list=['/vsicurl/%s' % url], output=output_path,
            box_coordinate_list=extent_list
        )
        return True
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False


def query_nasa_cmr(
        product, date_from, date_to, max_cloud_cover=100, result_number=50,
        name_filter=None, coordinate_list=None, progress_message=True,
        proxy_host=None, proxy_port=None, proxy_user=None, proxy_password=None
) -> OutputManager:
    """Perform the query of NASA CMR.

    This tool performs the query of NASA CMR Search
    https://cmr.earthdata.nasa.gov/search/site/search_api_docs.html.

    Args:
        product:
        date_from: date defining the starting period of the query
        date_to:
        max_cloud_cover:
        result_number:
        name_filter:
        coordinate_list:
        progress_message:
        proxy_host:
        proxy_port:
        proxy_user:
        proxy_password:

    Returns:
        object OutputManger

    """
    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(process='search', message='starting', start=True)
    if product == cfg.landsat_hls:
        collection = cfg.landsat_hls_collection
    elif product == cfg.sentinel2_hls:
        collection = cfg.sentinel2_hls_collection
    else:
        collection = cfg.landsat_hls_collection
    image_find_list = []
    # filter the results based on a string
    if name_filter:
        name_filter_split = name_filter.replace(' ', '').split(',')
        for f in name_filter_split:
            image_find_list.append(f)
    else:
        image_find_list.append('h')
    # coordinate list left, top, right, bottom
    if coordinate_list is not None:
        if abs(coordinate_list[0] - coordinate_list[2]) > 10 or abs(
                coordinate_list[1] - coordinate_list[3]
        ) > 10:
            cfg.logger.log.warning('search area extent beyond limits')
            cfg.messages.warning('search area extent beyond limits')
    # loop for results
    max_result_number = result_number
    if max_result_number > 2000:
        max_result_number = 2000
    page = 0
    for _results in range(0, result_number, max_result_number):
        page += 1
        if coordinate_list is None:
            cfg.logger.log.error('search area required')
            cfg.messages.error('search area required')
            return OutputManager(check=False)
        else:
            # ignoring cloud cover because of issue returning 0 results
            url = ''.join(
                ['https://cmr.earthdata.nasa.gov/search/granules.echo10',
                 '?polygon[]=', str(coordinate_list[0]), ',',
                 str(coordinate_list[1]), ',', str(coordinate_list[0]), ',',
                 str(coordinate_list[3]), ',', str(coordinate_list[2]), ',',
                 str(coordinate_list[3]), ',', str(coordinate_list[2]), ',',
                 str(coordinate_list[1]), ',', str(coordinate_list[0]), ',',
                 str(coordinate_list[1]), '&echo_collection_id=',
                 collection, '&temporal=', str(date_from), '%2C', str(date_to),
                 'T23%3A59%3A59.000Z&sort_key%5B%5D=-start_date&page_size=',
                 str(max_result_number), '&page_num=', str(page),
                 '&pretty=true']
            )
        # search
        response = download_tools.open_general_url(
            url=url, proxy_host=proxy_host, proxy_port=proxy_port,
            proxy_user=proxy_user, proxy_password=proxy_password
        )
        if response:
            xml_file = response.read()
            doc = minidom.parseString(xml_file)
            entries = doc.getElementsByTagName('Granule')
            product_table_list = []
            e = 0
            for entry in entries:
                e += 1
                step = int(80 * e / len(entries) + 20)
                percentage = int(100 * e / len(entries))
                cfg.progress.update(
                    message='search in progress', step=step,
                    percentage=percentage, ping=True
                )
                granule = entry.getElementsByTagName('GranuleUR')[0]
                product_name = granule.firstChild.data
                producer_id = entry.getElementsByTagName('ProducerGranuleId')[
                    0]
                producer_image_id = producer_id.firstChild.data
                on = entry.getElementsByTagName('ProviderBrowseUrl')
                url = on[0].getElementsByTagName('URL')[0]
                preview = url.firstChild.data
                dt = entry.getElementsByTagName('BeginningDateTime')[0]
                img_date = dt.firstChild.data
                img_acquisition_date = datetime.datetime.strptime(
                    img_date[0:19], '%Y-%m-%dT%H:%M:%S'
                ).strftime('%Y-%m-%d')
                add_attrs = entry.getElementsByTagName('AdditionalAttribute')
                cloud_cover_percentage = None
                path = None
                for add_attr in add_attrs:
                    add_attr_names = add_attr.getElementsByTagName('Name')
                    for add_attr_name in add_attr_names:
                        add_attr_name_c = add_attr_name.firstChild.data
                        if add_attr_name_c == 'CLOUD_COVERAGE':
                            add_attr_values = (
                                add_attr.getElementsByTagName('Values')[0]
                            )
                            add_attr_val = (
                                add_attr_values.getElementsByTagName('Value')[
                                    0]
                            )
                            cloud_cover_percentage = (
                                add_attr_val.firstChild.data
                            )
                            if path:
                                break
                        elif add_attr_name_c == 'MGRS_TILE_ID':
                            add_attr_values = (
                                add_attr.getElementsByTagName('Values')[0]
                            )
                            add_attr_val = (
                                add_attr_values.getElementsByTagName('Value')[
                                    0]
                            )
                            path = add_attr_val.firstChild.data
                            if cloud_cover_percentage:
                                break
                    if cloud_cover_percentage and path:
                        break
                if float(cloud_cover_percentage) < max_cloud_cover:
                    point_latitude = entry.getElementsByTagName(
                        'PointLatitude'
                    )
                    point_longitude = entry.getElementsByTagName(
                        'PointLongitude'
                    )
                    lat = []
                    for latitude in point_latitude:
                        lat.append(float(latitude.firstChild.data))
                    lon = []
                    for longitude in point_longitude:
                        lon.append(float(longitude.firstChild.data))
                    for f in image_find_list:
                        if f.lower() in product_name.lower():
                            product_table_list.append(
                                tm.create_product_table(
                                    product=product,
                                    product_id=producer_image_id,
                                    acquisition_date=img_acquisition_date,
                                    cloud_cover=float(cloud_cover_percentage),
                                    zone_path=path, row=None,
                                    min_lat=float(min(lat)),
                                    min_lon=float(min(lon)),
                                    max_lat=float(max(lat)),
                                    max_lon=float(max(lon)),
                                    collection=collection, size=None,
                                    preview=preview, uid=producer_image_id,
                                    image=product_name
                                )
                            )
            cfg.progress.update(end=True)
            cfg.logger.log.info('end')
            return OutputManager(
                extra={
                    'product_table': tm.stack_product_table(
                        product_list=product_table_list
                    )
                }
            )
        else:
            cfg.logger.log.error('error: search failed')
            cfg.messages.error('error: search failed')
            return OutputManager(check=False)


def export_product_table_as_xml(product_table, output_path=None):
    """Exports product table as xml.

    Exports a product table and attributes.
    """  # noqa: E501
    cfg.logger.log.debug('export product table: %s' % str(output_path))
    root = cElementTree.Element('product_table')
    root.set('version', str(cfg.version))
    total_products = product_table.shape[0]
    for i in range(total_products):
        if cfg.action is False:
            break
        product_element = cElementTree.SubElement(root, 'product')
        product_element.set('uid', str(product_table['uid'][i]))
        for attribute in product_table.dtype.names:
            if attribute != 'uid':
                element = cElementTree.SubElement(
                    product_element,
                    attribute
                )
                element.text = str(product_table[attribute][i])
    if output_path is None:
        return cElementTree.tostring(root)
    else:
        # save to file
        pretty_xml = minidom.parseString(
            cElementTree.tostring(root)
        ).toprettyxml()
        read_write_files.write_file(pretty_xml, output_path)
        return output_path


def import_as_xml(xml_path):
    """Imports a product table as xml.

    Imports a product table and attributes.
    """  # noqa: E501
    cfg.logger.log.debug('import product table: %s' % xml_path)
    tree = cElementTree.parse(xml_path)
    root = tree.getroot()
    version = root.get('version')
    if version is None:
        cfg.logger.log.error('failed importing product table: %s' % xml_path)
        cfg.messages.error('failed importing product table: %s' % xml_path)
        return OutputManager(check=False)
    else:
        product_table_list = []
        for child in root:
            if cfg.action is False:
                break
            uid = child.get('uid')
            attributes = {}
            for attribute in cfg.product_dtype_list:
                if attribute[0] != 'uid':
                    element = child.find(attribute[0]).text
                    if element == 'None':
                        element = None
                    attributes[attribute[0]] = element
            product_table_list.append(
                tm.create_product_table(
                    product=attributes['product'],
                    product_id=attributes['product_id'],
                    acquisition_date=attributes['acquisition_date'],
                    cloud_cover=float(attributes['cloud_cover']),
                    zone_path=attributes['zone_path'],
                    row=attributes['row'],
                    min_lat=float(attributes['min_lat']),
                    min_lon=float(attributes['min_lon']),
                    max_lat=float(attributes['max_lat']),
                    max_lon=float(attributes['max_lon']),
                    collection=attributes['collection'],
                    size=attributes['size'],
                    preview=attributes['preview'], uid=uid,
                    image=attributes['image']
                )
            )
        return OutputManager(
            extra={
                'product_table': tm.stack_product_table(
                    product_list=product_table_list
                )
            }
        )
