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
Tools to download files
"""

import urllib.request
import requests
from datetime import datetime
from http.cookiejar import CookieJar
from os import stat
from time import sleep

from remotior_sensus.core import configurations as cfg
from remotior_sensus.util.files_directories import create_parent_directory
from remotior_sensus.util.read_write_files import write_file


# get proxy handler
def get_proxy_handler(
        proxy_host, proxy_port, proxy_user=None, proxy_password=None
):
    if len(proxy_user) > 0:
        proxy_handler = urllib.request.ProxyHandler(
            {
                'http': 'http://{}:{}@{}:{}'.format(
                    proxy_user, proxy_password, proxy_host, proxy_port
                )
            }
        )
    else:
        proxy_handler = urllib.request.ProxyHandler(
            {'http': 'http://{}:{}'.format(proxy_host, proxy_port)}
        )
    return proxy_handler


# opener
def general_opener(
        proxy_host=None, proxy_port=None, proxy_user=None, proxy_password=None
):
    cookie_jar = CookieJar()
    if proxy_host:
        proxy_handler = get_proxy_handler(
            proxy_host, proxy_port, proxy_user,
            proxy_password
        )
        opener = urllib.request.build_opener(
            proxy_handler, urllib.request.HTTPCookieProcessor(cookie_jar)
        )
    else:
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cookie_jar)
        )
    return opener, cookie_jar


# open general url
def open_general_url(
        url, proxy_host=None, proxy_port=None, proxy_user=None,
        proxy_password=None
):
    opener, cookie_jar = general_opener(
        proxy_host=proxy_host, proxy_port=proxy_port, proxy_user=proxy_user,
        proxy_password=proxy_password
    )
    request = urllib.request.Request(url)
    try:
        response = opener.open(request)
    except Exception as err:
        response = None
        cfg.logger.log.error('{}; url:{}'.format(err, url))
        cfg.messages.error('{}; url:{}'.format(err, url))
    return response


# download file
def download_file(
        url, output_path, authentication_uri=None, user=None, password=None,
        proxy_host=None, proxy_port=None, proxy_user=None, proxy_password=None,
        progress=True, message=None, min_progress=0, max_progress=100,
        retried=False, timeout=20, callback=None, log=None
):
    cfg.logger.log.debug('url: %s' % url)
    if authentication_uri is None:
        auth_handler = None
    else:
        pass_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        pass_mgr.add_password(None, authentication_uri, user, password)
        auth_handler = urllib.request.HTTPBasicAuthHandler(pass_mgr)
    cookie_proc = urllib.request.HTTPCookieProcessor()
    # with proxy
    if proxy_host:
        proxy_handler = get_proxy_handler(
            proxy_host, proxy_port, proxy_user, proxy_password
        )
        if auth_handler:
            opener = urllib.request.build_opener(
                proxy_handler, cookie_proc, auth_handler
            )
        else:
            opener = urllib.request.build_opener(proxy_handler, cookie_proc)
    # without proxy
    elif auth_handler:
        opener = urllib.request.build_opener(cookie_proc, auth_handler)
    else:
        opener = urllib.request.build_opener(cookie_proc)
    urllib.request.install_opener(opener)
    try:
        url_request = urllib.request.urlopen(url, timeout=timeout)
        if log is not None:
            log.debug('url_request: %s' % str(url_request))
        try:
            file_size = int(url_request.headers['Content-Length'])
            # size megabyte
            total_size = round(file_size / 1048576, 2)
        except Exception as err:
            str(err)
            file_size = 1
            total_size = 1
        block_size = 1024 * 1024
        # set initial speed for adaptive block size
        speed = 0
        adaptive_block_size = True
        if log is not None:
            log.debug(
                'block_size: %s; file_size: %s'
                % (str(block_size), str(file_size))
                )
        # small files
        if block_size >= file_size:
            response = url_request.read()
            write_file(response, output_path, mode='wb')
        else:
            create_parent_directory(output_path)
            with open(output_path, 'wb') as file:
                while True:
                    if cfg.action is True:
                        if adaptive_block_size:
                            start_time = datetime.now()
                        block_read = url_request.read(block_size)
                        if not block_read:
                            break
                        # adapt block size
                        if adaptive_block_size:
                            try:
                                end_time = datetime.now()
                                time_delta = end_time - start_time
                                new_speed = (
                                        block_size / time_delta.microseconds
                                )
                                if new_speed >= speed:
                                    block_size = block_size + 1024 * 1024
                                    start_time = end_time
                                    speed = new_speed
                            except Exception as err:
                                str(err)
                        if progress is True:
                            try:
                                downloaded_part_size = round(
                                    int(stat(output_path).st_size) / 1048576, 2
                                )
                                if message is None:
                                    message = '({}/{} MB) {}'.format(
                                        downloaded_part_size, total_size, url
                                    )
                                step = int(
                                    (max_progress - min_progress)
                                    * downloaded_part_size / total_size
                                    + min_progress
                                )
                                percentage = int(
                                    100 * downloaded_part_size / total_size
                                )
                                if callback is None:
                                    cfg.progress.update(
                                        message=message, step=step,
                                        percentage=percentage, ping=True
                                    )
                                else:
                                    callback(percentage, False)
                            except Exception as err:
                                str(err)
                        # write file
                        file.write(block_read)
                    else:
                        return False, 'cancel'
        return True, output_path
    except Exception as err:
        if retried is False and '403' not in str(err):
            sleep(2)
            download_file(
                url=url, output_path=output_path,
                authentication_uri=authentication_uri, user=user,
                password=password,
                proxy_host=proxy_host, proxy_port=proxy_port,
                proxy_user=proxy_user,
                proxy_password=proxy_password,
                progress=progress, message=message, min_progress=min_progress,
                max_progress=max_progress, retried=True, timeout=timeout * 2
            )
        else:
            return False, str(err)


# download Copernicus file
def download_copernicus_file(
        url, output_path, authentication_uri=None,
        proxy_host=None, proxy_port=None, proxy_user=None, proxy_password=None,
        progress=True, message=None, min_progress=0, max_progress=100,
        timeout=20, callback=None, log=None, retried=None, access_token=None
):
    cfg.logger.log.debug('url: %s' % url)
    if authentication_uri is None:
        pass
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
    url_r = None
    try:
        url_request = session.get(url, allow_redirects=False)
        while url_request.status_code in (301, 302, 303, 307):
            if cfg.action is True:
                url_r = url_request.headers['Location']
                url_request = session.get(
                    url_r, allow_redirects=False, timeout=timeout
                )
            else:
                return False, 'cancel'
        if log is not None:
            log.debug('url_request: %s' % str(url_r))
        file_response = session.get(
            url_r, verify=False, allow_redirects=True, timeout=timeout,
            stream=True
        )
        try:
            file_size = int(file_response.headers['Content-Length'])
            # size megabyte
            total_size = round(file_size / 1048576, 2)
            if log is not None:
                log.debug('total_size: %s' % str(total_size))
        except Exception as err:
            return False, str(err)
        block_size = int(file_size / 10)
        if log is not None:
            log.debug(
                'block_size: %s; file_size: %s'
                % (str(block_size), str(file_size))
                )
        # small files
        if file_size < 300000:
            for chunk in file_response.iter_content(chunk_size=file_size):
                if chunk:
                    write_file(chunk, output_path, mode='wb')
        else:
            create_parent_directory(output_path)
            with open(output_path, 'wb') as file:
                while True:
                    if cfg.action is True:
                        for chunk in file_response.iter_content(
                                chunk_size=block_size
                        ):
                            if chunk:
                                if progress is True:
                                    try:
                                        downloaded_part_size = round(
                                            int(stat(output_path).st_size)
                                            / 1048576, 2
                                        )
                                        if message is None:
                                            message = '({}/{} MB) {}'.format(
                                                downloaded_part_size,
                                                total_size, url
                                            )
                                        step = int(
                                            (max_progress - min_progress)
                                            * downloaded_part_size / total_size
                                            + min_progress
                                        )
                                        percentage = int(
                                            100 * downloaded_part_size
                                            / total_size
                                        )
                                        if callback is None:
                                            cfg.progress.update(
                                                message=message, step=step,
                                                percentage=percentage,
                                                ping=True
                                            )
                                        else:
                                            callback(percentage, False)
                                    except Exception as err:
                                        str(err)
                                # write file
                                file.write(chunk)
                            else:
                                break
                    else:
                        return False, 'cancel'
        return True, output_path
    except Exception as err:
        if retried:
            pass
        return False, str(err)
