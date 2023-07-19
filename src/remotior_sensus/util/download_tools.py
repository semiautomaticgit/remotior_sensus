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

import datetime
import os
import time
import urllib.request
from http.cookiejar import CookieJar

from remotior_sensus.core import configurations as cfg
from remotior_sensus.util import read_write_files, files_directories


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
        retried=False, timeout=20
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
        # small files
        if block_size >= file_size:
            response = url_request.read()
            read_write_files.write_file(response, output_path, mode='wb')
        else:
            files_directories.create_parent_directory(output_path)
            with open(output_path, 'wb') as file:
                while True:
                    if cfg.action is True:
                        if adaptive_block_size:
                            start_time = datetime.datetime.now()
                        block_read = url_request.read(block_size)
                        if not block_read:
                            break
                        # adapt block size
                        if adaptive_block_size:
                            end_time = datetime.datetime.now()
                            time_delta = end_time - start_time
                            new_speed = block_size / time_delta.microseconds
                            if new_speed >= speed:
                                block_size = block_size + 1024 * 1024
                                start_time = end_time
                                speed = new_speed
                        if progress:
                            if message is None:
                                message = '({}/{} MB) {}'.format(
                                    downloaded_part_size, total_size, url
                                )
                            downloaded_part_size = round(
                                int(os.stat(output_path).st_size) / 1048576, 2
                            )
                            step = int(
                                (max_progress - min_progress)
                                * downloaded_part_size / total_size
                                + min_progress
                            )
                            percentage = int(
                                100 * downloaded_part_size / total_size
                            )
                            cfg.progress.update(
                                message=message, step=step,
                                percentage=percentage, ping=True
                            )
                        # write file
                        file.write(block_read)
                    else:
                        cfg.logger.log.error('cancel url: %s' % url)
                        cfg.messages.error('cancel url: %s' % url)
                        return False, 'cancel'
        return True, output_path
    except Exception as err:
        if retried is False and '403' not in str(err):
            cfg.logger.log.debug('retry url: %s' % url)
            time.sleep(2)
            download_file(
                url=url, output_path=output_path,
                authentication_uri=authentication_uri, user=user,
                password=password,
                proxy_host=proxy_host, proxy_port=proxy_port,
                proxy_user=proxy_user,
                proxy_password=proxy_password,
                progress=progress, message=message, min_progress=min_progress,
                max_progress=max_progress, retried=True, timeout=timeout*2
            )
        else:
            cfg.logger.log.error('%s; url: %s' % (err, url))
            cfg.messages.error('%s; url: %s' % (err, url))
            return False, str(err)
