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
Tools to manage files and directories
"""

import os
import shutil
import zipfile
from pathlib import Path

from remotior_sensus.core import configurations as cfg


# check if directory exists
def is_directory(path):
    return os.path.isdir(path)


# get parent directory of input_raster
def parent_directory(path):
    try:
        directory = os.path.dirname(path)
        cfg.logger.log.debug('directory: %s' % directory)
        return directory
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False


# remove directory
def remove_directory(directory):
    try:
        shutil.rmtree(directory)
    except Exception as err:
        cfg.logger.log.error(str(err))
    cfg.logger.log.debug('directory: %s' % directory)
    return directory


# convert to absolute path
def relative_to_absolute_path(path, root=None):
    if root is None:
        a_path = path
    else:
        a_path = os.path.join(root, path).replace('\\', '/').replace('//', '/')
    a = Path(a_path)
    original_path = Path(path)
    if a.is_dir() or a.is_file():
        absolute = a_path
    # if absolute path is not file or directory get relative path
    elif original_path.is_dir() or original_path.is_file():
        absolute = path
    else:
        absolute = a_path
        cfg.logger.log.warning('file not found: %s' % path)
    cfg.logger.log.debug(
        'path:{}; root:{}; absolute:{}'.format(path, root, absolute)
    )
    return absolute


# create parent directory of a file path
def create_parent_directory(file_path):
    try:
        path = os.path.dirname(file_path)
        if is_directory(path):
            return path
        else:
            os.makedirs(path)
            cfg.logger.log.debug('path: %s' % path)
            return path
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False


# create directory of a path
def create_directory(path):
    try:
        if not is_directory(path):
            os.makedirs(path)
            cfg.logger.log.debug('path: %s' % path)
            return path
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False


# file list in directory
def files_in_directory(
        path, subdirectories=False, path_filter=None, suffix_filter=None,
        sort_files=False, root_directory=None
):
    cfg.logger.log.debug('start')
    dir_f = relative_to_absolute_path(path, root_directory)
    p = Path(dir_f)
    if subdirectories:
        pattern = '**/*'
    else:
        pattern = '*'
    if suffix_filter is not None:
        pattern = '%s%s' % (pattern, suffix_filter)
    p_list = list(p.glob(pattern))
    path_list = []
    for i in p_list:
        if i.is_file():
            if root_directory is None:
                f = i.as_posix()
            else:
                f = absolute_to_relative_path(i.as_posix(), root_directory)
            if path_filter is None:
                path_list.append(f)
            else:
                if path_filter in i.as_posix():
                    path_list.append(f)
    if sort_files:
        path_list = sorted(path_list)
    cfg.logger.log.debug('path_list: %s' % str(path_list))
    return path_list


# file extension
def file_extension(path, lower=True):
    try:
        p = Path(path)
        suffix = p.suffix
        if lower:
            suffix = suffix.lower()
        cfg.logger.log.debug('path %s; suffix: %s' % (path, suffix))
        return suffix
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False


# file name
def file_name(path, suffix=False):
    try:
        p = Path(path)
        # with suffix
        if suffix:
            name = p.name
        # without suffix
        else:
            name = p.stem
        cfg.logger.log.debug('name: %s' % name)
        return name
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False


# convert to relative path
def absolute_to_relative_path(path, root=None):
    p = Path(path)
    try:
        relative = p.relative_to(root)
    except Exception as err:
        cfg.logger.log.error(str(err))
        relative = p
    cfg.logger.log.debug('relative: %s' % relative.as_posix())
    return relative.as_posix()


# check if file exists
def is_file(path):
    return os.path.isfile(path)


# check output path
def output_path(path, extension):
    if not (file_extension(path) == extension):
        path = '%s%s' % (path, extension)
    return path


# check raster output path
def raster_output_path(path, virtual_output=False, overwrite=False):
    if path is None:
        path = cfg.temp.temporary_raster_path(extension=cfg.vrt_suffix)
    elif is_file(path) and not overwrite:
        raise Exception('existing path %s' % path)
    try:
        # vrt
        if virtual_output or file_extension(path) == cfg.vrt_suffix:
            virtual = True
            o_path = os.path.join(
                parent_directory(path), '{}{}'.format(
                    file_name(path, False), cfg.vrt_suffix
                ).replace('\\', '/')
            )
        # tif
        elif file_extension(path) == cfg.tif_suffix:
            virtual = False
            o_path = os.path.join(
                parent_directory(path), '{}{}'.format(
                    file_name(path, False), cfg.tif_suffix
                ).replace('\\', '/')
            )
        # other formats
        else:
            virtual = False
            o_path = os.path.join(
                parent_directory(path), '{}{}'.format(
                    file_name(path, False), cfg.tif_suffix
                ).replace('\\', '/')
            )
        cfg.logger.log.debug('o_path: %s:; virtual: %s' % (o_path, virtual))
        create_parent_directory(o_path)
        return o_path, virtual
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False, False


# check input_path raster path
def input_path(path):
    if not is_file(path):
        raise Exception('file not found: ' % path)
    return path


# move file
def move_file(in_path, out_path):
    create_parent_directory(out_path)
    try:
        shutil.move(in_path, out_path)
    except Exception as err:
        cfg.logger.log.error(str(err))
    cfg.logger.log.debug('out_path: %s' % out_path)
    return out_path


# copy file
def copy_file(in_path, out_path):
    create_parent_directory(out_path)
    try:
        shutil.copy(in_path, out_path)
    except Exception as err:
        cfg.logger.log.error(str(err))
    cfg.logger.log.debug('out_path: %s' % out_path)
    return out_path


# zip file list
def zip_files(file_list, out_path, compression=None, compress_level=None):
    cfg.logger.log.debug('out_path: %s' % out_path)
    if compression is None:
        compression = zipfile.ZIP_DEFLATED
    if compress_level is None:
        compress_level = 2
    try:
        with zipfile.ZipFile(
                out_path, 'w', compression=compression,
                compresslevel=compress_level
        ) as file_zip:
            for f in file_list:
                file_zip.write(f, file_name(path=f, suffix=True))
    except Exception as err:
        cfg.logger.log.error(str(err))
    return out_path


# unzip file
def unzip_file(in_path, out_dir_path):
    file_list = []
    try:
        with zipfile.ZipFile(in_path) as file_zip:
            for name in file_zip.namelist():
                file = file_zip.open(name)
                name_f = file_name(name, suffix=True)
                output_file = open('%s/%s' % (out_dir_path, name_f), 'wb')
                file_list.append('%s/%s' % (out_dir_path, name_f))
                with file, output_file:
                    shutil.copyfileobj(file, output_file)
    except Exception as err:
        cfg.logger.log.error(str(err))
    return file_list
