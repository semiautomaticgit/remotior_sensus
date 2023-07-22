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
"""Session manager.

Core class that manages the Remotior Sensus' session. It defines fundamental
parameters such as available RAM and number of parallel processes.
Creates a temporary directory to store temporary files.
Exposes core functions and tools.
It includes a method to close the session removing temporary files
and stopping parallel processes.

Typical usage example:

    >>> # start the session
    >>> rs = Session()
    >>> # optionally set session parameters
    >>> rs.set(n_processes=2)
    >>> # close the session when processing is done
    >>> rs.close()
"""

import logging
from types import FunctionType
from typing import Optional

from remotior_sensus.core import configurations, messages, table_manager
from remotior_sensus.core.bandset_catalog import BandSet, BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.core.log import Log
from remotior_sensus.core.multiprocess_manager import Multiprocess
from remotior_sensus.core.progress import Progress
from remotior_sensus.core.spectral_signatures import SpectralSignaturesCatalog
from remotior_sensus.core.temporary import Temporary
from remotior_sensus.tools import (
    band_calc, band_classification, band_clip, band_combination, band_dilation,
    band_erosion, band_neighbor_pixels, band_pca, band_sieve, band_resample,
    band_stack, band_mask, raster_split,
    cross_classification, download_products, mosaic, preprocess_products,
    raster_reclassification, raster_report, raster_to_vector, vector_to_raster
)
from remotior_sensus.util import (
    dates_times, system_tools, files_directories, download_tools)


class Session(object):
    """Manages system parameters.

    This module allows for managing Remotior Sensus' session,
    setting fundamental processing parameters and exposing core functions 
    and tools.

    Attributes:
        configurations: module containing shared variables and functions
        bandset: access :func:`~remotior_sensus.core.bandset_catalog.BandSet` class
        bandset_catalog: access :func:`~remotior_sensus.core.bandset_catalog.BandSetCatalog` class
        spectral_signatures_catalog: access :func:`~remotior_sensus.core.spectral_signatures.SpectralSignaturesCatalog` class 
        output_manager: access :func:`~remotior_sensus.core.output_manager.OutputManager` class 
        table_manager: access functions of :func:`~remotior_sensus.core.table_manager` module
        dates_times: access dates and times utilities
        download_tools: access download utilities
        band_calc: tool :func:`~remotior_sensus.tools.band_calc`
        band_classification: tool :func:`~remotior_sensus.tools.band_classification`
        classifier: tool :func:`~remotior_sensus.tools.band_classification.Classifier`
        band_combination: tool :func:`~remotior_sensus.tools.band_combination`
        band_dilation: tool :func:`~remotior_sensus.tools.band_dilation`
        band_erosion: tool :func:`~remotior_sensus.tools.band_erosion`
        band_mask: tool :func:`~remotior_sensus.tools.band_mask`
        band_neighbor_pixels: tool :func:`~remotior_sensus.tools.band_neighbor_pixels`
        band_pca: tool :func:`~remotior_sensus.tools.band_pca`
        band_resample: tool :func:`~remotior_sensus.tools.band_resample`
        band_sieve: tool :func:`~remotior_sensus.tools.band_sieve`
        band_stack: tool :func:`~remotior_sensus.tools.band_stack`
        cross_classification: tool :func:`~remotior_sensus.tools.cross_classification`
        download_products: tool :func:`~remotior_sensus.tools.download_products`
        mosaic: tool :func:`~remotior_sensus.tools.mosaic`
        preprocess_products: tool :func:`~remotior_sensus.tools.preprocess_products`
        raster_reclassification: tool :func:`~remotior_sensus.tools.preprocess_products`
        raster_report: tool :func:`~remotior_sensus.tools.raster_report`
        raster_split: tool :func:`~remotior_sensus.tools.raster_split`
        raster_to_vector: tool :func:`~remotior_sensus.tools.raster_to_vector`
        vector_to_raster: tool :func:`~remotior_sensus.tools.vector_to_raster`

    Examples:
        Start a session
            >>> import remotior_sensus
            >>> rs = remotior_sensus.Session()

        Start a session defining number of parallel processes. and available RAM
            >>> import remotior_sensus
            >>> rs = remotior_sensus.Session(n_processes=4,available_ram=4096)

        Create a :func:`~remotior_sensus.core.bandset_catalog.BandSetCatalog`
            >>> catalog = rs.bandset_catalog()

        Run the tool for raster report
            >>> output = rs.raster_report(raster_path='file.tif', output_path='output.txt')

        Stop a session at the end to clear temporary directory
            >>> rs.close()
    """  # noqa: E501

    def __init__(
            self, n_processes: Optional[int] = 2, available_ram: int = 2048,
            temporary_directory: str = None,
            directory_prefix: str = None, log_level: int = 20,
            log_time: bool = True, progress_callback=None,
            multiprocess_module=None, messages_callback=None,
            smtp_server=None, smtp_user=None,  smtp_password=None,
            smtp_recipients=None, smtp_notification=None,
            sound_notification=None
    ):
        """Starts a session.

        Starts a new session setting fundamental parameters for processing.
        It sets the number of parallel processes (default 2) and available RAM
        (default 2048MB) to be used  in calculations.
        It starts the class Temporary to manage temporary files by creating a
        temporary directory with an optional name prefix.
        It starts the class Log for logging (with a default level INFO) and
        creates a logging formatter with the option to hide time.
        It starts the class Progress for displaying progress with a default
        callback function.
        A custom progress callback function can be passed optionally.

        The sessions also allows for accessing to the core functions
        and tools.

        In the end, the close() function should be called to clear
        the temporary directory and stop the parallel processes.

        Args:
            n_processes: number of parallel processes.
            available_ram: number of megabytes of RAM available to processes.
            temporary_directory: path to a temporary directory.
            directory_prefix: prefix of the name of the temporary directory.
            log_level: level of logging (10 for DEBUG, 20 for INFO).
            log_time: if True, logging includes the time.
            progress_callback: function for progress callback.
            multiprocess_module: multiprocess module, useful if Remotior Sensus' session is started from another Python module.
            messages_callback: message module, useful if Remotior Sensus' session is started from another Python module.
            smtp_server: optional server for SMTP notification.
            smtp_user: user for SMTP authentication.
            smtp_password: password for SMTP authentication.
            smtp_recipients: string of one or more email addresses separated by comma for SMTP notification.
            smtp_notification: optional, if True send SMTP notification.
            sound_notification: optional, if True play sound notification.

        Examples:
            Start a session
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
        """  # noqa: E501
        configurations.n_processes = n_processes
        configurations.available_ram = available_ram
        if sound_notification is not None:
            configurations.sound_notification = sound_notification
        if smtp_notification is not None:
            configurations.smtp_notification = smtp_notification
        if smtp_server is not None:
            configurations.smtp_server = smtp_server
            configurations.smtp_user = smtp_user
            configurations.smtp_password = smtp_password
            configurations.smtp_recipients = smtp_recipients
        # create temporary directory
        temp = Temporary()
        if directory_prefix is None:
            directory_prefix = configurations.root_name
        configurations.temp = temp.create_root_temporary_directory(
            prefix=directory_prefix, directory=temporary_directory
        )
        # create logger
        if log_level is None:
            log_level = logging.INFO
        self.log_level = log_level
        configurations.logger = Log(
            directory=configurations.temp.dir, level=self.log_level,
            time=log_time
        )
        # start progress
        if progress_callback is None:
            progress_callback = Progress.print_progress_replace
        configurations.progress = Progress(callback=progress_callback)
        if messages_callback is None:
            configurations.messages = messages
        else:
            configurations.messages = messages_callback
        system_tools.get_system_info()
        check = _check_dependencies(configurations)
        if check:
            self.configurations = configurations
            # create multiprocess instance
            self.configurations.multiprocess = Multiprocess(
                n_processes, multiprocess_module
                )
            # available core tools
            self.bandset = BandSet
            self.bandset_catalog = BandSetCatalog
            self.spectral_signatures_catalog = SpectralSignaturesCatalog
            self.output_manager = OutputManager
            self.table_manager = table_manager
            # available tools
            self.band_calc = band_calc.band_calc
            self.configurations.band_calc = band_calc.band_calc
            self.band_classification = band_classification.band_classification
            self.configurations.band_classification = \
                band_classification.band_classification
            self.classifier = band_classification.Classifier
            self.band_combination = band_combination.band_combination
            self.configurations.band_combination = \
                band_combination.band_combination
            self.band_dilation = band_dilation.band_dilation
            self.configurations.band_dilation = band_dilation.band_dilation
            self.band_erosion = band_erosion.band_erosion
            self.configurations.band_erosion = band_erosion.band_erosion
            self.band_mask = band_mask.band_mask
            self.configurations.band_mask = band_mask.band_mask
            self.mosaic = mosaic.mosaic
            self.band_neighbor_pixels = \
                band_neighbor_pixels.band_neighbor_pixels
            self.configurations.band_neighbor_pixels = \
                band_neighbor_pixels.band_neighbor_pixels
            self.band_pca = band_pca.band_pca
            self.band_clip = band_clip.band_clip
            self.configurations.band_pca = band_pca.band_pca
            self.band_sieve = band_sieve.band_sieve
            self.configurations.band_sieve = band_sieve.band_sieve
            self.band_resample = band_resample.band_resample
            self.configurations.band_resample = band_resample.band_resample
            self.band_stack = band_stack.band_stack
            self.configurations.band_stack = band_stack.band_stack
            self.cross_classification = \
                cross_classification.cross_classification
            self.download_products = download_products
            self.preprocess_products = preprocess_products
            self.raster_reclassification = \
                raster_reclassification.raster_reclassification
            self.raster_report = raster_report.raster_report
            self.raster_split = raster_split.raster_split
            self.raster_to_vector = raster_to_vector.raster_to_vector
            self.vector_to_raster = vector_to_raster.vector_to_raster
            self.dates_times = dates_times
            self.download_tools = download_tools
        else:
            self.configurations = None

    def close(self, log_path: str = None):
        """Closes a Session.

        This function closes current session by deleting the temporary files
        and stopping parallel processes.

        Args:
            log_path: path where the log file is saved

        Examples:
            Given that a session was previously started
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()

            Set the number of parallel processes and available RAM
                >>> rs.set(n_processes=8, available_ram=20480)

            Set the logging level to DEBUG
                >>> rs.set(log_level=10)
        """
        if log_path:
            try:
                files_directories.copy_file(
                    self.configurations.logger.file_path, log_path
                )
            except Exception as err:
                str(err)
        self.configurations.temp.clear()
        self.configurations.multiprocess.stop()

    def set(
            self, n_processes: int = None, available_ram: int = None,
            temporary_directory: str = None,
            directory_prefix: str = None, log_level: int = None,
            log_time: bool = None, progress_callback: FunctionType = None,
            smtp_server=None, smtp_user=None, smtp_password=None,
            smtp_recipients=None, sound_notification=None,
            smtp_notification=None
    ):
        """Sets or changes the parameters of an existing Session.

        Sets the parameters of an existing Session such as number
        of processes or temporary directory.

        Args:
            n_processes: number of parallel processes.
            available_ram: number of megabytes of RAM available to processes.
            temporary_directory: path to a temporary directory.
            directory_prefix: prefix of the name of the temporary directory.
            log_level: level of logging (10 for DEBUG, 20 for INFO).
            log_time: if True, logging includes the time.
            progress_callback: function for progress callback.
            smtp_server: optional server for SMTP notification.
            smtp_user: user for SMTP authentication.
            smtp_password: password for SMTP authentication.
            smtp_recipients: string of one or more email addresses separated by comma for SMTP notification.
            smtp_notification: optional, if True send SMTP notification.
            sound_notification: optional, if True play sound notification.
            
        Examples:
            Given that a session was previously started
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()

            Stop a session
                >>> rs.close()

            Stop a session saving also the log to a file
                >>> rs.close(log_path='file.txt')
        """  # noqa: E501
        if n_processes:
            self.configurations.n_processes = n_processes
            check = _check_dependencies(self.configurations)
            if check:
                self.configurations.multiprocess.stop()
                self.configurations.multiprocess = Multiprocess(n_processes)
            else:
                self.configurations = None
                return
        if available_ram:
            self.configurations.available_ram = available_ram
        if temporary_directory:
            self.configurations.temp.clear()
            # create temporary directory
            temp = Temporary()
            if directory_prefix is None:
                directory_prefix = self.configurations.root_name
            self.configurations.temp = temp.create_root_temporary_directory(
                prefix=directory_prefix, directory=temporary_directory
            )
        if log_level:
            self.log_level = log_level
            if log_time is None:
                log_time = True
            # create logger
            self.configurations.logger = Log(
                directory=self.configurations.temp.dir, level=self.log_level,
                time=log_time
            )
        elif log_time:
            # create logger
            self.configurations.logger = Log(
                directory=self.configurations.temp.dir, level=self.log_level,
                time=log_time
            )
        if sound_notification:
            self.configurations.sound_notification = sound_notification
        if smtp_notification:
            self.configurations.smtp_notification = smtp_notification
        if smtp_server:
            self.configurations.smtp_server = smtp_server
        if smtp_user:
            self.configurations.smtp_user = smtp_user
        if smtp_password:
            self.configurations.smtp_password = smtp_password
        if smtp_recipients:
            self.configurations.smtp_recipients = smtp_recipients
        if progress_callback:
            # start progress
            self.configurations.progress = Progress(callback=progress_callback)
        self.configurations.logger.log.info(
            'n_processes: %s; ram: %s; temp.dir: %s'
            % (self.configurations.n_processes,
               self.configurations.available_ram,
               self.configurations.temp.dir)
        )


def _check_dependencies(configuration_module: configurations) -> bool:
    """Checks the dependencies.

    Checks the dependencies and returns a boolean.

    Args:
        configuration_module: module configurations used for logging.
    """
    check = True
    try:
        import os
        try:
            import numpy
        except Exception as err:
            configuration_module.logger.log.error(str(err))
            configuration_module.messages.error('dependency error: numpy')
            check = False
        try:
            from scipy import signal
        except Exception as err:
            configuration_module.logger.log.error(str(err))
            configuration_module.messages.error('dependency error: scipy')
            check = False
        try:
            import torch
        except Exception as err:
            configuration_module.logger.log.warning(str(err))
            configuration_module.messages.warning('dependency error: pytorch')
        try:
            from sklearn import svm
        except Exception as err:
            configuration_module.logger.log.warning(str(err))
            configuration_module.messages.warning('dependency error: sklearn')
        if configuration_module.gdal_path is not None:
            os.add_dll_directory(configuration_module.gdal_path)
        try:
            from osgeo import gdal
        except Exception as err:
            configuration_module.logger.log.error(str(err))
            configuration_module.messages.error('dependency error: gdal')
            check = False
    except Exception as err:
        configuration_module.logger.log.error(str(err))
        configuration_module.messages.error(str(err))
        check = False
    return check
