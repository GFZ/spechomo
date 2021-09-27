# -*- coding: utf-8 -*-

# spechomo, Spectral homogenization of multispectral satellite data
#
# Copyright (C) 2019-2021
# - Daniel Scheffler (GFZ Potsdam, daniel.scheffler@gfz-potsdam.de)
# - Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences Potsdam,
#   Germany (https://www.gfz-potsdam.de/)
#
# This software was developed within the context of the GeoMultiSens project funded
# by the German Federal Ministry of Education and Research
# (project grant code: 01 IS 14 010 A-C).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Please note the following exception: `spechomo` depends on tqdm, which is
# distributed under the Mozilla Public Licence (MPL) v2.0 except for the files
# "tqdm/_tqdm.py", "setup.py", "README.rst", "MANIFEST.in" and ".gitignore".
# Details can be found here: https://github.com/tqdm/tqdm/blob/master/LICENCE.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SpecHomo logging module containing logging related classes and functions."""

import logging
import os
import warnings
import sys
try:
    # noinspection PyCompatibility
    from StringIO import StringIO  # Python 2
except ImportError:
    from io import StringIO  # Python 3


class SpecHomo_Logger(logging.Logger):
    """Class for the SpecHomo logger."""

    def __init__(self, name_logfile, fmt_suffix=None, path_logfile=None, log_level='INFO', append=True):
        # type: (str, any, str, any, bool) -> None
        """Return a logging.logger instance pointing to the given logfile path.

        :param name_logfile:
        :param fmt_suffix:      if given, it will be included into log formatter
        :param path_logfile:    if no path is given, only a StreamHandler is created
        :param log_level:       the logging level to be used (choices: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL';
                                default: 'INFO')
        :param append:          <bool> whether to append the log message to an existing logfile (1)
                                or to create a new logfile (0); default=1
        """
        # private attributes
        self._captured_stream = ''

        # attributes that need to be present in order to unpickle the logger via __setstate_
        self.name_logfile = name_logfile
        self.fmt_suffix = fmt_suffix
        self.path_logfile = path_logfile
        self.log_level = log_level

        super(SpecHomo_Logger, self).__init__(name_logfile)

        self.path_logfile = path_logfile
        self.formatter_fileH = logging.Formatter('%(asctime)s' + (' [%s]' % fmt_suffix if fmt_suffix else '') +
                                                 ' %(levelname)s:   %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
        self.formatter_ConsoleH = logging.Formatter('%(asctime)s' + (' [%s]' % fmt_suffix if fmt_suffix else '') +
                                                    ':   %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

        if path_logfile:
            # create output directory
            while not os.path.isdir(os.path.dirname(path_logfile)):
                try:
                    os.makedirs(os.path.dirname(path_logfile))
                except OSError as e:
                    if e.errno != 17:
                        raise
                    else:
                        pass

            # create FileHandler
            fileHandler = logging.FileHandler(path_logfile, mode='a' if append else 'w')
            fileHandler.setFormatter(self.formatter_fileH)
            fileHandler.setLevel(log_level)
        else:
            fileHandler = None

        # create StreamHandler
        self.streamObj = StringIO()
        streamHandler = logging.StreamHandler(stream=self.streamObj)
        streamHandler.setFormatter(self.formatter_fileH)
        streamHandler.set_name('StringIO handler')

        # create ConsoleHandler for logging levels DEGUG and INFO -> logging to sys.stdout
        consoleHandler_out = logging.StreamHandler(stream=sys.stdout)  # by default it would go to sys.stderr
        consoleHandler_out.setFormatter(self.formatter_ConsoleH)
        consoleHandler_out.set_name('console handler stdout')
        consoleHandler_out.setLevel(log_level)
        consoleHandler_out.addFilter(LessThanFilter(logging.WARNING))

        # create ConsoleHandler for logging levels WARNING, ERROR, CRITICAL -> logging to sys.stderr
        consoleHandler_err = logging.StreamHandler(stream=sys.stderr)
        consoleHandler_err.setFormatter(self.formatter_ConsoleH)
        consoleHandler_err.setLevel(logging.WARNING)
        consoleHandler_err.set_name('console handler stderr')

        self.setLevel(log_level)

        if not self.handlers:
            if fileHandler:
                self.addHandler(fileHandler)
            self.addHandler(streamHandler)
            self.addHandler(consoleHandler_out)
            self.addHandler(consoleHandler_err)

    def __getstate__(self):
        self.close()
        return self.__dict__

    def __setstate__(self, ObjDict):
        """Define how the attributes of SpecHomo_Logger are unpickled."""
        self.__init__(ObjDict['name_logfile'], fmt_suffix=ObjDict['fmt_suffix'], path_logfile=ObjDict['path_logfile'],
                      log_level=ObjDict['log_level'], append=True)
        ObjDict = self.__dict__
        return ObjDict

    @property
    def captured_stream(self) -> str:
        """Return the already captured logging stream.

        NOTE:
            - set self.captured_stream:
                self.captured_stream = 'any string'

        """
        if not self._captured_stream:
            self._captured_stream = self.streamObj.getvalue()

        return self._captured_stream

    @captured_stream.setter
    def captured_stream(self, string: str):
        assert isinstance(string, str), "'captured_stream' can only be set to a string. Got %s." % type(string)
        self._captured_stream = string

    def close(self):
        """Close all logging handlers."""
        # update captured_stream and flush stream
        self.captured_stream += self.streamObj.getvalue()

        for handler in self.handlers[:]:
            try:
                if handler.get_name() == 'StringIO handler':
                    self.streamObj.flush()
                self.removeHandler(handler)  # if not called with '[:]' the StreamHandlers are left open
                try:
                    handler.flush()
                except ValueError:
                    # ValueError: I/O operation on closed file
                    pass
                handler.close()
            except PermissionError:
                warnings.warn('Could not properly close logfile due to a PermissionError: %s' % sys.exc_info()[1])

        if self.handlers[:]:
            warnings.warn('Not all logging handlers could be closed. Remaining handlers: %s' % self.handlers[:])

    def view_logfile(self):
        """View the log file written to disk."""
        with open(self.path_logfile) as inF:
            print(inF.read())

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True if exc_type is None else False


def close_logger(logger):
    """Close the handlers of the given logging.Logger instance.

    :param logger:  logging.Logger instance or subclass instance
    """
    if logger and hasattr(logger, 'handlers'):
        for handler in logger.handlers[:]:  # if not called with '[:]' the StreamHandlers are left open
            try:
                logger.removeHandler(handler)
                handler.flush()
                handler.close()
            except PermissionError:
                warnings.warn('Could not properly close logfile due to a PermissionError: %s' % sys.exc_info()[1])

        if logger.handlers[:]:
            warnings.warn('Not all logging handlers could be closed. Remaining handlers: %s' % logger.handlers[:])


def shutdown_loggers():
    """Shutdown any currently active loggers."""
    logging.shutdown()


class LessThanFilter(logging.Filter):
    """Filter class to filter log messages by a maximum log level.

    Based on http://stackoverflow.com/questions/2302315/
        how-can-info-and-debug-logging-message-be-sent-to-stdout-and-higher-level-messag
    """

    def __init__(self, exclusive_maximum, name=""):
        """Get an instance of LessThanFilter.

        :param exclusive_maximum:  maximum log level, e.g., logger.WARNING
        :param name:
        """
        super(LessThanFilter, self).__init__(name)
        self.max_level = exclusive_maximum

    def filter(self, record):
        """Filter funtion.

        NOTE: Returns True if logging level of the given record is below the maximum log level.

        :param record:
        :return: bool
        """
        # non-zero return means we log this message
        return True if record.levelno < self.max_level else False
