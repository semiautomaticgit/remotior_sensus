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

import os
from contextlib import redirect_stdout

import numpy as np
from numpy.lib import stride_tricks

from remotior_sensus.core import configurations as cfg
from remotior_sensus.util import raster_vector

try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold
except Exception as error:
    str(error)
try:
    import scipy.stats.distributions as statdistr
except Exception as error:
    str(error)
try:
    from scipy import signal
    from scipy.stats import mode as scipy_mode
    from scipy.ndimage import label
except Exception as error:
    str(error)

try:
    from scipy.ndimage import maximum_filter
    from scipy.ndimage import minimum_filter
    from scipy.ndimage import percentile_filter
    from scipy.ndimage import generic_filter
    from scipy.ndimage import median_filter
except Exception as error:
    str(error)
    # for backward compatibility
    try:
        from scipy.ndimage.filters import maximum_filter
        from scipy.ndimage.filters import minimum_filter
        from scipy.ndimage.filters import percentile_filter
        from scipy.ndimage.filters import generic_filter
        from scipy.ndimage.filters import median_filter
    except Exception as error:
        str(error)

try:
    import torch
    from remotior_sensus.util.pytorch_tools import (
        train_pytorch_model
    )
except Exception as error:
    str(error)


# function to calculate percentile with the closest observation for band_calc
def percentile_calc(array, percentile=90, axis=0):
    if type(array) is list:
        array = np.stack(array)
    # count values that are not nan
    value_count = np.sum(~np.isnan(array), axis=axis)
    # sort array
    sorted_array = np.sort(array, axis=axis)
    # calculate percentile index
    perc_index = np.round(
        (value_count - 1) * float(percentile) / 100
    ).astype(int)
    # get percentile values
    result = np.take_along_axis(
        sorted_array, np.expand_dims(perc_index, axis=axis), axis=axis
    ).squeeze(axis)
    return result


# function to calculate percentile with the closest observation for band_calc
def percentile_calc_pytorch(array, percentile=90, dim=0):
    result = torch.quantile(array, q=percentile / 100, dim=dim)
    return result


# band calculation
# noinspection PyShadowingBuiltins,PyUnusedLocal
def band_calculation(*argv):
    # expose numpy functions
    log = np.ma.log
    log10 = np.ma.log10
    sqrt = np.ma.sqrt
    cos = np.ma.cos
    arccos = np.ma.arccos
    sin = np.ma.sin
    arcsin = np.ma.arcsin
    tan = np.ma.tan
    arctan = np.ma.arctan
    exp = np.ma.exp
    min = np.ma.min
    max = np.ma.max
    sum = np.ma.sum
    percentile = percentile_calc
    median = np.ma.median
    mean = np.ma.mean
    std = np.ma.std
    where = np.ma.where
    nan = np.nan
    # array variable name as defined in cfg.array_function_placeholder
    output_no_data = argv[0][2]
    _array = argv[1][0]
    _array_mask = argv[1][1]
    _array_function_placeholder = np.ma.array(_array, mask=_array_mask)
    nodata_mask = argv[2]
    function_argument = argv[7]
    cfg.logger.log.debug('function_argument: %s' % str(function_argument))
    cfg.logger.log.debug(
        '_array_function_placeholder.shape: %s; type: %s'
        % (str(_array_function_placeholder.shape),
           type(_array_function_placeholder))
    )
    # perform operation
    try:
        _o: np.ma.core.MaskedArray = eval(function_argument)
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False
    # if not array
    if (not isinstance(_o, np.ma.core.MaskedArray)
            and not isinstance(_o, np.ndarray)):
        cfg.logger.log.error('not array ' + str(type(_o)))
        return False
    # check nodata
    cfg.logger.log.debug(
        '_o.shape: %s; _o.n_bytes: %s; _o.dtype: %s'
        % (str(_o.shape), str(_o.nbytes), str(_o.dtype))
    )
    if _o.dtype == bool:
        _o = _o.astype(int)
    # replace nodata
    _o[::, ::].data[_o.mask] = output_no_data
    return [[_o.data, _o.mask], None]


# band calculation pytorch
# noinspection PyShadowingBuiltins,PyUnusedLocal
def band_calculation_pytorch(*argv):
    # expose numpy functions
    log = torch.log
    log10 = torch.log10
    sqrt = torch.sqrt
    cos = torch.cos
    arccos = torch.arccos
    sin = torch.sin
    arcsin = torch.arcsin
    tan = torch.tan
    arctan = torch.arctan
    exp = torch.exp
    min = torch.min
    max = torch.max
    sum = torch.sum
    percentile = percentile_calc_pytorch
    median = torch.median
    mean = torch.mean
    std = torch.std
    where = torch.where
    nan = float('nan')
    # array variable name as defined in cfg.array_function_placeholder
    output_no_data = argv[0][2]
    _array = argv[1][0]
    _array_mask = argv[1][1]
    nodata_mask = argv[2]
    function_argument = argv[7]
    device = argv[14]
    n_processes = argv[15]
    if device == 'cpu':
        torch.set_num_threads(n_processes)
    _array_function_placeholder = torch.from_numpy(
        _array.astype(np.float64)).to(device)
    _array_mask = torch.from_numpy(_array_mask).to(device)
    cfg.logger.log.debug('device: %s; function_argument: %s'
                         % (str(device), str(function_argument)))
    cfg.logger.log.debug('torch: %s; n_processes:%s'
                         % (str(torch.get_num_threads()), str(n_processes)))
    cfg.logger.log.debug(
        '_array_function_placeholder.shape: %s; type: %s'
        % (str(_array_function_placeholder.shape),
           type(_array_function_placeholder))
    )
    nan_array = torch.full(_array_function_placeholder.shape, float('nan'),
                           dtype=torch.float64).to(device)
    _array_function_placeholder[_array_mask] = nan_array[_array_mask]
    # perform operation
    try:
        function_argument.replace('axis=', 'dim=').replace('axis =', 'dim=')
        _o = eval(function_argument)
        if device == 'cuda':
            _o = _o.cpu().numpy()
            device_properties = torch.cuda.get_device_properties(device)
            total_memory = device_properties.total_memory
            memory_reserved = torch.cuda.memory_reserved(device)
            available_ram = int((total_memory - memory_reserved)
                                / (1024 ** 2))
            cfg.logger.log.debug('available_ram: %s' % str(available_ram))
        else:
            _o = _o.numpy()
        np.nan_to_num(_o, copy=False, nan=output_no_data)
        del _array_function_placeholder
        del _array_mask
        del nan_array
        torch.cuda.empty_cache()
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False
    # if not array
    if not isinstance(_o, np.ndarray):
        cfg.logger.log.error('not array ' + str(type(_o)))
        return False
    # check nodata
    cfg.logger.log.debug(
        '_o.shape: %s; _o.n_bytes: %s; _o.dtype: %s'
        % (str(_o.shape), str(_o.nbytes), str(_o.dtype))
    )
    if _o.dtype == bool:
        _o = _o.astype(int)
    # replace nodata
    _o[::, ::][nodata_mask == 1] = output_no_data
    return [[_o, None], None]


# classification maximum likelihood
def classification_maximum_likelihood(*argv):
    scale = argv[0][0]
    offset = argv[0][1]
    output_no_data = argv[0][2]
    _array_function_placeholder = argv[1][0]
    nodata_mask = argv[2]
    x = argv[4]
    y = argv[5]
    # function argument
    signatures = argv[7][cfg.spectral_signatures_framework]
    # get selected signatures
    signatures_table = signatures.table[signatures.table.selected == 1]
    covariance_matrices = argv[7][cfg.covariance_matrices_framework]
    normalization = argv[7][cfg.normalization_values_framework]
    function_variable_list = argv[8]
    macroclass = function_variable_list[0]
    threshold = function_variable_list[1]
    ro_x, ro_y = argv[10]
    output_signature_raster = argv[11]
    out_class = argv[12]
    out_alg = argv[13]
    previous_array = None
    classification_array = None
    if normalization[0] is not None:
        for n in range(0, _array_function_placeholder.shape[2]):
            _array_function_placeholder[::, ::, n] = eval(normalization[0][n])
    for s in signatures_table.signature_id.tolist():
        if macroclass:
            class_id = signatures_table[
                signatures_table.signature_id == s].macroclass_id
        else:
            class_id = signatures_table[
                signatures_table.signature_id == s].class_id
        values = signatures.signatures[s].value
        if _array_function_placeholder.shape[2] < values.shape[0]:
            values = values[0:_array_function_placeholder.shape[2]]
            cfg.logger.log.error(
                'signature values shape; trying to continue anyway'
            )
        elif _array_function_placeholder.shape[2] > values.shape[0]:
            _array_function_placeholder = _array_function_placeholder[
                                          :, :, 0:values.shape[0]
                                          ]
            cfg.logger.log.error(
                'signature values shape; trying to continue anyway'
            )
        cfg.logger.log.debug('signature: %s; values: %s' % (s, str(values)))
        try:
            cov_matrix = covariance_matrices[s]
            # natural logarithm of the determinant of covariance matrix
            (sign, log_det) = np.linalg.slogdet(cov_matrix)
            inverse_cov_matrix = np.linalg.inv(cov_matrix)
            d = _array_function_placeholder - values
            distance_array = - log_det - (
                    np.dot(d, inverse_cov_matrix) * d).sum(axis=2)
            if threshold is True:
                class_threshold = signatures_table[
                    signatures_table.signature_id == s].min_dist_thr
                p = class_threshold / 100
                chi = statdistr.chi2.isf(p, cov_matrix.shape[0])
                chi_threshold = -2 * chi - log_det
                distance_array[::, ::][
                    distance_array < chi_threshold] = cfg.nodata_val
            elif threshold is not False and threshold is not None:
                p = threshold / 100
                chi = statdistr.chi2.isf(p, cov_matrix.shape[0])
                chi_threshold = -2 * chi - log_det
                distance_array[::, ::][
                    distance_array < chi_threshold] = cfg.nodata_val
            if previous_array is None:
                previous_array = distance_array
                classification_array = np.ones(
                    (_array_function_placeholder.shape[0],
                     _array_function_placeholder.shape[1])
                ) * class_id
                classification_array[::, ::][
                    distance_array == cfg.nodata_val] = output_no_data
            else:
                maximum_array = np.maximum(distance_array, previous_array)
                maximum_array[::, ::][
                    maximum_array == cfg.nodata_val] = previous_array[
                    maximum_array == cfg.nodata_val]
                classification_array[
                    maximum_array != previous_array] = class_id
                classification_array[::, ::][
                    maximum_array == cfg.nodata_val] = output_no_data
                previous_array = maximum_array
            if len(output_signature_raster) > 0:
                distance_array[::, ::][
                    distance_array == cfg.nodata_val] = output_no_data
                distance_array[::, ::][
                    nodata_mask == 1] = output_no_data
                write_sig = raster_vector.write_raster(
                    output_signature_raster[s], x - ro_x, y - ro_y,
                    distance_array, output_no_data, scale, offset
                )
                cfg.logger.log.debug('write_sig: %s' % str(write_sig))
        except Exception as err:
            cfg.logger.log.error(str(err))
    if classification_array is not None:
        # write classification
        classification_array[::, ::][
            classification_array == cfg.nodata_val] = output_no_data
        classification_array[::, ::][
            nodata_mask == 1] = output_no_data
        write_class = raster_vector.write_raster(
            out_class, x - ro_x, y - ro_y, classification_array,
            output_no_data, scale, offset
        )
        cfg.logger.log.debug('write_class: %s' % str(write_class))
        # write the algorithm raster
        if out_alg is not None:
            previous_array[::, ::][
                classification_array == cfg.nodata_val] = output_no_data
            previous_array[::, ::][
                nodata_mask == 1] = output_no_data
            write_alg = raster_vector.write_raster(
                out_alg, x - ro_x, y - ro_y, previous_array, output_no_data,
                scale, offset
            )
            cfg.logger.log.debug('write_alg: %s' % str(write_alg))
        cfg.logger.log.debug(
            'classification_array.shape: %s' % str(classification_array.shape)
        )
        return [True, out_class]
    else:
        return [False, out_class]


# classification minimum distance
def classification_minimum_distance(*argv):
    scale = argv[0][0]
    offset = argv[0][1]
    output_no_data = argv[0][2]
    _array_function_placeholder = argv[1][0]
    nodata_mask = argv[2]
    x = argv[4]
    y = argv[5]
    signatures = argv[7][cfg.spectral_signatures_framework]
    # get selected signatures
    signatures_table = signatures.table[signatures.table.selected == 1]
    normalization = argv[7][cfg.normalization_values_framework]
    function_variable_list = argv[8]
    macroclass = function_variable_list[0]
    threshold = function_variable_list[1]
    ro_x, ro_y = argv[10]
    output_signature_raster = argv[11]
    out_class = argv[12]
    out_alg = argv[13]
    previous_array = None
    classification_array = None
    if normalization[0] is not None:
        for n in range(0, _array_function_placeholder.shape[2]):
            _array_function_placeholder[::, ::, n] = eval(normalization[0][n])
    for s in signatures_table.signature_id.tolist():
        if macroclass:
            class_id = signatures_table[
                signatures_table.signature_id == s].macroclass_id
        else:
            class_id = signatures_table[
                signatures_table.signature_id == s].class_id
        values = signatures.signatures[s].value
        if _array_function_placeholder.shape[2] < values.shape[0]:
            values = values[0:_array_function_placeholder.shape[2]]
            cfg.logger.log.error(
                'signature values shape; trying to continue anyway'
            )
        elif _array_function_placeholder.shape[2] > values.shape[0]:
            _array_function_placeholder = _array_function_placeholder[
                                          :, :, 0:values.shape[0]
                                          ]
            cfg.logger.log.error(
                'signature values shape; trying to continue anyway'
            )
        # Euclidean distance
        distance_array = np.sqrt(
            ((_array_function_placeholder - values) ** 2).sum(axis=2)
        )
        if threshold is True:
            class_threshold = signatures_table[
                signatures_table.signature_id == s].min_dist_thr
            distance_array[::, ::][
                distance_array < class_threshold] = cfg.nodata_val_Int32
        elif threshold is not False and threshold is not None:
            distance_array[::, ::][
                distance_array < threshold] = cfg.nodata_val_Int32
        if previous_array is None:
            previous_array = distance_array
            classification_array = np.ones(
                (_array_function_placeholder.shape[0],
                 _array_function_placeholder.shape[1])
            ) * class_id
        else:
            minimum_array = np.minimum(distance_array, previous_array)
            classification_array[minimum_array != previous_array] = class_id
            previous_array = minimum_array
        classification_array[::, ::][
            distance_array == cfg.nodata_val_Int32] = output_no_data
        if len(output_signature_raster) > 0:
            distance_array[::, ::][
                distance_array == cfg.nodata_val_Int32] = output_no_data
            distance_array[::, ::][nodata_mask == 1] = output_no_data
            write_sig = raster_vector.write_raster(
                output_signature_raster[s], x - ro_x, y - ro_y, distance_array,
                output_no_data, scale, offset
            )
            cfg.logger.log.debug('write_sig: %s' % str(write_sig))
    # write classification
    classification_array[::, ::][
        classification_array == cfg.nodata_val_Int32] = output_no_data
    classification_array[::, ::][nodata_mask == 1] = output_no_data
    write_class = raster_vector.write_raster(
        out_class, x - ro_x, y - ro_y, classification_array, output_no_data,
        scale, offset
    )
    cfg.logger.log.debug('write_class: %s' % str(write_class))
    # write the algorithm raster
    if out_alg is not None:
        previous_array[::, ::][
            classification_array == cfg.nodata_val_Int32] = output_no_data
        previous_array[::, ::][nodata_mask == 1] = output_no_data
        write_alg = raster_vector.write_raster(
            out_alg, x - ro_x, y - ro_y, previous_array, output_no_data, scale,
            offset
        )
        cfg.logger.log.debug('write_alg: %s' % str(write_alg))
    cfg.logger.log.debug(
        'classification_array.shape: %s' % str(classification_array.shape)
    )
    return [True, out_class]


# classification spectral angle mapping
def classification_spectral_angle_mapping(*argv):
    scale = argv[0][0]
    offset = argv[0][1]
    output_no_data = argv[0][2]
    _array_function_placeholder = argv[1][0]
    nodata_mask = argv[2]
    x = argv[4]
    y = argv[5]
    signatures = argv[7][cfg.spectral_signatures_framework]
    # get selected signatures
    signatures_table = signatures.table[signatures.table.selected == 1]
    normalization = argv[7][cfg.normalization_values_framework]
    function_variable_list = argv[8]
    macroclass = function_variable_list[0]
    threshold = function_variable_list[1]
    ro_x, ro_y = argv[10]
    output_signature_raster = argv[11]
    out_class = argv[12]
    out_alg = argv[13]
    previous_array = None
    classification_array = None
    if normalization[0] is not None:
        for n in range(0, _array_function_placeholder.shape[2]):
            _array_function_placeholder[::, ::, n] = eval(normalization[0][n])
    for s in signatures_table.signature_id.tolist():
        if macroclass:
            class_id = signatures_table[
                signatures_table.signature_id == s].macroclass_id
        else:
            class_id = signatures_table[
                signatures_table.signature_id == s].class_id
        values = signatures.signatures[s].value
        if _array_function_placeholder.shape[2] < values.shape[0]:
            values = values[0:_array_function_placeholder.shape[2]]
            cfg.logger.log.error(
                'signature values shape; trying to continue anyway'
            )
        elif _array_function_placeholder.shape[2] > values.shape[0]:
            _array_function_placeholder = _array_function_placeholder[
                                          :, :, 0:values.shape[0]
                                          ]
            cfg.logger.log.error(
                'signature values shape; trying to continue anyway'
            )
        # spectral angle
        distance_array = np.arccos(
            (_array_function_placeholder * values).sum(axis=2) / np.sqrt(
                (_array_function_placeholder ** 2).sum(axis=2) * (
                        values ** 2).sum()
            )
        ) * 180 / np.pi
        if threshold is True:
            class_threshold = signatures_table[
                signatures_table.signature_id == s].min_dist_thr
            distance_array[::, ::][
                distance_array < class_threshold] = cfg.nodata_val_Int32
        elif threshold is not False and threshold is not None:
            distance_array[::, ::][
                distance_array < threshold] = cfg.nodata_val_Int32
        if previous_array is None:
            previous_array = distance_array
            classification_array = np.ones(
                (_array_function_placeholder.shape[0],
                 _array_function_placeholder.shape[1])
            ) * class_id
        else:
            minimum_array = np.minimum(distance_array, previous_array)
            classification_array[minimum_array != previous_array] = class_id
            previous_array = minimum_array
        classification_array[::, ::][
            distance_array == cfg.nodata_val_Int32] = output_no_data
        if len(output_signature_raster) > 0:
            distance_array[::, ::][
                distance_array == cfg.nodata_val_Int32] = output_no_data
            distance_array[::, ::][nodata_mask == 1] = output_no_data
            write_sig = raster_vector.write_raster(
                output_signature_raster[s], x - ro_x, y - ro_y, distance_array,
                output_no_data, scale, offset
            )
            cfg.logger.log.debug('write_sig: %s' % str(write_sig))
    # write classification
    classification_array[::, ::][
        classification_array == cfg.nodata_val_Int32] = output_no_data
    classification_array[::, ::][nodata_mask == 1] = output_no_data
    write_class = raster_vector.write_raster(
        out_class, x - ro_x, y - ro_y, classification_array, output_no_data,
        scale, offset
    )
    cfg.logger.log.debug('write_class: %s' % str(write_class))
    # write the algorithm raster
    if out_alg is not None:
        previous_array[::, ::][
            classification_array == cfg.nodata_val_Int32] = output_no_data
        previous_array[::, ::][nodata_mask == 1] = output_no_data
        write_alg = raster_vector.write_raster(
            out_alg, x - ro_x, y - ro_y, previous_array, output_no_data, scale,
            offset
        )
        cfg.logger.log.debug('write_alg: %s' % str(write_alg))
    cfg.logger.log.debug(
        'classification_array.shape: %s' % str(classification_array.shape)
    )
    return [True, out_class]


# calculation of spectral distance
def spectral_distance(*argv):
    # array variable name as defined in cfg.array_function_placeholder
    output_no_data = argv[0][2]
    _array_function_placeholder = argv[1][0]
    nodata_mask = argv[2]
    # number of bands
    function_argument = argv[7]
    # algorithm name
    function_variable = argv[8][0]
    threshold = argv[8][1]
    cfg.logger.log.debug(
        'function_argument: %s; function_variable: %s'
        % (str(function_argument), str(function_variable))
    )
    a = _array_function_placeholder[::, ::, :function_argument]
    b = _array_function_placeholder[::, ::, function_argument:]
    if function_variable.lower() == cfg.minimum_distance_a:
        # Euclidean distance
        _o = np.sqrt(((a - b) ** 2).sum(axis=2))
    else:
        # spectral angle
        _o = np.arccos(
            (a * b).sum(axis=2) /
            np.sqrt((a ** 2).sum(axis=2) * (b ** 2).sum(axis=2))
        ) * 180 / np.pi
    if threshold is not None:
        _o = _o > threshold
    # check nodata
    cfg.logger.log.debug(
        '_o: %s; nodata_mask: %s' % (str(_o.shape), str(nodata_mask.shape))
    )
    if nodata_mask is not None:
        _o[::, ::][nodata_mask == 1] = output_no_data
    return [[_o, None], None]


# classification through scikit-learn model
def classification_scikit(*argv):
    scale = argv[0][0]
    offset = argv[0][1]
    output_no_data = argv[0][2]
    _array_function_placeholder = argv[1][0]
    nodata_mask = argv[2]
    x = argv[4]
    y = argv[5]
    classifier = argv[7][cfg.model_classifier_framework]
    cfg.logger.log.debug('classifier: %s' % str(classifier.classes_))
    normalization = argv[7][cfg.normalization_values_framework]
    function_variable_list = argv[8]
    x_min_piece, y_min_piece = argv[10]
    threshold = function_variable_list[1]
    out_class = argv[12]
    out_alg = argv[13]
    x_array = None
    if normalization[0] is not None:
        for n in range(0, _array_function_placeholder.shape[2]):
            _array_function_placeholder[::, ::, n] = eval(normalization[0][n])
    for n in range(_array_function_placeholder.shape[2]):
        # replace nodata
        _array_function_placeholder[::, ::, n][
            nodata_mask == 1] = cfg.nodata_val
        if x_array is None:
            x_array = _array_function_placeholder[::, ::, n].ravel()
        else:
            x_array = np.vstack(
                [x_array, _array_function_placeholder[::, ::, n].ravel()]
            )
    # replace nan
    x_array[np.isnan(x_array)] = cfg.nodata_val
    # prediction
    if not threshold and out_alg is None:
        _prediction = classifier.predict(x_array.T)
        classification_array = _prediction.reshape(
            _array_function_placeholder.shape[0],
            _array_function_placeholder.shape[1]
        )
        _prediction = None
        # write classification
        classification_array[::, ::][nodata_mask == 1] = output_no_data
        try:
            write_class = raster_vector.write_raster(
                out_class, x - x_min_piece, y - y_min_piece,
                classification_array, output_no_data, scale, offset
            )
            cfg.logger.log.debug(
                'write_class: %s; classification_array.shape: %s'
                % (str(write_class), str(classification_array.shape))
            )
        except Exception as err:
            cfg.logger.log.debug(
                'classification_array.shape: %s; x: %s; y: %s: '
                % (str(classification_array.shape), str(x), str(y))
            )
            cfg.logger.log.error(str(err))
    # write the probability raster
    elif out_alg is not None:
        try:
            _prediction_proba = classifier.predict_proba(x_array.T)
            prediction_proba_array = np.max(_prediction_proba, axis=1).reshape(
                _array_function_placeholder.shape[0],
                _array_function_placeholder.shape[1]
            )
            classification_argmax = np.argmax(
                _prediction_proba, axis=1
            ).reshape(
                _array_function_placeholder.shape[0],
                _array_function_placeholder.shape[1]
            )
            _prediction_proba = None
            classes = classifier.classes_
            classification_array = np.zeros_like(classification_argmax)
            for c, class_c in enumerate(classes):
                classification_array = np.where(
                    classification_argmax == c, class_c, classification_array
                )
            if threshold is not False:
                classification_array[::, ::][
                    prediction_proba_array < threshold] = output_no_data
            prediction_proba_array[::, ::][nodata_mask == 1] = output_no_data
            write_alg = raster_vector.write_raster(
                out_alg, x - x_min_piece, y - y_min_piece,
                prediction_proba_array, output_no_data, scale, offset
            )
            cfg.logger.log.debug('write_alg: %s' % str(write_alg))
        except Exception as err:
            cfg.logger.log.error(str(err))
            _prediction = classifier.predict(x_array.T)
            classification_array = _prediction.reshape(
                _array_function_placeholder.shape[0],
                _array_function_placeholder.shape[1]
            )
            _prediction = None
            # write classification
            classification_array[::, ::][nodata_mask == 1] = output_no_data
        # write classification
        classification_array[::, ::][nodata_mask == 1] = output_no_data
        write_class = raster_vector.write_raster(
            out_class, x - x_min_piece, y - y_min_piece, classification_array,
            output_no_data, scale, offset
        )
        cfg.logger.log.debug(
            'write_class: %s; classification_array.shape: %s'
            % (str(write_class), str(classification_array.shape))
        )
    return [True, out_class]


# classification through pytorch model
def classification_pytorch(*argv):
    scale = argv[0][0]
    offset = argv[0][1]
    output_no_data = argv[0][2]
    _array_function_placeholder = argv[1][0]
    nodata_mask = argv[2]
    x = argv[4]
    y = argv[5]
    classifier = argv[7][cfg.model_classifier_framework]
    normalization = argv[7][cfg.normalization_values_framework]
    function_variable_list = argv[8]
    x_min_piece, y_min_piece = argv[10]
    threshold = function_variable_list[1]
    out_class = argv[12]
    out_alg = argv[13]
    x_array = None
    if normalization[0] is not None:
        for n in range(0, _array_function_placeholder.shape[2]):
            _array_function_placeholder[::, ::, n] = eval(normalization[0][n])
    for n in range(_array_function_placeholder.shape[2]):
        # replace nodata
        _array_function_placeholder[::, ::, n][
            nodata_mask == 1] = cfg.nodata_val
        if x_array is None:
            x_array = _array_function_placeholder[::, ::, n].ravel()
        else:
            x_array = np.vstack(
                [x_array, _array_function_placeholder[::, ::, n].ravel()]
            )
    # replace nan
    x_array[np.isnan(x_array)] = cfg.nodata_val
    # prediction
    torch.set_num_threads(1)
    classifier.eval()
    data_type = eval('torch.%s' % str(x_array.dtype))
    x_array = torch.tensor(x_array.T, dtype=data_type)
    if not threshold and out_alg is None:
        _prediction = classifier(x_array).argmax(1).numpy()
        classification_array = _prediction.reshape(
            _array_function_placeholder.shape[0],
            _array_function_placeholder.shape[1]
        )
        _prediction = None
        # write classification
        classification_array[::, ::][nodata_mask == 1] = output_no_data
        try:
            write_class = raster_vector.write_raster(
                out_class, x - x_min_piece, y - y_min_piece,
                classification_array, output_no_data, scale, offset
            )
            cfg.logger.log.debug('write_class: %s' % str(write_class))
            cfg.logger.log.debug(
                'classification_array.shape: %s' % str(
                    classification_array.shape
                )
            )
        except Exception as err:
            cfg.logger.log.debug(
                'classification_array.shape: %s; x: %s; y: %s'
                % (str(classification_array.shape), str(x), str(y))
            )
            cfg.logger.log.error(str(err))
    # write the probability raster
    elif out_alg is not None:
        try:
            _softmax = torch.softmax(classifier(x_array), dim=1)
            prediction_proba_array, classification_array = _softmax.topk(
                1, dim=1
            )
            _softmax = None
            prediction_proba_array = prediction_proba_array.detach().numpy(

            ).reshape(
                _array_function_placeholder.shape[0],
                _array_function_placeholder.shape[1]
            )
            classification_array = classification_array.detach().numpy(
            ).reshape(
                _array_function_placeholder.shape[0],
                _array_function_placeholder.shape[1]
            )
            if threshold is not False:
                classification_array[::, ::][
                    prediction_proba_array < threshold] = output_no_data
            prediction_proba_array[::, ::][nodata_mask == 1] = output_no_data
            write_alg = raster_vector.write_raster(
                out_alg, x - x_min_piece, y - y_min_piece,
                prediction_proba_array, output_no_data, scale, offset
            )
            cfg.logger.log.debug('write_alg: %s' % str(write_alg))
        except Exception as err:
            cfg.logger.log.error(str(err))
            prediction = classifier(x_array).argmax(1).numpy()
            classification_array = prediction.reshape(
                _array_function_placeholder.shape[0],
                _array_function_placeholder.shape[1]
            )
            # write classification
            classification_array[::, ::][nodata_mask == 1] = output_no_data
        # write classification
        classification_array[::, ::][nodata_mask == 1] = output_no_data
        write_class = raster_vector.write_raster(
            out_class, x - x_min_piece, y - y_min_piece, classification_array,
            output_no_data, scale, offset
        )
        cfg.logger.log.debug(
            'write_class: %s; classification_array.shape: %s'
            % (str(write_class), str(classification_array.shape))
        )
    return [True, out_class]


# calculate PCA
def calculate_pca(*argv):
    # array variable name as defined in cfg.array_function_placeholder
    output_no_data = argv[0][2]
    _array_function_placeholder = argv[1][0]
    nodata_mask = argv[2]
    # principal component vector
    function_argument = argv[7]
    # band mean vector
    function_variable = argv[8]
    cfg.logger.log.debug(
        'function_argument: %s; function_variable: %s'
        % (str(function_argument), str(function_variable))
    )
    # perform calculation
    _o = (function_argument * (
            _array_function_placeholder - function_variable
    )).sum(axis=2, dtype=np.float32)
    # check nodata
    cfg.logger.log.debug(
        '_o: %s; nodata_mask: %s' % (str(_o.shape), str(nodata_mask.shape))
    )
    if nodata_mask is not None:
        _o[::, ::][nodata_mask == 1] = output_no_data
    # update masked array
    _o_mask = np.zeros_like(_o, dtype=bool)
    # TODO output mask
    return [[_o, _o_mask], None]


# reclassify raster
def reclassify_raster(*argv):
    output_no_data = argv[0][2]
    raster_array_band = argv[1][0]
    raster_mask_array_band = argv[1][1]
    # conditions
    function_argument = argv[7]
    # variable raster name
    function_variable = argv[8]
    cfg.logger.log.debug('start')
    _o = None
    replace_nodata = True
    try:
        old = function_argument['old_value']
        new = function_argument['new_value']
        assert old.astype(int).all()
        _raster = raster_array_band[:, :, 0]
        _raster_mask = raster_mask_array_band[:, :, 0]
        # if all integer values
        if np.all(_raster.astype(int) == _raster):
            # create empty reclass array of length equal to maximum value
            reclass = np.zeros(
                max(
                    old.astype(int).max(), raster_array_band.astype(int).max()
                ) + 1
            ) * np.nan
            # fill array with new values at index corresponding to old value
            reclass[old.astype(int)] = new
            # perform reclassification
            _o = reclass[_raster.astype(int)]
            _o[::, ::][np.isnan(_o)] = _raster[::, ::][np.isnan(_o)]
            _o_mask = _raster_mask
        else:
            # raise exception to try expressions
            raise Exception
    except Exception as err:
        str(err)
        _raster = None
        # raster array
        _o = np.copy(raster_array_band[:, :, 0])
        _o_mask = np.copy(raster_mask_array_band[:, :, 0])
        _x = raster_array_band[:, :, 0]
        for i in range(function_argument.shape[0]):
            cfg.logger.log.debug(str(function_argument[i]))
            # if reclassify from nodata to new value
            if 'nan' in function_argument[i][cfg.old_value]:
                try:
                    # replace nodata
                    _o[::, ::][_o_mask] = np.array(
                        float(function_argument[i][cfg.new_value])
                    )
                    _o_mask[_o_mask] = 0
                    replace_nodata = False
                except Exception as err:
                    str(err)
            else:
                # create condition for single values
                try:
                    _o[_x == int(function_argument[i][cfg.old_value])] = int(
                        float(
                            function_argument[i][
                                cfg.new_value].lower().replace(
                                'np.nan', str(output_no_data)
                            ).replace('nan', str(output_no_data))
                        )
                    )
                except Exception as err:
                    str(err)
                    # execute conditional expression
                    try:
                        exp = ('_o[%s] = %s' % (
                            function_argument[i][cfg.old_value].replace(
                                function_variable, '_x'
                            ),
                            str(
                                int(
                                    float(
                                        function_argument[i][
                                            cfg.new_value].lower().replace(
                                            'np.nan', str(output_no_data)
                                        ).replace('nan', str(output_no_data))
                                    )
                                )
                            )
                        )
                               )
                        exec(exp)
                    except Exception as err:
                        cfg.logger.log.error(
                            '%s; function_argument[i]: %s' % (
                                str(err), str(function_argument[i]))
                        )
                        break
    if replace_nodata:
        try:
            # replace nodata
            _o[::, ::][_o_mask] = output_no_data
            _o_mask = np.zeros_like(_o, dtype=bool)
        except Exception as err:
            str(err)
    cfg.logger.log.debug('end')
    return [[_o, _o_mask], None]


# calculate bands covariance
def bands_covariance(*argv):
    cfg.logger.log.debug('start')
    raster_array_band = argv[1][0]
    nodata_mask = argv[2]
    band_number = argv[7]
    band_dict = argv[8]
    covariance_dictionary = {}
    if band_dict['normalize'] is True:
        for _x in band_number:
            raster_array_band[::, ::, _x] = (
                    (raster_array_band[::, ::, _x] - band_dict['mean_%s' % _x])
                    / np.sqrt(band_dict['variance_%s' % _x])
            )
    # iterate bands
    for _x in band_number:
        # calculate covariance SUM((x - Mean_x) * (y - Mean_y))
        for _y in band_number:
            # mask nodata
            if _x == _y:
                x = raster_array_band[::, ::, _x][nodata_mask != 1].ravel()
                y = x
            else:
                x = raster_array_band[::, ::, _x][nodata_mask != 1].ravel()
                y = raster_array_band[::, ::, _y][nodata_mask != 1].ravel()
            # covariance
            cov = ((x - band_dict['mean_%s' % _x]) * (
                    y - band_dict['mean_%s' % _y])).sum()
            covariance_dictionary['cov_%s-%s' % (_x, _y)] = cov
    cfg.logger.log.debug('end')
    return [None, covariance_dictionary]


# calculate raster pixel count
def raster_pixel_count(*argv):
    cfg.logger.log.debug('start')
    raster_array_band = argv[1][0]
    nodata_mask = argv[2]
    band_number = argv[7]
    band = raster_array_band[nodata_mask != 1].ravel()
    count = band.shape[0]
    band_sum = np.nansum(band)
    variance = np.square(band - band_sum / count) / count
    raster_dictionary = {
        'count_%s' % str(band_number): count,
        'sum_%s' % str(band_number): band_sum,
        'var_%s' % str(band_number): variance.sum()
    }
    cfg.logger.log.debug('end')
    return [None, raster_dictionary]


# calculate raster unique values with sum
def raster_unique_values_with_sum(*argv):
    cfg.logger.log.debug('start')
    raster_array_band = argv[1][0]
    try:
        nodata_mask = argv[2]
        b = raster_array_band[nodata_mask != 1]
        stats = np.array(np.unique(b[~np.isnan(b)], return_counts=True))
    except Exception as err:
        stats = np.array(
            np.unique(
                raster_array_band[~np.isnan(raster_array_band)],
                return_counts=True
            )
        )
        cfg.logger.log.error(str(err))
    cfg.logger.log.debug('end')
    return [None, stats]


# calculate raster class unique values with sum
def raster_class_unique_values_with_sum(*argv):
    cfg.logger.log.debug('start')
    raster_array_band = argv[1][0]
    nodata_mask = argv[2]
    band_number = argv[7]
    classes = argv[8]
    raster_dictionary = {}
    classification = raster_array_band[::, ::, -1][nodata_mask != 1].ravel()
    # iterate bands
    for _x in band_number:
        band = raster_array_band[::, ::, _x][nodata_mask != 1].ravel()
        # iterate classes
        for c in classes:
            band_c = band[classification == c]
            count = band_c.shape[0]
            band_sum = np.nansum(band_c)
            raster_dictionary['count_%i_%i' % (c, _x)] = count
            raster_dictionary['sum_%i_%i' % (c, _x)] = band_sum
    cfg.logger.log.debug('end')
    return [None, raster_dictionary]


# calculate raster unique values of combinations
def raster_unique_values(*argv):
    cfg.logger.log.debug('start')
    raster_array_band = argv[1][0]
    nodata_mask = argv[2]
    # stack arrays
    try:
        arr = raster_array_band[:, :, 0][nodata_mask != 1].ravel()
        for i in range(1, raster_array_band.shape[2]):
            arr = np.vstack(
                (arr, raster_array_band[:, :, i][nodata_mask != 1].ravel())
            )
    except Exception as err:
        str(err)
        arr = raster_array_band[:, :, 0].ravel()
        for i in range(1, raster_array_band.shape[2]):
            arr = np.vstack((arr, raster_array_band[:, :, i].ravel()))
    arr = arr.T
    arr = arr[~np.isnan(arr).any(axis=1)]
    # adapted from Jaime answer at
    # https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    b = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    ff, index_a = np.unique(b, return_index=True, return_counts=False)
    cfg.logger.log.debug('end')
    return [None, arr[index_a]]


# calculate raster dilation
def raster_dilation(*argv):
    cfg.logger.log.debug('start')
    output_no_data = argv[0][2]
    raster_array_band = argv[1][0]
    nodata_mask = argv[2]
    structure = argv[7]
    function_variable_list = argv[8]
    a = np.nan_to_num(raster_array_band[:, :, 0])
    # value dictionary
    val_dict = {}
    # calculate
    for i in function_variable_list:
        try:
            val_dict['arr_%s' % str(i)] = (signal.oaconvolve(
                a == i, structure, mode='same'
            ) > 0.999)
        except Exception as err:
            str(err)
            # if scipy version < 1.4
            val_dict['arr_%s' % str(i)] = (
                    signal.fftconvolve(a == i, structure, mode='same') > 0.999)
    # core
    var_array = np.array(function_variable_list)
    core = ~np.isin(a, var_array)
    # dilation
    o = np.copy(a)
    try:
        for v in function_variable_list:
            o[(core * val_dict['arr_%s' % str(v)]) > 0] = v
    except Exception as err:
        cfg.logger.log.error(str(err))
    # replace nodata
    o_mask = nodata_mask == 1
    o[::, ::][o_mask] = output_no_data
    cfg.logger.log.debug('end')
    return [[o, o_mask], None]


# calculate raster erosion
def raster_erosion(*argv):
    cfg.logger.log.debug('start')
    output_no_data = argv[0][2]
    raster_array_band = argv[1][0]
    raster_mask_array_band = argv[1][1]
    nodata_mask = argv[2]
    structure = argv[7]
    function_variable_list = argv[8]
    a = np.copy(raster_array_band[::, ::, 0])
    a_mask = np.copy(raster_mask_array_band[:, :, 0])
    np.copyto(a, cfg.nodata_val, where=np.isnan(raster_array_band[:, :, 0]))
    np.copyto(a, cfg.nodata_val, where=nodata_mask == 1)
    # unique value list
    unique_val = np.unique(a)
    unique_val_list = list(unique_val.astype(int))
    try:
        unique_val_list.remove(cfg.nodata_val)
    except Exception as err:
        str(err)
    # iteration of erosion size
    for _ in range(function_variable_list[0]):
        # empty array
        erosion = np.zeros(a.shape)
        # structure core pixels
        try:
            sum_structure = signal.oaconvolve(
                np.ones(a.shape), structure, mode='same'
            )
        except Exception as err:
            str(err)
            # if scipy version < 1.4
            sum_structure = signal.fftconvolve(
                np.ones(a.shape), structure, mode='same'
            )
        # value that fills erosion
        fill_value = np.ones(a.shape) * float(cfg.nodata_val)
        # maximum of all value convolution
        max_sum_unique = np.zeros(a.shape)
        # iteration of erosion values
        for i in unique_val_list:
            # frequency of values
            try:
                sum_unique = signal.oaconvolve(a == i, structure, mode='same')
            except Exception as err:
                str(err)
                # if scipy version < 1.4
                sum_unique = signal.fftconvolve(a == i, structure, mode='same')
            # expand
            if i not in function_variable_list[1]:
                # fill with most frequent value
                fill_value[sum_unique > max_sum_unique] = float(i)
                # maximum of all value convolution
                # noinspection PyUnresolvedReferences
                max_sum_unique[sum_unique > max_sum_unique] = sum_unique[
                    sum_unique > max_sum_unique]
            # erode
            else:
                # erosion values
                erosion[((sum_structure - sum_unique) > 0.01) & (a == i)] = 1
        np.copyto(a, fill_value, where=erosion == 1)
    # replace nodata
    a_mask[(nodata_mask == 1) | (a == cfg.nodata_val)] = 1
    a[a_mask] = output_no_data
    cfg.logger.log.debug('end')
    return [[a, a_mask], None]


# calculate raster resample
def raster_resample(*argv):
    cfg.logger.log.debug('start')
    output_no_data = argv[0][2]
    raster_array_band = argv[1][0]
    x_y_size = argv[7]
    function_variable_list = argv[8]
    resize_factor_x = x_y_size[0] // function_variable_list[0]
    resize_factor_y = x_y_size[1] // function_variable_list[1]
    _a = np.nan_to_num(raster_array_band[:, :, 0])
    stride_shape_x = _a.shape[1] // resize_factor_x
    stride_shape_y = _a.shape[0] // resize_factor_y
    # pad raster
    if stride_shape_y < _a.shape[0] // resize_factor_y:
        pad_y = int(round((stride_shape_y + 1) * resize_factor_y))
        pad_shape = [(0, pad_y - _a.shape[0]), (0, 0)]
        # noinspection PyTypeChecker
        _a = np.pad(_a, pad_width=pad_shape, mode='constant',
                    constant_values=output_no_data)
    # get strides
    stride_shape = (int(stride_shape_y), int(stride_shape_x),
                    int(resize_factor_y), int(resize_factor_x))
    sub_array_strides = (_a.strides[0] * int(resize_factor_y),
                         _a.strides[1] * int(resize_factor_x), *_a.strides)
    sub_arrays = stride_tricks.as_strided(
        _a, shape=stride_shape, strides=sub_array_strides)
    reshaped = sub_arrays.reshape(int(stride_shape_y * stride_shape_x),
                                  int(resize_factor_y * resize_factor_x))
    # calculate mode
    # noinspection PyRedundantParentheses,PyUnresolvedReferences
    subarray_modes = scipy_mode(reshaped, axis=(1), nan_policy='omit',
                                keepdims=True).mode
    _a = None
    o = subarray_modes.reshape(int(stride_shape_y), int(stride_shape_x))
    # update masked array
    o_mask = o == output_no_data
    cfg.logger.log.debug('end')
    return [[o, o_mask], None]


# calculate raster neighbor
def raster_neighbor(*argv):
    cfg.logger.log.debug('start')
    raster_array_band = argv[1][0]
    nodata_mask = argv[2]
    structure = argv[7]
    function_variable_list = argv[8]
    cfg.logger.log.debug(
        'structure.shape: %s; function_variable_list[0]: %s; '
        'raster_array_band[0, 0 , 0]: %s'
        % (str(structure.shape), str(function_variable_list[0]),
           str(raster_array_band[0, 0, 0]))
    )
    o = None
    # calculate
    if 'nansum' in function_variable_list[0]:
        try:
            o = np.round(
                signal.oaconvolve(
                    np.nan_to_num(raster_array_band[:, :, 0]), structure,
                    mode='same'
                ), 6
            )
        except Exception as err:
            cfg.logger.log.error(str(err))
            # if scipy version < 1.4
            o = np.round(
                signal.fftconvolve(
                    np.nan_to_num(raster_array_band[:, :, 0]), structure,
                    mode='same'
                ), 6
            )
        o[::, ::][np.isnan(raster_array_band[:, :, 0])] = np.nan
    elif 'nanmean' in function_variable_list[0]:
        try:
            o = np.round(
                np.divide(
                    signal.oaconvolve(
                        np.nan_to_num(raster_array_band[:, :, 0]), structure,
                        mode='same'
                    ),
                    signal.oaconvolve(
                        ~np.isnan(raster_array_band[:, :, 0]), structure,
                        mode='same'
                    )
                ), 6
            )
        except Exception as err:
            cfg.logger.log.error(str(err))
            # if scipy version < 1.4
            o = np.round(
                np.divide(
                    signal.fftconvolve(
                        np.nan_to_num(raster_array_band[:, :, 0]), structure,
                        mode='same'
                    ),
                    signal.fftconvolve(
                        ~np.isnan(raster_array_band[:, :, 0]), structure,
                        mode='same'
                    )
                ), 6
            )
        o[::, ::][np.isnan(raster_array_band[:, :, 0])] = np.nan
    elif 'nanmax' in function_variable_list[0]:
        o = maximum_filter(
            raster_array_band[:, :, 0], footprint=structure, mode='constant',
            cval=np.nan
        )
    elif 'nanmin' in function_variable_list[0]:
        o = minimum_filter(
            raster_array_band[:, :, 0], footprint=structure, mode='constant',
            cval=np.nan
        )
    elif 'median' in function_variable_list[0]:
        o = median_filter(
            raster_array_band[:, :, 0], footprint=structure, mode='constant',
            cval=np.nan
        )
    elif 'count' in function_variable_list[0]:
        try:
            o = np.round(
                signal.oaconvolve(
                    ~np.isnan(raster_array_band[:, :, 0]), structure,
                    mode='same'
                ), 6
            )
        except Exception as err:
            cfg.logger.log.error(str(err))
            # if scipy version < 1.4
            o = np.round(
                signal.fftconvolve(
                    ~np.isnan(raster_array_band[:, :, 0]), structure,
                    mode='same'
                ), 6
            )
        o[::, ::][np.isnan(raster_array_band[:, :, 0])] = np.nan
    elif 'std' in function_variable_list[0]:
        o = generic_filter(
            raster_array_band[:, :, 0], np.std, footprint=structure,
            mode='constant', cval=np.nan
        )
    elif 'percentile' in function_variable_list[0]:
        o = percentile_filter(
            raster_array_band[:, :, 0],
            percentile=int(function_variable_list[0].split(',')[1].strip(')')),
            footprint=structure, mode='constant', cval=np.nan
        )
    # update masked array
    o_mask = (nodata_mask == 1) | (o == cfg.nodata_val)
    cfg.logger.log.debug('end')
    return [[o, o_mask], None]


# calculate cross rasters
def cross_rasters(*argv):
    cfg.logger.log.debug('start')
    output_no_data = argv[0][2]
    array_function_placeholder = argv[1][0]
    nodata_mask = argv[2]
    output_list = argv[6]
    function_argument = argv[7]
    function_variable = argv[8]
    _a = eval(
        function_variable.replace(
            cfg.array_function_placeholder, 'array_function_placeholder'
        )
    )
    o = np.searchsorted(function_argument, _a.ravel(), side='right').reshape(
        array_function_placeholder.shape[0],
        array_function_placeholder.shape[1]
    )
    _a = None
    if nodata_mask is not None:
        np.copyto(o, output_no_data, where=nodata_mask[::, ::] != 0,
                  casting='unsafe'
                  )
        stats = np.array(
            np.unique(o[nodata_mask[::, ::] == 0], return_counts=True)
        )
    else:
        try:
            o[np.isnan(array_function_placeholder[:, :, 0])] = np.nan
            stats = np.array(np.unique(o[~np.isnan(o)], return_counts=True))
        except Exception as err:
            str(err)
            try:
                stats = np.array(
                    np.unique(o[~np.isnan(o)], return_counts=True)
                )
            except Exception as err:
                cfg.logger.log.error(str(err))
                stats = None
    cfg.logger.log.debug('end')
    if output_list is None:
        return [None, stats]
    else:
        return [[o, None], stats]


# calculate zonal rasters
def zonal_rasters(*argv):
    cfg.logger.log.debug('start')
    _output_no_data = argv[0][2]
    array_function_placeholder = argv[1][0]
    function_argument = argv[7]
    _function_variable = argv[8]
    raster_array_band = array_function_placeholder[::, ::, 1]
    try:
        nodata_mask = argv[2]
        b = raster_array_band[nodata_mask != 1]
        classes = np.array(np.unique(b[~np.isnan(b)], return_counts=False))
    except Exception as err:
        str(err)
        classes = np.array(
            np.unique(
                raster_array_band[~np.isnan(raster_array_band)],
                return_counts=False
            )
        )
    result_list = []
    result_dict = {}
    for c in classes:
        result_list.append(c)
        function_dict = {}
        for f in function_argument:
            _a = np.where(array_function_placeholder[::, ::, 1] == c,
                          array_function_placeholder[::, ::, 0], np.nan)
            result = eval(function_argument[f])
            function_dict[f] = result
            result_list.append(result)
        result_dict[c] = function_dict
    _a = None
    cfg.logger.log.debug('end')
    return [None, result_dict]


# calculate spectral signature
def spectral_signature(*argv):
    cfg.logger.log.debug('start')
    array_function_placeholder = argv[1][0]
    nodata_mask = argv[2]
    # vector path
    function_argument = argv[7]
    # reference path
    function_variable = argv[8]
    temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
    raster_vector.vector_to_raster(
        vector_path=function_argument, burn_values=1, output_path=temp,
        reference_raster_path=function_variable, extent=True
    )
    _a = raster_vector.read_raster(temp)
    _a[::, ::][nodata_mask == 1] = np.nan
    array_roi = array_function_placeholder[_a == 1]
    mean = np.nanmean(array_roi)
    std = np.nanstd(array_roi)
    count = np.count_nonzero(_a == 1)
    cfg.logger.log.debug('end')
    return [None, [mean, std, count]]


# get raster band values for scatter plot
def get_values_for_scatter_plot(*argv):
    cfg.logger.log.debug('start')
    array_function_placeholder = argv[1][0]
    nodata_mask = argv[2]
    # vector path
    function_argument = argv[7]
    # reference path
    function_variable = argv[8]
    temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
    raster_vector.vector_to_raster(
        vector_path=function_argument, burn_values=1, output_path=temp,
        reference_raster_path=function_variable, extent=True
    )
    _a = raster_vector.read_raster(temp)
    _a[::, ::][nodata_mask == 1] = np.nan
    array_roi = array_function_placeholder[_a == 1]
    cfg.logger.log.debug('end')
    return [None, array_roi.ravel()]


# calculate region growing from seed value
def region_growing(*argv):
    cfg.logger.log.debug('start')
    array_function_placeholder = argv[1][0]
    nodata_mask = argv[2]
    # roi parameters
    function_variable = argv[8]
    array_roi = array_function_placeholder
    array_roi[::, ::][nodata_mask == 1] = np.nan
    seed_x = function_variable[0]
    seed_y = function_variable[1]
    max_spectral_distance = function_variable[2]
    minimum_size = function_variable[3]
    seed_array = np.zeros(array_roi.shape)
    seed_value = float(array_roi[seed_y, seed_x])
    cfg.logger.log.debug('array_roi.shape: %s; seed_value: %s'
                         % (str(array_roi.shape), str(seed_value)))
    # if nodata
    if np.sum(np.isnan(seed_value)) > 0:
        return seed_array
    seed_array.fill(seed_value)
    difference_array = abs(array_roi - seed_array)
    # calculate minimum difference
    unique_difference_array = np.unique(difference_array)
    unique_difference_distance = unique_difference_array[
        unique_difference_array > float(max_spectral_distance)]
    unique_difference_array = np.insert(
        unique_difference_distance, 0, float(max_spectral_distance)
    )
    region = None
    region_seed_value = None
    region_value_mask = seed_array
    for i in unique_difference_array:
        region_label, num_features = label(difference_array <= i)
        # value of ROI seed
        # noinspection PyUnresolvedReferences
        region_seed_value = region_label[seed_y, seed_x]
        region_value_mask = (region_label == region_seed_value)
        if (region_seed_value != 0
                and np.count_nonzero(region_value_mask) >= minimum_size):
            region = np.copy(region_value_mask)
            break
    if region is None and region_seed_value != 0:
        region = np.copy(region_value_mask)
    cfg.logger.log.debug('end')
    return [None, region]


# get band arrays
def get_band_arrays(
        process, progress_queue, argument_list, logger, temp
):
    cfg.logger = logger
    cfg.temp = temp
    results = []
    errors = False
    for n, d in enumerate(argument_list, start=1):
        try:
            # iterate signatures
            array_dict = {}
            # optional calc data type, if None use input data type
            calc_data_type = d['calc_data_type']
            if 'field_name' in d:
                field_name = d['field_name']
            else:
                field_name = str(cfg.uid_field_name)
            for s in d['signature_id_list']:
                r_arr_v_arr_list = []
                nd_bool_list = []
                # iterate rasters
                for v in d['virtual_path_list']:
                    extract = raster_vector.extract_vector_to_raster(
                        vector_path=d['roi_path'],
                        reference_raster_path=v, calc_data_type=calc_data_type,
                        available_ram=d['available_ram'],
                        attribute_filter="%s = '%s'" % (field_name, str(s))
                    )
                    if extract is False:
                        return [None, 'error extract', str(process)]
                    # get raster array and signature raster array
                    if extract[0] is not None:
                        r_arr_v_arr_list.append([extract[0], extract[1]])
                        # get raster nodata array
                        nd_bool_list.append(extract[2])
                # create nodata array for all rasters
                nd_array = None
                for nd_bool in nd_bool_list:
                    if nd_array is None:
                        nd_array = nd_bool
                    else:
                        nd_array = nd_array * nd_bool
                # create array list
                array_list = []
                for i in r_arr_v_arr_list:
                    _a = i[0][nd_array]
                    _b = i[1][nd_array]
                    array_list.append(_a[_b == 1])
                # calculate numpy functions
                if 'numpy_functions' in d:
                    result_dict = {}
                    for f in d['numpy_functions']:
                        for _ in array_list:
                            result_dict[f] = eval(d['numpy_functions'][f])
                    array_dict[s] = result_dict
                else:
                    array_dict[s] = array_list
            results.append(array_dict)
            if progress_queue is not None and progress_queue.empty():
                progress_queue.put([n, len(argument_list)], False)
        except Exception as err:
            errors = str(err)
    return [results, errors, str(process)]


# fit
def fit_classifier(*argv):
    p, temp, ram, log_process, cfg.logger = argv[0]
    classifier_list = argv[1]
    arg_dict_list = argv[2]
    result = []
    for i, value in enumerate(arg_dict_list):
        classifier = classifier_list[i]
        if log_process and log_process is not None:
            with open('%s/scikit' % temp.dir, 'w') as f:
                with redirect_stdout(f):
                    classifier.fit(**value)
        else:
            # set verbose
            try:
                classifier.verbose = 0
            except Exception as err:
                str(err)
            classifier.fit(**arg_dict_list[i])
        classifier.verbose = 0
        result.append(classifier)
    return [result, False, None]


# score
def score_classifier(*argv):
    p, temp, ram, log_process, cfg.logger = argv[0]
    classifier_list = argv[1]
    arg_dict_list = argv[2]
    result = []
    for i, value in enumerate(arg_dict_list):
        classifier = classifier_list[i]
        if log_process and log_process is not None:
            with open('%s/scikit' % temp.dir, 'w') as f:
                with redirect_stdout(f):
                    score = classifier.score(**value)
        else:
            # set verbose
            try:
                classifier.verbose = 0
            except Exception as err:
                str(err)
            score = classifier.score(**arg_dict_list[i])
        result.append(score)
    return [result, False, None]


# score classifier with stratified k fold
def score_classifier_stratified(
        process, progress_queue, argument_list, logger, temp
):
    cfg.logger = logger
    cfg.temp = temp
    results = []
    errors = False
    for n, d in enumerate(argument_list, start=1):
        try:
            # calculate score by cross-validation with stratification
            scores = cross_val_score(
                d['classifier'], d['x_matrix'], d['y'],
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            )
            score = np.array(scores)
            results.append([d['classifier'], score.mean(), score.std()])
            if progress_queue is not None and progress_queue.empty():
                progress_queue.put([n, len(argument_list)], False)
        except Exception as err:
            errors = str(err)
    return [results, errors, str(process)]


# clip_raster
def clip_raster(
        process, progress_queue, argument_list, logger, temp
):
    cfg.logger = logger
    cfg.temp = temp
    results = []
    errors = False
    for n, d in enumerate(argument_list, start=1):
        gdal_path = d['gdal_path']
        cfg.logger.log.debug('start')
        if gdal_path is not None:
            for path in gdal_path.split(';'):
                try:
                    os.add_dll_directory(path)
                    cfg.gdal_path = path
                except Exception as err:
                    str(err)
        from osgeo import gdal, ogr, osr
        # GDAL config
        try:
            memory = str(int(d['available_ram']) * 1000000)
            gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
            gdal.SetConfigOption('GDAL_CACHEMAX', str(memory))
            gdal.SetConfigOption('VSI_CACHE', 'FALSE')
            gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
            gdal.DontUseExceptions()
        except Exception as err:
            str(err)
        try:
            _r_d = gdal.Open(d['input_raster'], gdal.GA_ReadOnly)
            input_band = _r_d.GetRasterBand(1)
            no_data = input_band.GetNoDataValue()
            sr = osr.SpatialReference()
            sr.ImportFromWkt(_r_d.GetProjectionRef())
            if d['extent_list'] is not None:
                # copy raster band
                op = ' -co BIGTIFF=YES -co COMPRESS=%s' % d['compress_format']
                op += ' -of %s' % 'GTiff'
                if no_data is not None:
                    op += ' -srcnodata %s -dstnodata %s' % (no_data, no_data)
                to = gdal.WarpOptions(gdal.ParseCommandLine(op))
            else:
                cutline = d['vector_path']
                _vector = ogr.Open(cutline)
                _v_layer = _vector.GetLayer()
                extent = _v_layer.GetExtent()
                v_sr = _v_layer.GetSpatialRef()
                if sr.IsSame(v_sr) != 1:
                    c_t = osr.CoordinateTransformation(v_sr, sr)
                    driver = ogr.GetDriverByName('GPKG')
                    # create temp vector
                    cutline = cfg.temp.temporary_file_path(
                        name_suffix=cfg.gpkg_suffix
                    )
                    _data_source = driver.CreateDataSource(cutline)
                    spatial_reference = osr.SpatialReference()
                    spatial_reference.ImportFromWkt(_r_d.GetProjectionRef())
                    temp_layer = _data_source.CreateLayer(
                        'temp', spatial_reference, ogr.wkbPolygon
                    )
                    _v_layerDefn = _v_layer.GetLayerDefn()
                    for i in range(0, _v_layerDefn.GetFieldCount()):
                        field_def = _v_layerDefn.GetFieldDefn(i)
                        temp_layer.CreateField(field_def)
                    temp_layer_def = temp_layer.GetLayerDefn()
                    input_feature = _v_layer.GetNextFeature()
                    while input_feature:
                        geom = input_feature.GetGeometryRef()
                        geom.Transform(c_t)
                        _out_feature = ogr.Feature(temp_layer_def)
                        _out_feature.SetGeometry(geom)
                        for i in range(0, temp_layer_def.GetFieldCount()):
                            _out_feature.SetField(
                                temp_layer_def.GetFieldDefn(i).GetNameRef(),
                                input_feature.GetField(i)
                            )
                        temp_layer.CreateFeature(_out_feature)
                        _out_feature = None
                        input_feature = _v_layer.GetNextFeature()
                _vector = None
                crop = True
                op = ' -co BIGTIFF=YES -co COMPRESS=%s' % d['compress_format']
                # op += ' -wo CUTLINE_ALL_TOUCHED = TRUE'
                to = gdal.WarpOptions(
                    gdal.ParseCommandLine(op),
                    format='GTiff', outputBounds=extent,
                    dstSRS=sr.ExportToWkt(), cutlineDSName=cutline,
                    cropToCutline=crop, cutlineWhere=d['where'],
                    srcNodata=no_data, dstNodata=no_data
                )
            gdal.Warp(d['output'], d['input_raster'], options=to)
            _r_d = None
            results.append([d['output']])
            if progress_queue is not None and progress_queue.empty():
                progress_queue.put([n, len(argument_list)], False)
        except Exception as err:
            errors = str(err)
    return [results, errors, str(process)]


# vector to raster
def vector_to_raster_iter(
        process, progress_queue, argument_list, logger, temp
):
    cfg.logger = logger
    cfg.temp = temp
    results = []
    errors = False
    for n, d in enumerate(argument_list, start=1):
        gdal_path = d['gdal_path']
        cfg.logger.log.debug('start')
        if gdal_path is not None:
            for path in gdal_path.split(';'):
                try:
                    os.add_dll_directory(path)
                    cfg.gdal_path = path
                except Exception as err:
                    str(err)
        from osgeo import gdal, ogr, osr
        # GDAL config
        try:
            memory = str(int(d['available_ram']) * 1000000)
            gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
            gdal.SetConfigOption('GDAL_CACHEMAX', str(memory))
            gdal.SetConfigOption('VSI_CACHE', 'FALSE')
            gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
            gdal.DontUseExceptions()
        except Exception as err:
            str(err)
        try:
            feature = d['feature']
            vector_crs = d['vector_crs']
            i_layer_sr = osr.SpatialReference()
            i_layer_sr.ImportFromWkt(vector_crs)
            field_name = d['field_name']
            reference_raster_path = d['reference_raster_path']
            background_value = d['background_value']
            # targe pixel size
            x_y_size = d['x_y_size']
            # buffer to increase the grid size to match reference grid size
            buffer_size = d['buffer_size']
            minimum_extent = d['minimum_extent']
            output_path = d['output']
            compress = d['compress']
            compress_format = d['compress_format']
            if background_value is None:
                background_value = 0
            (
                gt, reference_crs, unit, xy_count, nd, number_of_bands,
                block_size, scale_offset, data_type
            ) = raster_vector.raster_info(reference_raster_path)
            if x_y_size is not None:
                x_size = x_y_size[0]
                y_size = x_y_size[1]
            else:
                x_size = gt[1]
                y_size = abs(gt[5])
            cfg.logger.log.debug('x_size, y_size: %s,%s' % (x_size, y_size))
            orig_x = gt[0]
            orig_y = gt[3]
            # number of x pixels
            grid_columns = int(round(xy_count[0] * gt[1] / x_size))
            # number of y pixels
            grid_rows = int(round(xy_count[1] * abs(gt[5]) / y_size))
            cfg.logger.log.debug(
                'grid_columns, grid_rows: %s,%s' % (grid_columns, grid_rows)
            )
            # check crs
            same_crs = raster_vector.compare_crs(reference_crs, vector_crs)
            cfg.logger.log.debug('same_crs: %s' % str(same_crs))
            if not same_crs:
                cfg.logger.log.error('different crs')
                logger = cfg.logger.stream.getvalue()
                return None, 'different crs', logger
            # create memory layer
            memory_driver = ogr.GetDriverByName('MEM')
            # for backward compatibility
            if memory_driver is None:
                memory_driver = ogr.GetDriverByName('memory')
            memory_source = memory_driver.CreateDataSource('in_memory')
            memory_layer = memory_source.CreateLayer(
                'temp', i_layer_sr, geom_type=ogr.wkbMultiPolygon
            )
            field_definitions = feature[0]
            geometry = ogr.CreateGeometryFromWkt(feature[1])
            attributes = feature[2]
            for f_d in field_definitions:
                field_defn = ogr.FieldDefn(f_d['name'], f_d['type'])
                # noinspection PyArgumentList
                field_defn.SetWidth(f_d['width'])
                # noinspection PyArgumentList
                field_defn.SetPrecision(f_d['precision'])
                memory_layer.CreateField(field_defn)
            o_layer_definition = memory_layer.GetLayerDefn()
            # field_value count
            o_field_count = o_layer_definition.GetFieldCount()
            o_feature = ogr.Feature(o_layer_definition)
            o_feature.SetGeometry(geometry)
            for i in range(o_field_count):
                field_n = o_layer_definition.GetFieldDefn(i).GetNameRef()
                field_value = attributes[i]
                o_feature.SetField(field_n, field_value)
            memory_layer.CreateFeature(o_feature)
            if minimum_extent is True:
                # calculate minimum extent
                min_x, max_x, min_y, max_y = memory_layer.GetExtent()
                if buffer_size is not None:
                    min_x -= buffer_size
                    max_x += buffer_size
                    min_y -= buffer_size
                    max_y += buffer_size
                orig_x = gt[0] + gt[1] * int((min_x - gt[0]) / gt[1])
                orig_y = gt[3] + abs(gt[5]) * int(
                    round((max_y - gt[3]) / abs(gt[5])))
                cfg.logger.log.debug('orig_x, orig_y: %s,%s'
                                     % (orig_x, orig_y))
                grid_columns = abs(int(round((max_x - min_x) / x_size)))
                grid_rows = abs(int(round((max_y - min_y) / y_size)))
            cfg.logger.log.debug(
                'grid_columns, grid_rows: %s,%s' % (grid_columns, grid_rows)
            )
            if grid_columns > 0 and grid_rows > 0:
                # create raster _grid
                _grid = memory_driver.Create(
                    '', grid_columns, grid_rows, 1, gdal.GDT_Float32
                )
                if _grid is None:
                    _grid = memory_driver.Create(
                        '', grid_columns, grid_rows, 1, gdal.GDT_Int16
                    )
                if _grid is None:
                    cfg.logger.log.error('error output raster')
                    logger = cfg.logger.stream.getvalue()
                    return None, 'error grid', logger
                try:
                    _band = _grid.GetRasterBand(1)
                except Exception as err:
                    cfg.logger.log.error(err)
                    logger = cfg.logger.stream.getvalue()
                    return None, str(err), logger
                # set raster projection from reference
                _grid.SetGeoTransform([orig_x, x_size, 0, orig_y, 0, -y_size])
                _grid.SetProjection(reference_crs)
                _band.Fill(background_value)
                _band.FlushCache()
                _band = None
                gdal.RasterizeLayer(
                    _grid, [1], memory_layer, options=[
                        'ATTRIBUTE=%s' % str(field_name)]
                )
                src_nodata = d['src_nodata']
                dst_nodata = d['dst_nodata']
                resample_method = d['resample_method']
                # alignment raster extent and pixel size
                left_align = gt[0]
                top_align = gt[3]
                p_x_align = gt[1]
                p_y_align = abs(gt[5])
                right_align = gt[0] + gt[1] * xy_count[0]
                bottom_align = gt[3] + gt[5] * xy_count[1]
                # memory raster
                right_x = orig_x + grid_columns * x_size
                bottom_y = orig_y - grid_rows * y_size
                # minimum extent
                if orig_x < left_align:
                    left_r = left_align - int(
                        2 + (left_align - orig_x) / p_x_align
                    ) * p_x_align
                else:
                    left_r = left_align + int(
                        (orig_x - left_align) / p_x_align - 2
                    ) * p_x_align
                if right_x > right_align:
                    right_r = right_align + int(
                        2 + (right_x - right_align) / p_x_align
                    ) * p_x_align
                else:
                    right_r = right_align - int(
                        (right_align - right_x) / p_x_align - 2
                    ) * p_x_align
                if orig_y > top_align:
                    top_r = top_align + int(
                        2 + (orig_y - top_align) / p_y_align
                    ) * p_y_align
                else:
                    top_r = top_align - int(
                        (top_align - orig_y) / p_y_align - 2
                    ) * p_y_align
                if bottom_y > bottom_align:
                    bottom_r = bottom_align + int(
                        (bottom_y - bottom_align) / p_y_align - 2
                    ) * p_y_align
                else:
                    bottom_r = bottom_align - int(
                        2 + (bottom_align - bottom_y) / p_y_align
                    ) * p_y_align

                op = ' -r %s' % resample_method
                if compress is None:
                    if cfg.raster_compression:
                        op += ' -co COMPRESS=%s' % compress_format
                elif compress:
                    op += ' -co COMPRESS=%s' % compress_format
                if src_nodata is not None:
                    op += ' -srcnodata %s' % str(src_nodata)
                if dst_nodata is not None:
                    op += ' -dstnodata %s' % str(dst_nodata)
                additional_params = '-tr %s %s -te %s %s %s %s' % (
                    str(p_x_align), str(p_y_align), str(left_r), str(bottom_r),
                    str(right_r), str(top_r))
                op = ' %s %s' % (additional_params, op)
                to = gdal.WarpOptions(gdal.ParseCommandLine(op))
                gdal.PushErrorHandler('CPLQuietErrorHandler')
                gdal.Warp(output_path, _grid, options=to)
                gdal.PushErrorHandler()
                if progress_queue is not None and progress_queue.empty():
                    progress_queue.put([n, len(argument_list)], False)
                results.append([output_path])
            if progress_queue is not None and progress_queue.empty():
                progress_queue.put([n, len(argument_list)], False)

        except Exception as err:
            errors = str(err)
    return [results, errors, str(process)]


# warped to virtual raster
def create_warped_vrt_iter(
        process, progress_queue, argument_list, logger, temp
):
    cfg.logger = logger
    cfg.temp = temp
    results = []
    errors = False
    for n, d in enumerate(argument_list, start=1):
        gdal_path = d['gdal_path']
        cfg.logger.log.debug('start')
        if gdal_path is not None:
            for path in gdal_path.split(';'):
                try:
                    os.add_dll_directory(path)
                    cfg.gdal_path = path
                except Exception as err:
                    str(err)
        from osgeo import gdal
        # GDAL config
        try:
            memory = str(int(d['available_ram']) * 1000000)
            gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
            gdal.SetConfigOption('GDAL_CACHEMAX', str(memory))
            gdal.SetConfigOption('VSI_CACHE', 'FALSE')
            gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
            gdal.DontUseExceptions()
        except Exception as err:
            str(err)
        try:
            raster_path = d['raster_path']
            output_path = d['output_path']
            align_raster_path = d['align_raster_path']
            src_nodata = d['src_nodata']
            dst_nodata = d['dst_nodata']
            same_extent = d['same_extent']
            resample_method = d['resample_method']
            # align raster extent and pixel size
            try:
                info = raster_vector.image_geotransformation(
                    align_raster_path)
                left_align = info['left']
                top_align = info['top']
                right_align = info['right']
                bottom_align = info['bottom']
                p_x_align = info['pixel_size_x']
                p_y_align = info['pixel_size_y']
                output_wkt = info['projection']
                # check projections
                align_sys_ref = raster_vector.get_spatial_reference(
                    output_wkt)
            except Exception as err:
                cfg.logger.log.error(str(err))
                return False
            # input_path raster extent and pixel size
            try:
                info = raster_vector.image_geotransformation(raster_path)
                left_input = info['left']
                top_input = info['top']
                right_input = info['right']
                bottom_input = info['bottom']
                proj_input = info['projection']
                input_sys_ref = raster_vector.get_spatial_reference(
                    proj_input)
                left_projected, top_projected = \
                    raster_vector.project_point_coordinates(
                        left_input, top_input, input_sys_ref, align_sys_ref
                    )
                right_projected, bottom_projected = \
                    raster_vector.project_point_coordinates(
                        right_input, bottom_input, input_sys_ref,
                        align_sys_ref
                    )
            # Error latitude or longitude exceeded limits
            except Exception as err:
                cfg.logger.log.error(str(err))
                return False
            if not same_extent:
                # minimum extent
                if left_projected < left_align:
                    left_r = left_align - int(
                        2 + (left_align - left_projected) / p_x_align
                    ) * p_x_align
                else:
                    left_r = left_align + int(
                        (left_projected - left_align) / p_x_align - 2
                    ) * p_x_align
                if right_projected > right_align:
                    right_r = right_align + int(
                        2 + (right_projected - right_align) / p_x_align
                    ) * p_x_align
                else:
                    right_r = right_align - int(
                        (right_align - right_projected) / p_x_align - 2
                    ) * p_x_align
                if top_projected > top_align:
                    top_r = top_align + int(
                        2 + (top_projected - top_align) / p_y_align
                    ) * p_y_align
                else:
                    top_r = top_align - int(
                        (top_align - top_projected) / p_y_align - 2
                    ) * p_y_align
                if bottom_projected > bottom_align:
                    bottom_r = bottom_align + int(
                        (bottom_projected - bottom_align) / p_y_align - 2
                    ) * p_y_align
                else:
                    bottom_r = bottom_align - int(
                        2 + (bottom_align - bottom_projected) / p_y_align
                    ) * p_y_align
            else:
                left_r = left_align
                right_r = right_align
                top_r = top_align
                bottom_r = bottom_align
            additional_params = '-tr %s %s -te %s %s %s %s' % (
                str(p_x_align), str(p_y_align), str(left_r), str(bottom_r),
                str(right_r), str(top_r))
            op = ' -r %s' % resample_method
            if src_nodata is not None:
                op += ' -srcnodata %s' % str(src_nodata)
            if dst_nodata is not None:
                op += ' -dstnodata %s' % str(dst_nodata)
            op += ' -of VRT'
            op = ' %s %s' % (additional_params, op)
            to = gdal.WarpOptions(gdal.ParseCommandLine(op))
            gdal.Warp(output_path, raster_path, options=to)
            if progress_queue is not None and progress_queue.empty():
                progress_queue.put([n, len(argument_list)], False)
            results.append([d['output_path']])
        except Exception as err:
            errors = str(err)
    return [results, errors, str(process)]


# virtual raster
def create_vrt_iter(
        process, progress_queue, argument_list, logger, temp
):
    cfg.logger = logger
    cfg.temp = temp
    results = []
    errors = False
    for n, d in enumerate(argument_list, start=1):
        gdal_path = d['gdal_path']
        cfg.logger.log.debug('start')
        if gdal_path is not None:
            for path in gdal_path.split(';'):
                try:
                    os.add_dll_directory(path)
                    cfg.gdal_path = path
                except Exception as err:
                    str(err)
        from osgeo import gdal
        # GDAL config
        try:
            memory = str(int(d['available_ram']) * 1000000)
            gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
            gdal.SetConfigOption('GDAL_CACHEMAX', str(memory))
            gdal.SetConfigOption('VSI_CACHE', 'FALSE')
            gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
            gdal.DontUseExceptions()
        except Exception as err:
            str(err)
        try:
            input_raster_list = d['input_raster_list']
            output = d['output']
            band_number_list = d['band_number_list']
            src_nodata = d['src_nodata']
            dst_nodata = d['dst_nodata']
            if dst_nodata is None:
                dst_nodata = False
            relative_to_vrt = d['relative_to_vrt']
            if relative_to_vrt is None:
                relative_to_vrt = 0
            data_type = d['data_type']
            box_coordinate_list = d['box_coordinate_list']
            override_box_coordinate_list = d['override_box_coordinate_list']
            if override_box_coordinate_list is None:
                override_box_coordinate_list = False
            pixel_size = d['pixel_size']
            grid_reference = d['grid_reference']
            scale_offset_list = d['scale_offset_list']
            resampling = d['resampling']
            virtual_out = raster_vector.create_virtual_raster_2_mosaic(
                input_raster_list=input_raster_list, output=output,
                band_number_list=band_number_list, src_nodata=src_nodata,
                dst_nodata=dst_nodata, relative_to_vrt=relative_to_vrt,
                data_type=data_type, box_coordinate_list=box_coordinate_list,
                override_box_coordinate_list=override_box_coordinate_list,
                pixel_size=pixel_size, grid_reference=grid_reference,
                scale_offset_list=scale_offset_list, resampling=resampling
            )
            if progress_queue is not None and progress_queue.empty():
                progress_queue.put([n, len(argument_list)], False)
            results.append([virtual_out])
        except Exception as err:
            errors = str(err)
    return [results, errors, str(process)]


# reclassify raster
def raster_reclass(
        process, progress_queue, argument_list, logger, temp
):
    cfg.logger = logger
    cfg.temp = temp
    results = []
    errors = False
    for n, d in enumerate(argument_list, start=1):
        gdal_path = d['gdal_path']
        cfg.logger.log.debug('start')
        if gdal_path is not None:
            for path in gdal_path.split(';'):
                try:
                    os.add_dll_directory(path)
                    cfg.gdal_path = path
                except Exception as err:
                    str(err)
        from osgeo import gdal
        # GDAL config
        try:
            memory = str(int(d['available_ram']) * 1000000)
            gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
            gdal.SetConfigOption('GDAL_CACHEMAX', str(memory))
            gdal.SetConfigOption('VSI_CACHE', 'FALSE')
            gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
            gdal.DontUseExceptions()
        except Exception as err:
            str(err)
        try:
            _o = raster_vector.read_raster(d['input_raster'])
            if d['reclass_table'] is not None:
                old = d['reclass_table'][:, 0]
                new = d['reclass_table'][:, 1]
                # create empty reclass array of length equal to maximum value
                reclass = np.zeros(
                    max(
                        old.astype(int).max(), _o.astype(int).max()
                    ) + 1
                ) * np.nan
                # fill array with new value at index corresponding to old value
                reclass[old.astype(int)] = new
                # perform reclassification
                _o = reclass[_o.astype(int)]
            # write raster
            raster_vector.create_raster_from_reference(
                d['input_raster'], 1, [d['output']],
                nodata_value=0, driver='GTiff', gdal_format='UInt32',
                compress=True, compress_format='LZW'
            )
            section_raster = raster_vector.write_raster(d['output'], 0, 0, _o)
            _o = None
            cfg.logger.log.debug('section_raster: %s' % str(section_raster))
            results.append([d['output']])
            if progress_queue is not None and progress_queue.empty():
                progress_queue.put([n, len(argument_list)], False)
        except Exception as err:
            errors = str(err)
    return [results, errors, str(process)]


# edit raster
# noinspection PyShadowingBuiltins
def edit_raster(
        process, progress_queue, argument_list, logger, temp
):
    cfg.logger = logger
    cfg.temp = temp
    results = []
    errors = False
    for n, d in enumerate(argument_list, start=1):
        gdal_path = d['gdal_path']
        cfg.logger.log.debug('start')
        if gdal_path is not None:
            for path in gdal_path.split(';'):
                try:
                    os.add_dll_directory(path)
                    cfg.gdal_path = path
                except Exception as err:
                    str(err)
        from osgeo import gdal
        # GDAL config
        try:
            memory = str(int(d['available_ram']) * 1000000)
            gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
            gdal.SetConfigOption('GDAL_CACHEMAX', str(memory))
            gdal.SetConfigOption('VSI_CACHE', 'FALSE')
            gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
            gdal.DontUseExceptions()
        except Exception as err:
            str(err)
        try:
            raster = d['old_array']
            if raster is not None:
                _r_column_start = d['column_start']
                _r_row_start = d['row_start']
                # write raster
                raster_vector.write_raster(
                    d['input_raster'], _r_column_start, _r_row_start, raster
                )
                results.append(None)
                results.append(None)
                results.append(None)
            else:
                constant_value = d['constant_value']
                expression = d['expression']
                _r_d = gdal.Open(d['input_raster'], gdal.GA_Update)
                _v_d = gdal.Open(d['vector_raster'], gdal.GA_ReadOnly)
                # get pixel size and top left
                _r_d_gt = _r_d.GetGeoTransform()
                _r_left = _r_d_gt[0]
                _r_top = _r_d_gt[3]
                r_p_x_size = _r_d_gt[1]
                r_p_y_size = _r_d_gt[5]
                _v_d_gt = _v_d.GetGeoTransform()
                _v_left = _v_d_gt[0]
                _v_top = _v_d_gt[3]
                # get pixel columns
                _r_x_size = _r_d.RasterXSize
                _v_x_size = _v_d.RasterXSize
                # get pixel rows
                _r_y_size = _r_d.RasterYSize
                _v_y_size = _v_d.RasterYSize
                if _v_left < _r_left:
                    orig_x = _r_left
                else:
                    orig_x = _v_left
                if _v_top > _r_top:
                    orig_y = _r_top
                else:
                    orig_y = _v_top
                _r_column_start = abs(int((_r_left - orig_x) / r_p_x_size))
                _r_row_start = abs(int((_r_top - orig_y) / r_p_y_size))
                _v_column_start = abs(int((_v_left - orig_x) / r_p_x_size))
                _v_row_start = abs(int((_v_top - orig_y) / r_p_y_size))
                columns = _v_x_size - _v_column_start
                rows = _v_y_size - _v_row_start
                if columns < 1 or rows < 1:
                    raise 'not enough pixels'
                if _r_column_start + columns > _r_x_size:
                    r_column = _r_x_size - _r_column_start
                else:
                    r_column = columns
                if r_column < 0:
                    raise 'outside raster'
                if _r_row_start + rows > _r_y_size:
                    r_rows = _r_y_size - _r_row_start
                else:
                    r_rows = rows
                if r_rows < 0:
                    raise 'outside raster'
                # get raster band
                _r_band = _r_d.GetRasterBand(1)
                try:
                    _r_offset = _r_band.GetOffset()
                    _r_scale = _r_band.GetScale()
                    if _r_offset is None:
                        _r_offset = 0
                    if _r_scale is None:
                        _r_scale = 1
                except Exception as err:
                    str(err)
                    _r_offset = 0
                    _r_scale = 1
                # set variable name as cfg.variable_raster_name
                o_raster = _r_band.ReadAsArray(
                    _r_column_start, _r_row_start, r_column, r_rows
                )
                raster = o_raster * _r_scale + _r_offset
                # get vector raster band
                _v_band = _v_d.GetRasterBand(1)
                try:
                    _v_offset = _v_band.GetOffset()
                    _v_scale = _v_band.GetScale()
                    if _v_offset is None:
                        _v_offset = 0
                    if _v_scale is None:
                        _v_scale = 1
                except Exception as err:
                    str(err)
                    _v_offset = 0
                    _v_scale = 1
                _v_band = _v_band.ReadAsArray(
                    _v_column_start, _v_row_start, r_column, r_rows
                )
                # set variable name as cfg.variable_vector_name
                vector = _v_band * _v_scale + _v_offset
                # expression
                if expression is not None:
                    # expose numpy functions
                    log = np.log
                    _log = log
                    log10 = np.log10
                    _log10 = log10
                    sqrt = np.sqrt
                    _sqrt = sqrt
                    cos = np.cos
                    _cos = cos
                    arccos = np.arccos
                    _arccos = arccos
                    sin = np.sin
                    _sin = sin
                    arcsin = np.arcsin
                    _arcsin = arcsin
                    tan = np.tan
                    _tan = tan
                    arctan = np.arctan
                    _arctan = arctan
                    exp = np.exp
                    _exp = exp
                    min = np.nanmin
                    _min = min
                    max = np.nanmax
                    _max = max
                    sum = np.nansum
                    _sum = sum
                    percentile = np.nanpercentile
                    _percentile = percentile
                    median = np.nanmedian
                    _median = median
                    mean = np.nanmean
                    _mean = mean
                    std = np.nanstd
                    _std = std
                    where = np.where
                    _where = where
                    nan = np.nan
                    _nan = nan
                    data_array = eval(expression)
                else:
                    data_array = np.where(
                        vector > 0, constant_value, raster
                    )
                _r_band = _v_band = None
                _r_d = _v_d = None
                # write raster
                raster_vector.write_raster(
                    d['input_raster'], _r_column_start, _r_row_start,
                    data_array / _r_scale - _r_offset
                )
                results.append(o_raster)
                results.append(_r_column_start)
                results.append(_r_row_start)
            if progress_queue is not None and progress_queue.empty():
                progress_queue.put([n, len(argument_list)], False)
        except Exception as err:
            errors = str(err)
    return [results, errors, str(process)]


# pixel value
def raster_point_values(
        process, progress_queue, argument_list, logger, temp
):
    cfg.logger = logger
    cfg.temp = temp
    results = []
    errors = False
    for n, d in enumerate(argument_list, start=1):
        try:
            pixel_value = raster_vector.get_pixel_value(
                point_coordinates=d['point_coordinate'],
                reference_raster_path=d['input_raster'],
                point_crs=None, available_ram=None
            )
            if pixel_value == d['output_no_data']:
                pixel_value = None
            results.append(pixel_value)
            if progress_queue is not None and progress_queue.empty():
                progress_queue.put([n, len(argument_list)], False)
        except Exception as err:
            errors = str(err)
    return [results, errors, str(process)]


# raster label
def raster_label_part(*argv):
    _array_function_placeholder = argv[1][0]
    nodata_mask = argv[2]
    function_argument = argv[7]
    gt = function_argument[0]
    orig_x = gt[0]
    x_size = gt[1]
    y_size = abs(gt[5])
    projection = function_argument[1]
    section = argv[16]
    # replace nodata with 0
    _array_function_placeholder[::, ::, 0][nodata_mask == 1] = 0
    # get array without boundary
    top = section.y_size_boundary_top
    cfg.logger.log.debug('section top: %s' % str(top))
    bottom = section.y_size_boundary_bottom
    if bottom > 0:
        bottom = -bottom
    else:
        bottom = _array_function_placeholder.shape[0]
    arr = _array_function_placeholder[::, ::, 0]
    cfg.logger.log.debug('arr.shape: %s' % str(arr.shape))
    region_label, num_features = label(arr)
    # get overlapping regions on boundary
    if section.y_size_boundary_top > 0:
        # get first two rows (first one belongs to the previous part)
        check_top = region_label[0, ::]
    else:
        check_top = None
    if section.y_size_boundary_bottom > 0:
        # get last two rows (last one belongs to the following part)
        check_bottom = region_label[-1, ::]
    else:
        check_bottom = None
    # create raster
    out_specific = cfg.temp.temporary_raster_path(
        name_suffix=str(section.y_min_no_boundary), name_prefix='lb_'
    )
    orig_y = gt[3] - section.y_min_no_boundary * y_size
    # remove boundary and count unique values
    region_label = region_label[top:bottom, ::]
    unique, counts = np.unique(region_label, return_counts=True)
    unique_counts = np.array(list(zip(unique[unique != 0],
                                      counts[unique != 0])))
    file_output = raster_vector.create_raster_from_grid(
        output_raster=out_specific, projection=projection,
        x_size=x_size, y_size=y_size, orig_x=orig_x, orig_y=orig_y,
        grid_columns=region_label.shape[1], grid_rows=region_label.shape[0],
        nodata_value=0, output_format='GTiff',
        gdal_format=cfg.uint32_dt, compress=True, compress_format='LZW'
    )
    section_raster = raster_vector.write_raster(file_output, 0, 0,
                                                region_label)
    cfg.logger.log.debug('section_raster: %s' % str(section_raster))
    label_dict = {'orig_y': orig_y, 'unique_counts': unique_counts,
                  'check_top': check_top, 'check_bottom': check_bottom,
                  'section_raster': section_raster}
    return [True, label_dict]
