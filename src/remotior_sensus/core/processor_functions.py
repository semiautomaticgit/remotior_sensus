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
except Exception as error:
    str(error)
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
    from remotior_sensus.util.pytorch_tools import train_pytorch_model
except Exception as error:
    str(error)


# band calculation
# noinspection PyShadowingBuiltins
def band_calculation(*argv):
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
    # array variable name as defined in cfg.array_function_placeholder
    _array_function_placeholder = argv[1]
    nodata_mask = argv[2]
    function_argument = argv[7]
    cfg.logger.log.debug('function_argument: %s' % str(function_argument))
    if (_array_function_placeholder.dtype == np.float32
            or _array_function_placeholder.dtype == np.float64):
        cfg.logger.log.debug(
            '_array_function_placeholder.shape: %s; '
            '_array_function_placeholder.dtype: %s; '
            '_array_function_placeholder.n_bytes: %s'
            % (
                str(_array_function_placeholder.shape),
                str(_array_function_placeholder.dtype),
                str(_array_function_placeholder.nbytes)
               )
        )
        _array_function_placeholder = ArrayLike(_array_function_placeholder)
    cfg.logger.log.debug(
        '_array_function_placeholder.shape: %s'
        % str(_array_function_placeholder.shape)
    )
    # perform operation
    try:
        _o = eval(function_argument)
    except Exception as err:
        cfg.logger.log.error(str(err))
        return False
    # if not array
    if not isinstance(_o, np.ndarray):
        cfg.logger.log.error('not array')
        return False
    # check nodata
    cfg.logger.log.debug(
        '_o.shape: %s; nodata_mask.shape: %s; _o.n_bytes: %s; _o.dtype: %s'
        % (
            str(_o.shape), str(nodata_mask.shape), str(_o.nbytes),
            str(_o.dtype)
        )
    )
    if nodata_mask is not None:
        np.copyto(
            _o, nodata_mask.reshape(_o.shape),
            where=nodata_mask[::, ::].reshape(_o.shape) != 0
        )
    return [_o, None]


# classification maximum likelihood
def classification_maximum_likelihood(*argv):
    scale = argv[0][0]
    offset = argv[0][1]
    output_no_data = argv[0][2]
    _array_function_placeholder = argv[1]
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
        cfg.logger.log.debug('signature: %s; values: %s' % (s, str(values)))
        try:
            cov_matrix = covariance_matrices[s]
            # natural logarithm of the determinant of covariance matrix
            (sign, log_det) = np.linalg.slogdet(cov_matrix)
            inverse_cov_matrix = np.linalg.inv(cov_matrix)
            d = _array_function_placeholder - values
            distance_array = - log_det - (
                    np.dot(d, inverse_cov_matrix) * d).sum(axis=2)
            if threshold:
                class_threshold = signatures_table[
                    signatures_table.signature_id == s].min_dist_thr
                p = class_threshold / 100
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
            else:
                maximum_array = np.maximum(distance_array, previous_array)
                classification_array[
                    maximum_array != previous_array] = class_id
                previous_array = maximum_array
            if len(output_signature_raster) > 0:
                distance_array[::, ::][
                    distance_array == cfg.nodata_val] = output_no_data
                distance_array[::, ::][
                    nodata_mask == output_no_data] = output_no_data
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
            nodata_mask == output_no_data] = output_no_data
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
                nodata_mask == output_no_data] = output_no_data
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
    _array_function_placeholder = argv[1]
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
        # euclidean distance
        distance_array = np.sqrt(
            ((_array_function_placeholder - values) ** 2).sum(axis=2)
        )
        if threshold:
            class_threshold = signatures_table[
                signatures_table.signature_id == s].min_dist_thr
            distance_array[::, ::][
                distance_array < class_threshold] = cfg.nodata_val_Int32
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
        if len(output_signature_raster) > 0:
            distance_array[::, ::][
                distance_array == cfg.nodata_val_Int32] = output_no_data
            distance_array[::, ::][
                nodata_mask == output_no_data] = output_no_data
            write_sig = raster_vector.write_raster(
                output_signature_raster[s], x - ro_x, y - ro_y, distance_array,
                output_no_data, scale, offset
            )
            cfg.logger.log.debug('write_sig: %s' % str(write_sig))
    # write classification
    classification_array[::, ::][
        classification_array == cfg.nodata_val_Int32] = output_no_data
    classification_array[::, ::][
        nodata_mask == output_no_data] = output_no_data
    write_class = raster_vector.write_raster(
        out_class, x - ro_x, y - ro_y, classification_array, output_no_data,
        scale, offset
    )
    cfg.logger.log.debug('write_class: %s' % str(write_class))
    # write the algorithm raster
    if out_alg is not None:
        previous_array[::, ::][
            classification_array == cfg.nodata_val_Int32] = output_no_data
        previous_array[::, ::][nodata_mask == output_no_data] = output_no_data
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
    _array_function_placeholder = argv[1]
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
        # spectral angle
        distance_array = np.arccos(
            (_array_function_placeholder * values).sum(axis=2) / np.sqrt(
                (_array_function_placeholder ** 2).sum(axis=2) * (
                        values ** 2).sum()
            )
        ) * 180 / np.pi
        if threshold:
            class_threshold = signatures_table[
                signatures_table.signature_id == s].min_dist_thr
            distance_array[::, ::][
                distance_array < class_threshold] = cfg.nodata_val_Int32
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
        if len(output_signature_raster) > 0:
            distance_array[::, ::][
                distance_array == cfg.nodata_val_Int32] = output_no_data
            distance_array[::, ::][
                nodata_mask == output_no_data] = output_no_data
            write_sig = raster_vector.write_raster(
                output_signature_raster[s], x - ro_x, y - ro_y, distance_array,
                output_no_data, scale, offset
            )
            cfg.logger.log.debug('write_sig: %s' % str(write_sig))
    # write classification
    classification_array[::, ::][
        classification_array == cfg.nodata_val_Int32] = output_no_data
    classification_array[::, ::][
        nodata_mask == output_no_data] = output_no_data
    write_class = raster_vector.write_raster(
        out_class, x - ro_x, y - ro_y, classification_array, output_no_data,
        scale, offset
    )
    cfg.logger.log.debug('write_class: %s' % str(write_class))
    # write the algorithm raster
    if out_alg is not None:
        previous_array[::, ::][
            classification_array == cfg.nodata_val_Int32] = output_no_data
        previous_array[::, ::][nodata_mask == output_no_data] = output_no_data
        write_alg = raster_vector.write_raster(
            out_alg, x - ro_x, y - ro_y, previous_array, output_no_data, scale,
            offset
        )
        cfg.logger.log.debug('write_alg: %s' % str(write_alg))
    cfg.logger.log.debug(
        'classification_array.shape: %s' % str(classification_array.shape)
    )
    return [True, out_class]


# classification through scikit-learn model
def classification_scikit(*argv):
    scale = argv[0][0]
    offset = argv[0][1]
    output_no_data = argv[0][2]
    _array_function_placeholder = argv[1]
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
        classification_array[::, ::][
            nodata_mask == output_no_data] = output_no_data
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
            for c in range(0, len(classes)):
                classification_array = np.where(
                    classification_argmax == c, classes[c],
                    classification_array
                )
            if threshold is not False:
                classification_array[::, ::][
                    prediction_proba_array < threshold] = output_no_data
            prediction_proba_array[::, ::][
                nodata_mask == output_no_data] = output_no_data
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
            classification_array[::, ::][
                nodata_mask == output_no_data] = output_no_data
        # write classification
        classification_array[::, ::][
            nodata_mask == output_no_data] = output_no_data
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
    _array_function_placeholder = argv[1]
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
        classification_array[::, ::][
            nodata_mask == output_no_data] = output_no_data
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
            prediction_proba_array[::, ::][
                nodata_mask == output_no_data] = output_no_data
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
            classification_array[::, ::][
                nodata_mask == output_no_data] = output_no_data
        # write classification
        classification_array[::, ::][
            nodata_mask == output_no_data] = output_no_data
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
    _array_function_placeholder = argv[1]
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
        np.copyto(
            _o, nodata_mask.reshape(_o.shape),
            where=nodata_mask[::, ::].reshape(_o.shape) != 0
        )
    return [_o, None]


# reclassify raster
def reclassify_raster(*argv):
    output_no_data = argv[0][2]
    raster_array_band = argv[1]
    nodata_mask = argv[2]
    # conditions
    function_argument = argv[7]
    # variable raster name
    function_variable = argv[8]
    cfg.logger.log.debug('start')
    _o = None
    replace_nodata = True
    try:
        old = function_argument.old_value
        new = function_argument.old_value
        _raster = np.nan_to_num(raster_array_band[:, :, 0])
        # if all integer values
        if np.all(_raster.astype(int) == _raster) and np.all(
                old.astype(int) == old
        ):
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
        else:
            # raise exception to try expressions
            raise Exception
    except Exception as err:
        str(err)
        _raster = None
        # raster array
        _o = np.copy(raster_array_band[:, :, 0])
        _x = raster_array_band[:, :, 0]
        for i in range(function_argument.shape[0]):
            cfg.logger.log.debug(str(function_argument[i]))
            # if reclassify from nodata to new value
            if 'nan' in function_argument[i][cfg.old_value]:
                try:
                    # replace nodata
                    _o[::, ::][nodata_mask == output_no_data] = int(
                        float(function_argument[i][cfg.new_value])
                    )
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
            _o[::, ::][nodata_mask == output_no_data] = output_no_data
        except Exception as err:
            str(err)
    cfg.logger.log.debug('end')
    return [_o, None]


# calculate bands covariance
def bands_covariance(*argv):
    cfg.logger.log.debug('start')
    output_no_data = argv[0][2]
    raster_array_band = argv[1]
    nodata_mask = argv[2]
    band_number = argv[7]
    band_dict = argv[8]
    covariance_dictionary = {}
    # iterate bands
    for _x in band_number:
        # calculate covariance SUM((x - Mean_x) * (y - Mean_y))
        for _y in band_number:
            # mask nodata
            if _x == _y:
                x = raster_array_band[::, ::, _x][
                    nodata_mask != output_no_data].ravel()
                y = x
            else:
                x = raster_array_band[::, ::, _x][
                    nodata_mask != output_no_data].ravel()
                y = raster_array_band[::, ::, _y][
                    nodata_mask != output_no_data].ravel()
            # covariance
            cov = ((x - band_dict['mean_%s' % _x]) * (
                    y - band_dict['mean_%s' % _y])).sum()
            covariance_dictionary['cov_%s-%s' % (_x, _y)] = cov
    cfg.logger.log.debug('end')
    return [None, covariance_dictionary]


# calculate raster pixel count
def raster_pixel_count(*argv):
    cfg.logger.log.debug('start')
    output_no_data = argv[0][2]
    raster_array_band = argv[1]
    nodata_mask = argv[2]
    band_number = argv[7]
    band = raster_array_band[nodata_mask != output_no_data].ravel()
    count = band.shape[0]
    band_sum = np.nansum(band)
    raster_dictionary = {
        'count_%s' % str(band_number): count,
        'sum_%s' % str(band_number): band_sum
    }
    cfg.logger.log.debug('end')
    return [None, raster_dictionary]


# calculate raster unique values with sum
def raster_unique_values_with_sum(*argv):
    cfg.logger.log.debug('start')
    output_no_data = argv[0][2]
    raster_array_band = argv[1]
    try:
        nodata_mask = argv[2]
        b = raster_array_band[nodata_mask != output_no_data]
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


# calculate raster unique values of combinations
def raster_unique_values(*argv):
    cfg.logger.log.debug('start')
    output_no_data = argv[0][2]
    raster_array_band = argv[1]
    nodata_mask = argv[2]
    # stack arrays
    try:
        arr = raster_array_band[:, :, 0][nodata_mask != output_no_data].ravel()
        for i in range(1, raster_array_band.shape[2]):
            arr = np.vstack(
                (arr, raster_array_band[:, :, i][
                    nodata_mask != output_no_data].ravel())
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
    raster_array_band = argv[1]
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
    o = np.array(a, copy=True)
    try:
        for v in function_variable_list:
            o[(core * val_dict['arr_%s' % str(v)]) > 0] = v
    except Exception as err:
        cfg.logger.log.error(str(err))
    o[::, ::][np.isnan(raster_array_band[:, :, 0])] = np.nan
    o[::, ::][nodata_mask == output_no_data] = np.nan
    cfg.logger.log.debug('end')
    return [o, None]


# calculate raster erosion
def raster_erosion(*argv):
    cfg.logger.log.debug('start')
    output_no_data = argv[0][2]
    raster_array_band = argv[1]
    nodata_mask = argv[2]
    structure = argv[7]
    function_variable_list = argv[8]
    a = np.array(raster_array_band[::, ::, 0], copy=True)
    np.copyto(a, cfg.nodata_val, where=np.isnan(raster_array_band[:, :, 0]))
    np.copyto(a, cfg.nodata_val, where=nodata_mask == output_no_data)
    # unique value list
    unique_val = np.unique(a)
    unique_val_list = list(unique_val.astype(int))
    try:
        unique_val_list.remove(cfg.nodata_val)
    except Exception as err:
        str(err)
    # iteration of erosion size
    for _s in range(function_variable_list[0]):
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
                max_sum_unique[sum_unique > max_sum_unique] = sum_unique[
                    sum_unique > max_sum_unique]
            # erode
            else:
                # erosion values
                erosion[((sum_structure - sum_unique) > 0.01) & (a == i)] = 1
        np.copyto(a, fill_value, where=erosion == 1)
    a[::, ::][a == cfg.nodata_val] = np.nan
    a[::, ::][np.isnan(raster_array_band[:, :, 0])] = np.nan
    a[::, ::][nodata_mask == output_no_data] = np.nan
    cfg.logger.log.debug('end')
    return [a, None]


# calculate raster resample
def raster_resample(*argv):
    cfg.logger.log.debug('start')
    output_no_data = argv[0][2]
    raster_array_band = argv[1]
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
    cfg.logger.log.debug('end')
    return [o, None]


# calculate raster neighbor
def raster_neighbor(*argv):
    cfg.logger.log.debug('start')
    output_no_data = argv[0][2]
    raster_array_band = argv[1]
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
    o[::, ::][o == cfg.nodata_val] = np.nan
    o[::, ::][np.isnan(raster_array_band[:, :, 0])] = np.nan
    o[::, ::][nodata_mask == output_no_data] = np.nan
    cfg.logger.log.debug('end')
    return [o, None]


# calculate cross rasters
def cross_rasters(*argv):
    cfg.logger.log.debug('start')
    array_function_placeholder = argv[1]
    nodata_mask = argv[2]
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
        np.copyto(o, nodata_mask, where=nodata_mask[::, ::] != 0)
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
    return [o, stats]


# calculate spectral signature
def spectral_signature(*argv):
    cfg.logger.log.debug('start')
    output_no_data = argv[0][2]
    array_function_placeholder = argv[1]
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
    _a[::, ::][nodata_mask == output_no_data] = np.nan
    array_roi = array_function_placeholder[_a == 1]
    mean = np.nanmean(array_roi)
    std = np.nanstd(array_roi)
    cfg.logger.log.debug('end')
    return [None, [mean, std]]


# get band arrays
def get_band_arrays(*argv):
    cfg.logger.log.debug('start')
    output_no_data = argv[0][2]
    array_function_placeholder = argv[1]
    nodata_mask = argv[2]
    # vector path
    function_argument = argv[7][0]
    spectral_signatures_table = argv[7][1]
    # reference path
    function_variable = argv[8]
    # iterate ROIs
    array_dictionary = {}
    for s in spectral_signatures_table.signature_id:
        vector = raster_vector.get_polygon_from_vector(
            vector_path=function_argument,
            attribute_filter="%s = '%s'" % (cfg.uid_field_name, s)
        )
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        raster_vector.vector_to_raster(
            vector_path=vector, burn_values=1, output_path=temp,
            reference_raster_path=function_variable, extent=True
        )
        _a = raster_vector.read_raster(temp)
        _a[::, ::][nodata_mask == output_no_data] = np.nan
        array_roi = array_function_placeholder[_a == 1]
        array_dictionary[s] = array_roi.flatten()
    cfg.logger.log.debug('end')
    return [None, array_dictionary]


# fit
def fit_classifier(*argv):
    p, temp, ram, log_process, cfg.logger = argv[0]
    classifier_list = argv[1]
    arg_dict_list = argv[2]
    result = []
    for i in range(0, len(arg_dict_list)):
        classifier = classifier_list[i]
        if log_process and log_process is not None:
            with open('%s/scikit' % temp.dir, 'w') as f:
                with redirect_stdout(f):
                    classifier.fit(**arg_dict_list[i])
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
    for i in range(0, len(arg_dict_list)):
        classifier = classifier_list[i]
        if log_process and log_process is not None:
            with open('%s/scikit' % temp.dir, 'w') as f:
                with redirect_stdout(f):
                    score = classifier.score(**arg_dict_list[i])
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
        process, progress_queue, argument_list, logger
):
    cfg.logger = logger
    results = []
    n = 0
    errors = False
    for d in argument_list:
        n += 1
        try:
            # calculate score by cross-validation with stratification
            scores = cross_val_score(
                d['classifier'], d['x_matrix'], d['y'],
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            )
            score = np.array(scores)
            results.append([d['classifier'], score.mean(), score.std()])
            if progress_queue is not None:
                progress_queue.put([n, len(argument_list)], False)
        except Exception as err:
            errors = str(err)
    return [results, errors, str(process)]


# clip_raster
def clip_raster(
        process, progress_queue, argument_list, logger
):
    cfg.logger = logger
    results = []
    n = 0
    errors = False
    for d in argument_list:
        n += 1
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
            gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
            gdal.SetConfigOption('GDAL_CACHEMAX', str(d['available_ram']))
            gdal.SetConfigOption('VSI_CACHE', 'FALSE')
            gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
        except Exception as err:
            str(err)
        try:
            _r_d = gdal.Open(d['input_raster'], gdal.GA_ReadOnly)
            sr = osr.SpatialReference()
            sr.ImportFromWkt(_r_d.GetProjectionRef())
            if d['extent_list'] is not None:
                # copy raster band
                op = ' -co BIGTIFF=YES -co COMPRESS=%s' % d['compress_format']
                op += ' -of %s' % 'GTiff'
                to = gdal.WarpOptions(gdal.ParseCommandLine(op))
            else:
                _vector = ogr.Open(d['vector_path'])
                _v_layer = _vector.GetLayer()
                v_sr = _v_layer.GetSpatialRef()
                if sr.IsSame(v_sr) == 1:
                    pass
                else:
                    c_t = osr.CoordinateTransformation(v_sr, sr)
                    _v_layer.Transform(c_t)
                extent = _v_layer.GetExtent()
                crop = True
                op = ' -co BIGTIFF=YES -co COMPRESS=%s' % d['compress_format']
                # op += ' -wo CUTLINE_ALL_TOUCHED = TRUE'
                to = gdal.WarpOptions(
                    gdal.ParseCommandLine(op),
                    format='GTiff', outputBounds=extent,
                    dstSRS=sr.ExportToWkt(), cutlineDSName=d['vector_path'],
                    cropToCutline=crop, cutlineWhere=d['where']
                    )
            gdal.Warp(d['output'], d['input_raster'], options=to)
            _vector = None
            _r_d = None
            results.append([d['output']])
            if progress_queue is not None:
                progress_queue.put([n, len(argument_list)], False)
        except Exception as err:
            errors = str(err)
    return [results, errors, str(process)]


# class to override numpy ufunc
class ArrayLike(np.ndarray):

    def __new__(cls, input_array):
        obj = input_array.view(cls)
        return obj

    def __mul__(self, other):
        a = self
        nan_a_mask = np.isnan(a)
        if nan_a_mask is not None:
            a = np.array(a, subok=True, copy=True)
            np.copyto(a, 1, where=nan_a_mask)
        nan_other_mask = np.isnan(other)
        if nan_other_mask is not None:
            other = np.array(other, subok=True, copy=True)
            np.copyto(other, 1, where=nan_other_mask)
        return np.multiply(a, other)

    def __rmul__(self, other):
        a = self
        nan_a_mask = np.isnan(a)
        if nan_a_mask is not None:
            a = np.array(a, subok=True, copy=True)
            np.copyto(a, 1, where=nan_a_mask)
        nan_other_mask = np.isnan(other)
        if nan_other_mask is not None:
            other = np.array(other, subok=True, copy=True)
            np.copyto(other, 1, where=nan_other_mask)
        return np.multiply(other, a)

    def __truediv__(self, other):
        nan_mask = np.isnan(other)
        if nan_mask is not None:
            other = np.array(other, subok=True, copy=True)
            np.copyto(other, 1, where=nan_mask)
        division = np.divide(self, other)
        try:
            division[nan_mask] = np.nan
        except Exception as err:
            str(err)
        return division

    def __rtruediv__(self, other):
        a = self
        nan_mask = np.isnan(self)
        if nan_mask is not None:
            a = np.array(a, subok=True, copy=True)
            np.copyto(a, 1, where=nan_mask)
        division = np.divide(other, a)
        try:
            division[nan_mask] = np.nan
        except Exception as err:
            str(err)
        return division

    def __add__(self, other):
        a = self
        nan_a_mask = np.isnan(a)
        if nan_a_mask is not None:
            a = np.array(a, subok=True, copy=True)
            np.copyto(a, 0, where=nan_a_mask)
        nan_other_mask = np.isnan(other)
        if nan_other_mask is not None:
            other = np.array(other, subok=True, copy=True)
            np.copyto(other, 0, where=nan_other_mask)
        return np.add(a, other)

    def __radd__(self, other):
        a = self
        nan_a_mask = np.isnan(a)
        if nan_a_mask is not None:
            a = np.array(a, subok=True, copy=True)
            np.copyto(a, 0, where=nan_a_mask)
        nan_other_mask = np.isnan(other)
        if nan_other_mask is not None:
            other = np.array(other, subok=True, copy=True)
            np.copyto(other, 0, where=nan_other_mask)
        return np.add(other, a)

    def __sub__(self, other):
        a = self
        nan_a_mask = np.isnan(a)
        if nan_a_mask is not None:
            a = np.array(a, subok=True, copy=True)
            np.copyto(a, 0, where=nan_a_mask)
        nan_other_mask = np.isnan(other)
        if nan_other_mask is not None:
            other = np.array(other, subok=True, copy=True)
            np.copyto(other, 0, where=nan_other_mask)
        return np.add(a, -other)

    def __rsub__(self, other):
        a = self
        nan_a_mask = np.isnan(a)
        if nan_a_mask is not None:
            a = np.array(a, subok=True, copy=True)
            np.copyto(a, 0, where=nan_a_mask)
        nan_other_mask = np.isnan(other)
        if nan_other_mask is not None:
            other = np.array(other, subok=True, copy=True)
            np.copyto(other, 0, where=nan_other_mask)
        return np.add(other, -a)
