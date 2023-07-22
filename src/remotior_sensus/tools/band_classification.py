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
"""Band classification.

This tool allows for the classification of remote sensing images, providing
several algorithms such as Minimum Distance, Maximum Likelihood,
Spectral Angle Mapping.
Also, machine learning algorithms are provided through
`PyTorch <https://pytorch.org/>`_ (pytorch_multi_layer_perceptron) and
`scikit-learn <https://scikit-learn.org/stable/>`_ (random_forest,
random_forest_ovr, support_vector_machine, multi_layer_perceptron).
This module includes tools for training the algorithms using
Regions of Interest (ROIs) or spectral signatures.

Typical usage example:

    >>> # import Remotior Sensus and start the session
    >>> import remotior_sensus
    >>> rs = remotior_sensus.Session()
    >>> # start the process
    >>> classification = rs.band_classification(
    ... input_bands=['file1.tif', 'file2.tif'], output_path='output.tif',
    ... algorithm_name=cfg.maximum_likelihood
    ... )
"""

import pickle
from copy import deepcopy
from typing import Union, Optional

import numpy as np

from remotior_sensus.core import configurations as cfg
from remotior_sensus.core.bandset_catalog import BandSet
from remotior_sensus.core.bandset_catalog import BandSetCatalog
from remotior_sensus.core.output_manager import OutputManager
from remotior_sensus.core.processor_functions import (
    classification_maximum_likelihood, classification_minimum_distance,
    classification_spectral_angle_mapping, classification_scikit,
    classification_pytorch, get_band_arrays, fit_classifier,
    score_classifier_stratified, score_classifier
)
from remotior_sensus.core.spectral_signatures import SpectralSignaturesCatalog
from remotior_sensus.util import (
    files_directories, raster_vector, shared_tools, read_write_files
)

try:
    import torch
    from remotior_sensus.util.pytorch_tools import train_pytorch_model
except Exception as error:
    if cfg.logger is not None:
        cfg.logger.log.error(str(error))
    else:
        print(str(error))

try:
    from sklearn import svm
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neural_network import MLPClassifier
except Exception as error:
    if cfg.logger is not None:
        cfg.logger.log.error(str(error))
    else:
        print(str(error))


class Classifier(object):
    """Manages classifiers.

    A classifier is an object which includes the required parameters to perform 
    a classification, including the tools to perform the training.

    Attributes:
        algorithm_name: algorithm name selected form cfg.classification_algorithms.
        spectral_signatures: a SpectralSignaturesCatalog containing spectral signatures.
        covariance_matrices: dictionary of previously calculated covariance matrices 
            (used in maximum_likelihood).
        model_classifier: classifier object.
        input_normalization: perform input normalization; options are z_score or linear_scaling.
        normalization_values: list of normalization paramters defined for each variable
            [normalization expressions, mean values, 
            standar deviation values, minimum values, maximum values].
        framework_name: name of framework such as 
            classification_framework, scikit_framework, or pytorch_framework.
        classification_function: the actual classification function.
        function_argument = a dictionary including arguments for the classification function
            such as model_classifier, covariance_matrices, normalization_values, 
            spectral_signatures_catalog.

    Examples:
        Fit a classifier
            >>> # Start a session
            >>> import remotior_sensus
            >>> rs = remotior_sensus.Session()
            >>> # create a BandSet
            >>> catalog = rs.bandset_catalog()
            >>> file_list = ['file1.tif', 'file2.tif', 'file3.tif']
            >>> catalog.create_bandset(file_list, wavelengths=['Landsat 8'])
            >>> # set a BandSet reference in signature catalog
            >>> signature_catalog = rs.spectral_signatures_catalog(
            >>>     bandset=catalog.get(1))
            >>> # import vector in signature catalog
            >>> signature_catalog.import_vector(
            >>>     file_path='file.gpkg', macroclass_field='macroclass', class_field='class',
            >>>     macroclass_name_field='macroclass', class_name_field='class',
            >>>     calculate_signature=True)
            >>> # train a minimum distance classifier
            >>> classifier = Classifier.train(
            >>>     spectral_signatures=signature_catalog,
            >>>     algorithm_name='minimum distance'
            >>> )
        """  # noqa: E501

    def __init__(
            self, algorithm_name, spectral_signatures, covariance_matrices,
            model_classifier, input_normalization, normalization_values
    ):
        """Initializes a classifier.

        A classifier is an object which includes the 

        Args:
            algorithm_name: algorithm name selected form cfg.classification_algorithms.
            spectral_signatures: a SpectralSignaturesCatalog containing spectral signatures.
            covariance_matrices: dictionary of previously calculated covariance matrices 
                (used in maximum_likelihood).
            model_classifier: classifier object.
            input_normalization: perform input normalization; options are z_score or linear_scaling.
            normalization_values: list of normalization parameters defined for each variable
                [normalization expressions, mean values, 
                standard deviation values, minimum values, maximum values].
        """  # noqa: E501
        self.algorithm_name = algorithm_name
        self.spectral_signatures = spectral_signatures
        self.covariance_matrices = covariance_matrices
        self.model_classifier = model_classifier
        self.input_normalization = input_normalization
        self.normalization_values = normalization_values
        self.framework_name = None
        self.classification_function = None
        # spectral signatures catalog
        if (algorithm_name == cfg.minimum_distance
                or algorithm_name == cfg.spectral_angle_mapping
                or algorithm_name == cfg.maximum_likelihood):
            if spectral_signatures.signatures is None:
                # import vector
                spectral_signatures.import_vector(
                    file_path=spectral_signatures.geometry_file,
                    macroclass_field=spectral_signatures.macroclass_field,
                    class_field=spectral_signatures.class_field,
                    macroclass_name_field=(
                        spectral_signatures.macroclass_field),
                    class_name_field=spectral_signatures.class_field,
                    calculate_signature=True
                )
            spectral_signatures_catalog = _normalize_signatures(
                spectral_signatures=spectral_signatures,
                normalization_values=normalization_values,
                input_normalization=input_normalization
            )
        else:
            spectral_signatures_catalog = spectral_signatures
        # minimum distance
        if algorithm_name == cfg.minimum_distance:
            self.classification_function = classification_minimum_distance
        # spectral angle mapping
        elif algorithm_name == cfg.spectral_angle_mapping:
            self.classification_function = \
                classification_spectral_angle_mapping
        # maximum likelihood
        elif algorithm_name == cfg.maximum_likelihood:
            self.classification_function = classification_maximum_likelihood
        # scikit framework
        elif (
                algorithm_name == cfg.random_forest or algorithm_name ==
                cfg.random_forest_ovr or algorithm_name ==
                cfg.support_vector_machine or algorithm_name ==
                cfg.multi_layer_perceptron):
            self.classification_function = classification_scikit
            self.framework_name = cfg.scikit_framework
        # pytorch framework
        elif algorithm_name == cfg.pytorch_multi_layer_perceptron:
            self.classification_function = classification_pytorch
            self.framework_name = cfg.pytorch_framework
        self.function_argument = {
            cfg.model_classifier_framework: model_classifier,
            cfg.covariance_matrices_framework: covariance_matrices,
            cfg.normalization_values_framework: normalization_values,
            cfg.spectral_signatures_framework: spectral_signatures_catalog
        }

    def save_model(self, output_path: str) -> OutputManager:
        """Saves classifier model.

        Saves a classifier model to file to be loaded later.

        Args:
            output_path: path of output file.
        
        Returns:
            OutputManager object with
                - path = [output path]
            
        Examples:
            Save a trainied classifier
                >>> classifier = Classifier()
                >>> # after the training
                >>> saved_model = classifier.save_model(output_path=output_path)
        """  # noqa: E501
        # list of zipped files
        files = []
        # save classification framework
        file_path = cfg.temp.temporary_file_path(
            name=cfg.classification_framework
        )
        classification_parameters = '%s=%s%s%s=%s' % (
            cfg.algorithm_name_framework, str(self.algorithm_name),
            cfg.new_line, cfg.input_normalization_framework,
            str(self.input_normalization))
        read_write_files.write_file(
            data=classification_parameters, output_path=file_path
        )
        files.append(file_path)
        if self.spectral_signatures is not None:
            file_path = cfg.temp.temporary_file_path(
                name=cfg.spectral_signatures_framework, name_suffix='.pickle'
            )
            # save model using pickle
            with open(file_path, 'wb') as file:
                pickle.dump(self.spectral_signatures, file, protocol=4)
            files.append(file_path)
        if self.normalization_values is not None:
            file_path = cfg.temp.temporary_file_path(
                name=cfg.normalization_values_framework, name_suffix='.pickle'
            )
            # save model using pickle
            with open(file_path, 'wb') as file:
                pickle.dump(self.normalization_values, file, protocol=4)
            files.append(file_path)
        if self.covariance_matrices is not None:
            file_path = cfg.temp.temporary_file_path(
                name=cfg.covariance_matrices_framework, name_suffix='.pickle'
            )
            # save model using pickle
            with open(file_path, 'wb') as file:
                pickle.dump(self.covariance_matrices, file, protocol=4)
            files.append(file_path)
        if self.framework_name == cfg.scikit_framework:
            file_path = cfg.temp.temporary_file_path(
                name=cfg.model_classifier_framework, name_suffix='.pickle'
            )
            # save model using pickle
            with open(file_path, 'wb') as file:
                pickle.dump(self.model_classifier, file, protocol=4)
            files.append(file_path)
        elif self.framework_name == cfg.pytorch_framework:
            file_path = cfg.temp.temporary_file_path(
                name=cfg.model_classifier_framework, name_suffix='.pth'
            )
            torch.save(self.model_classifier, file_path)
            files.append(file_path)
        # zip files
        if files_directories.file_extension(output_path) != cfg.rsmo_suffix:
            output_path = shared_tools.join_path(
                files_directories.parent_directory(output_path), '{}{}'.format(
                    files_directories.file_name(output_path, suffix=False),
                    cfg.rsmo_suffix
                )
            ).replace('\\', '/')
        files_directories.zip_files(files, output_path)
        cfg.logger.log.debug('output_path: %s' % str(output_path))
        return OutputManager(path=output_path)

    @classmethod
    def load_classifier(
            cls, algorithm_name=None, spectral_signatures=None,
            covariance_matrices=None, model_classifier=None,
            input_normalization=None, normalization_values=None
    ):
        """Loads a classifier.

        Creates a classifier from loading.

        Args:
            algorithm_name: algorithm name selected form cfg.classification_algorithms.
            spectral_signatures: a SpectralSignaturesCatalog containing spectral signatures.
            covariance_matrices: dictionary of previously calculated covariance matrices 
                (used in maximum_likelihood).
            model_classifier: classifier object.
            input_normalization: perform input normalization; options are z_score or linear_scaling.
            normalization_values: list of normalization parameters defined for each variable
                [normalization expressions, mean values, 
                standard deviation values, minimum values, maximum values].

        Returns:
            Classifier object.

        Examples:
            Load a classifier
                >>> classifier = Classifier.load_classifier(
                >>> algorithm_name=algorithm_name, spectral_signatures=spectral_signatures,
                >>> covariance_matrices=covariance_matrices, model_classifier=model_classifier,
                >>> input_normalization=input_normalization, normalization_values=normalization_values)
        """  # noqa: E501
        # return classifier
        return cls(
            algorithm_name=algorithm_name,
            spectral_signatures=spectral_signatures,
            covariance_matrices=covariance_matrices,
            model_classifier=model_classifier,
            input_normalization=input_normalization,
            normalization_values=normalization_values
        )

    # noinspection PyTypeChecker
    @classmethod
    def train(
            cls, spectral_signatures=None, algorithm_name=None,
            covariance_matrices=None, svc_classification_confidence=None,
            n_processes: int = None, available_ram: int = None,
            cross_validation=True, x_matrix=None,
            y=None, class_weight=None, input_normalization=None,
            normalization_values=None, find_best_estimator=False,
            rf_max_features=None, rf_number_trees=100,
            rf_min_samples_split=None, svm_c=None, svm_gamma=None,
            svm_kernel=None, pytorch_model=None, pytorch_optimizer=None,
            mlp_training_portion=None, pytorch_loss_function=None,
            mlp_hidden_layer_sizes=None, mlp_alpha=None,
            mlp_learning_rate_init=None, mlp_max_iter=None,
            mlp_batch_size=None, mlp_activation=None,
            pytorch_optimization_n_iter_no_change=None,
            pytorch_optimization_tol=None, pytorch_device=None, min_progress=1,
            max_progress=100
    ):
        """Trains a classifier.

        Trains a classifier using ROIs or spectral signatures.

        Args:
            spectral_signatures: a SpectralSignaturesCatalog containing spectral signatures.
            algorithm_name: algorithm name selected from cfg.classification_algorithms; 
                if None, minimum distance is used.
            n_processes: number of parallel processes.
            available_ram: number of megabytes of RAM available to processes.
            cross_validation: if True, perform cross validation for algorithms
                provided through scikit-learn (random_forest, random_forest_ovr, 
                support_vector_machine, multi_layer_perceptron).
            x_matrix: optional previously saved x matrix.
            y: optional previously saved y matrix.
            covariance_matrices: dictionary of previously calculated covariance matrices 
                (used in maximum_likelihood).
            svc_classification_confidence: if True, write also additional
                classification confidence rasters as output; required information for support_vector_machine.
            input_normalization: perform input normalization; options are z_score or linear_scaling.
            normalization_values: list of normalization paramters defined for each variable
                [normalization expressions, mean values, 
                standar deviation values, minimum values, maximum values].
            class_weight: specific for random forest and support_vector_machine, if None each class
                has equal weight 1, if 'balanced' weight is computed inversely
                proportional to class frequency.
            find_best_estimator: specific for scikit classifiers, if True,
                find automatically the best parameters and fit the model, if
                integer the greater the value the more are the tested combinations.
            rf_max_features: specific for random forest, if None all features
                are considered in node splitting, available options are 'sqrt' as
                square root of all the features, an integer number, or
                a float number for a fraction of all the features.
            rf_number_trees: specific for random forest, number of trees in the forest.
            rf_min_samples_split: specific for random forest through scikit, 
                sets the minimum number of samples required to split an internal node; default = 2.
            svm_c: specific for support_vector_machine through scikit,
                sets the regularization parameter C; default = 1.
            svm_gamma: specific for support_vector_machine through scikit, 
                sets the kernel coefficient; default = scale.
            svm_kernel: specific for support_vector_machine through scikit, 
                sets the kernel; default = rbf.
            mlp_training_portion: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
                the proportion of data to be used as training (default = 0.9) and the remaining part
                as test (default = 0.1).
            mlp_hidden_layer_sizes: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
                list of values where each value defines the number of neurons in a hidden layer 
                (e.g., [200, 100] for two hidden layers of 200 and 100 neurons respectively); default = [100].
            mlp_alpha: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
                weight decay (also L2 regularization term) for Adam optimizer (default = 0.0001).
            mlp_learning_rate_init: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
                sets initial learning rate (default = 0.001).
            mlp_max_iter: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
                sets the maximum number of iterations (default = 200).
            mlp_batch_size: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
                sets the number of samples per batch for optimizer; if "auto", the batch is the 
                minimum value between 200 and the number of samples (default = auto).
            mlp_activation: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
                sets the activation function (default relu).
            pytorch_model: specific for pytorch_multi_layer_perceptron,
                custom pytorch nn.Module.
            pytorch_optimizer: specific for pytorch_multi_layer_perceptron,
                custom pytorch optimizer.
            pytorch_loss_function: specific for pytorch_multi_layer_perceptron,
                sets a custom loss function (default CrossEntropyLoss).
            pytorch_optimization_n_iter_no_change: specific for pytorch_multi_layer_perceptron,
                sets the maximum number of epochs where the loss is not improving by 
                at least the value pytorch_optimization_tol (default 5).
            pytorch_optimization_tol: specific for pytorch_multi_layer_perceptron,
                sets the tolerance of optimization (default = 0.0001).
            pytorch_device: specific for pytorch_multi_layer_perceptron,
                processing device 'cpu' (default) or 'cuda' if available.
            min_progress: minimum progress value for :func:`~remotior_sensus.core.progress.Progress`.
            max_progress: maximum progress value for :func:`~remotior_sensus.core.progress.Progress`.

        Returns:
            Classifier object.

        Examples:
            Load a classifier
                >>> import remotior_sensus
                >>> rs = remotior_sensus.Session()
                >>> signature_catalog = rs.spectral_signatures_catalog()
                >>> classifier = Classifier.train(
                >>> spectral_signatures=spectral_signatures,
                >>> algorithm_name='minimum distance')
        """  # noqa: E501
        cfg.logger.log.info('start')
        # classification parameters
        model_classifier = None
        if n_processes is None:
            n_processes = cfg.n_processes
        if available_ram is None:
            available_ram = cfg.available_ram
        # random forest scikit
        if algorithm_name == cfg.random_forest:
            if (find_best_estimator is not None
                    and find_best_estimator is not False):
                if type(find_best_estimator) is int:
                    n_steps = find_best_estimator
                else:
                    n_steps = 5
                if rf_max_features is None:
                    list_rf_max_features = [0.3, 0.65, 1]
                else:
                    list_rf_max_features = [rf_max_features]
                # calculate grid steps
                n_estimators_step = int((rf_number_trees - 10) / n_steps)
                if n_estimators_step <= 0:
                    n_estimators_step = 1
                min_samples_split_step = round((10 - 1) / n_steps)
                if min_samples_split_step <= 0:
                    min_samples_split_step = 1
                # build function argument list of dictionaries
                argument_list = []
                function_list = []
                for n_estimators in range(
                        10, rf_number_trees, n_estimators_step
                ):
                    for min_samples_split in range(
                            2, 10,
                            min_samples_split_step
                    ):
                        for max_features in list_rf_max_features:
                            for class_weight in [None, 'balanced']:
                                argument_list.append(
                                    {
                                        'classifier': RandomForestClassifier(
                                            criterion='gini',
                                            n_estimators=n_estimators,
                                            oob_score=False,
                                            max_features=max_features,
                                            class_weight=class_weight,
                                            n_jobs=1, min_samples_split=(
                                                min_samples_split),
                                            verbose=0, random_state=0
                                        ), 'x_matrix': x_matrix, 'y': y
                                    }
                                )
                                function_list.append(
                                    score_classifier_stratified
                                )
                cfg.multiprocess.run_iterative_process(
                    function_list=function_list, argument_list=argument_list,
                    min_progress=min_progress, max_progress=max_progress
                )
                results = cfg.multiprocess.output
                # fit classifiers
                score_list = []
                score_std_list = []
                classifier_list = []
                for result in results:
                    for r in result:
                        classifier_list.append(r[0])
                        score_list.append(r[1])
                        score_std_list.append(r[2])
                score_index = score_list.index(max(score_list))
                cfg.logger.log.debug(
                    'cross validation score max: %s; score std: %s' % (
                        str(max(score_list)), str(score_std_list[score_index]))
                )
                model_classifier = classifier_list[score_index]
                model_classifier.oob_score = True
                model_classifier.n_jobs = n_processes
                cfg.progress.update(message='fitting')
                model_classifier.fit(x_matrix, y)
                cfg.logger.log.info(
                    'best parameters: %s; feature importance: %s; '
                    'accuracy score: %s' % (
                        str(model_classifier.get_params()), str(
                            model_classifier.feature_importances_
                        ), str(model_classifier.oob_score_))
                )
            else:
                if rf_min_samples_split is None:
                    rf_min_samples_split = 2
                cfg.logger.log.debug(
                    'n_estimators: %s; max_features: %s; class_weight: %s; '
                    'min_samples_split: %s' % (
                        str(rf_number_trees), str(rf_max_features),
                        str(class_weight), str(rf_min_samples_split))
                )
                # build classifier
                try:
                    model_classifier = RandomForestClassifier(
                        criterion='gini', n_estimators=rf_number_trees,
                        oob_score=False, max_features=rf_max_features,
                        class_weight=class_weight, n_jobs=1,
                        min_samples_split=rf_min_samples_split, verbose=2,
                        random_state=0
                    )
                except Exception as err:
                    cfg.logger.log.error(str(err))
                    cfg.messages.error(str(err))
                    return cls(
                        algorithm_name=None,
                        spectral_signatures=None,
                        covariance_matrices=None,
                        model_classifier=None,
                        input_normalization=None,
                        normalization_values=None
                    )
                # perform cross validation if cross_validation is True
                _perform_cross_validation_scikit(
                    cross_validation=cross_validation,
                    classifier=model_classifier, x_matrix=x_matrix, y=y,
                    n_processes=n_processes, available_ram=available_ram
                )
                # fit classifier
                model_classifier.oob_score = True
                model_classifier.n_jobs = n_processes
                cfg.progress.update(message='fitting')
                model_classifier.fit(x_matrix, y)
                model_classifier.n_jobs = 1
                cfg.logger.log.info(
                    'feature importance: %s; accuracy score: %s' % (
                        str(model_classifier.feature_importances_),
                        str(model_classifier.oob_score_))
                )
        # random forest ovr scikit
        elif algorithm_name == cfg.random_forest_ovr:
            if (
                    find_best_estimator is not None and find_best_estimator
                    is not False):
                if type(find_best_estimator) is int:
                    n_steps = find_best_estimator
                else:
                    n_steps = 5
                if rf_max_features is None:
                    list_rf_max_features = [0.3, 0.65, 1]
                else:
                    list_rf_max_features = [rf_max_features]
                # calculate grid steps
                n_estimators_step = int((rf_number_trees - 10) / n_steps)
                if n_estimators_step <= 0:
                    n_estimators_step = 1
                min_samples_split_step = round((10 - 1) / n_steps)
                if min_samples_split_step <= 0:
                    min_samples_split_step = 1
                # build function argument list of dictionaries
                argument_list = []
                function_list = []
                for n_estimators in range(
                        10, rf_number_trees, n_estimators_step
                ):
                    for min_samples_split in range(
                            2, 10, min_samples_split_step
                    ):
                        for max_features in list_rf_max_features:
                            for class_weight in [None, 'balanced']:
                                argument_list.append(
                                    {
                                        'classifier': OneVsRestClassifier(
                                            RandomForestClassifier(
                                                criterion='gini',
                                                n_estimators=n_estimators,
                                                oob_score=False,
                                                max_features=max_features,
                                                class_weight=class_weight,
                                                n_jobs=1,
                                                min_samples_split=(
                                                    min_samples_split),
                                                verbose=0, random_state=0
                                            )
                                        ), 'x_matrix': x_matrix, 'y': y
                                    }
                                )
                                function_list.append(
                                    score_classifier_stratified
                                )
                cfg.multiprocess.run_iterative_process(
                    function_list=function_list, argument_list=argument_list,
                    min_progress=min_progress, max_progress=max_progress
                )
                results = cfg.multiprocess.output
                # fit classifiers
                score_list = []
                score_std_list = []
                classifier_list = []
                for result in results:
                    for r in result:
                        classifier_list.append(r[0])
                        score_list.append(r[1])
                        score_std_list.append(r[2])
                score_index = score_list.index(max(score_list))
                model_classifier = classifier_list[score_index]
                cfg.progress.update(message='fitting')
                model_classifier.fit(x_matrix, y)
                cfg.logger.log.info(
                    'best parameters: %s; cross validation score max: %s; '
                    'score std: %s' % (str(model_classifier.get_params()),
                                       str(max(score_list)),
                                       str(score_std_list[score_index]))
                )
            else:
                if rf_min_samples_split is None:
                    rf_min_samples_split = 2
                cfg.logger.log.debug(
                    'n_estimators: %s; max_features: %s; class_weight: %s; '
                    'min_samples_split: %s' % (
                        str(rf_number_trees), str(rf_max_features),
                        str(class_weight), str(rf_min_samples_split))
                )
                # build classifier
                model_classifier = OneVsRestClassifier(
                    RandomForestClassifier(
                        criterion='gini', n_estimators=rf_number_trees,
                        oob_score=True, max_features=rf_max_features,
                        class_weight=class_weight, n_jobs=1,
                        min_samples_split=rf_min_samples_split, random_state=0
                    )
                )
                # perform cross validation if cross_validation is True
                _perform_cross_validation_scikit(
                    cross_validation=cross_validation,
                    classifier=model_classifier, x_matrix=x_matrix, y=y,
                    n_processes=n_processes, available_ram=available_ram
                )
                # fit classifier
                cfg.progress.update(message='fitting')
                model_classifier.fit(x_matrix, y)
                cfg.logger.log.info(
                    'OneVsRestClassifier: RandomForestClassifier'
                )
        # support vector machine scikit
        elif algorithm_name == cfg.support_vector_machine:
            if svc_classification_confidence is None:
                svc_classification_confidence = False
            if (
                    find_best_estimator is not None and find_best_estimator
                    is not False):
                if type(find_best_estimator) is int:
                    n_steps = find_best_estimator
                else:
                    n_steps = 5
                # calculate grid steps
                kernels = ['linear', 'rbf']
                c_steps = np.logspace(-5, 10, n_steps).tolist()
                gamma_steps = np.logspace(-10, 3, n_steps).tolist()
                gamma_steps.append('scale')
                # build function argument list of dictionaries
                argument_list = []
                function_list = []
                for kernel in kernels:
                    for c_step in c_steps:
                        for gamma in gamma_steps:
                            for class_weight in [None, 'balanced']:
                                argument_list.append(
                                    {
                                        'classifier': svm.SVC(
                                            probability=(
                                                svc_classification_confidence),
                                            kernel=kernel, gamma=gamma,
                                            C=c_step,
                                            class_weight=class_weight,
                                            cache_size=int(
                                                available_ram / (
                                                        2 * n_processes)
                                            ), verbose=0, random_state=0
                                        ), 'x_matrix': x_matrix, 'y': y
                                    }
                                )
                                function_list.append(
                                    score_classifier_stratified
                                )
                cfg.multiprocess.run_iterative_process(
                    function_list=function_list, argument_list=argument_list,
                    min_progress=min_progress, max_progress=max_progress
                )
                results = cfg.multiprocess.output
                # fit classifier
                score_list = []
                score_std_list = []
                classifier_list = []
                for result in results:
                    for r in result:
                        classifier_list.append(r[0])
                        score_list.append(r[1])
                        score_std_list.append(r[2])
                score_index = score_list.index(max(score_list))
                model_classifier = classifier_list[score_index]
                cfg.progress.update(message='fitting')
                model_classifier.fit(x_matrix, y)
                cfg.logger.log.info(
                    'best parameters: %s; cross validation score max: %s; '
                    'score std: %s' % (str(model_classifier.get_params()),
                                       str(max(score_list)),
                                       str(score_std_list[score_index]))
                )
            else:
                if svm_c is None:
                    svm_c = 1
                if svm_gamma is None:
                    svm_gamma = 'scale'
                if svm_kernel is None:
                    svm_kernel = 'rbf'
                cfg.logger.log.debug(
                    'kernel: %s; gamma: %s; c: %s; class_weight: %s' % (
                        str(svm_kernel), str(svm_gamma), str(svm_c),
                        str(class_weight))
                )
                # build classifier
                model_classifier = svm.SVC(
                    probability=svc_classification_confidence,
                    kernel=svm_kernel, gamma=svm_gamma, C=svm_c,
                    random_state=0, class_weight=class_weight,
                    cache_size=int(available_ram / (2 * n_processes)),
                    verbose=0
                )
                # perform cross validation if cross_validation is True
                _perform_cross_validation_scikit(
                    cross_validation=cross_validation,
                    classifier=model_classifier, x_matrix=x_matrix, y=y,
                    n_processes=n_processes, available_ram=available_ram
                )
                # fit classifier
                model_classifier.cache_size = int(available_ram / 3)
                cfg.progress.update(message='fitting')
                model_classifier.fit(x_matrix, y)
        # multilayer perceptron scikit
        elif algorithm_name == cfg.multi_layer_perceptron:
            if mlp_training_portion is None:
                mlp_training_portion = 0.9
            elif mlp_training_portion > 1:
                mlp_training_portion = 0.9
            if mlp_activation is None:
                mlp_activation = 'relu'
            if (
                    find_best_estimator is not None and find_best_estimator
                    is not False):
                if type(find_best_estimator) is int:
                    n_steps = find_best_estimator
                else:
                    n_steps = 5
                if mlp_max_iter is None:
                    mlp_max_iter = 200
                # calculate grid steps
                hidden_layer_sizes_steps = list(range(50, 500, 100))
                if n_steps > 1:
                    for neurons_1 in range(50, 200, 50):
                        for neurons_2 in range(50, 200, 50):
                            hidden_layer_sizes_steps.append(
                                [neurons_1, neurons_2]
                            )
                if n_steps > 3:
                    for neurons_1 in range(50, 150, 50):
                        for neurons_2 in range(50, 150, 50):
                            for neurons_3 in range(50, 150, 50):
                                hidden_layer_sizes_steps.append(
                                    [neurons_1, neurons_2, neurons_3]
                                )
                if n_steps > 5:
                    for neurons_1 in range(25, 75, 25):
                        for neurons_2 in range(25, 75, 25):
                            for neurons_3 in range(25, 75, 25):
                                for neurons_4 in range(25, 75, 25):
                                    hidden_layer_sizes_steps.append(
                                        [neurons_1, neurons_2, neurons_3,
                                         neurons_4]
                                    )
                learning_rate_init_steps = np.logspace(
                    -4, -1, n_steps
                ).tolist()
                alpha_steps = np.logspace(-5, -2, n_steps).tolist()
                batch_max = max(200, int(x_matrix.shape[0] / 100))
                batch_min = int(batch_max / n_steps)
                batch_size_steps = [batch_min, batch_max]
                # build function argument list of dictionaries
                argument_list = []
                function_list = []
                for alpha in alpha_steps:
                    for learning_rate_init in learning_rate_init_steps:
                        for hidden_layer_sizes in hidden_layer_sizes_steps:
                            for batch in batch_size_steps:
                                argument_list.append(
                                    {
                                        'classifier': MLPClassifier(
                                            hidden_layer_sizes=(
                                                hidden_layer_sizes),
                                            alpha=alpha,
                                            learning_rate_init=(
                                                learning_rate_init),
                                            validation_fraction=(
                                                    1 - mlp_training_portion),
                                            activation=mlp_activation,
                                            random_state=0, batch_size=batch,
                                            max_iter=mlp_max_iter, verbose=0
                                        ), 'x_matrix': x_matrix, 'y': y
                                    }
                                )
                                function_list.append(
                                    score_classifier_stratified
                                )
                cfg.multiprocess.run_iterative_process(
                    function_list=function_list, argument_list=argument_list,
                    min_progress=min_progress, max_progress=max_progress
                )
                results = cfg.multiprocess.output
                # fit classifier
                score_list = []
                score_std_list = []
                classifier_list = []
                for result in results:
                    for r in result:
                        classifier_list.append(r[0])
                        score_list.append(r[1])
                        score_std_list.append(r[2])
                score_index = score_list.index(max(score_list))
                model_classifier = classifier_list[score_index]
                cfg.progress.update(message='fitting')
                model_classifier.fit(x_matrix, y)
                cfg.logger.log.info(
                    'best parameters: %s; cross validation score max: %s; '
                    'score std: %s' % (str(model_classifier.get_params()),
                                       str(max(score_list)),
                                       str(score_std_list[score_index]))
                )
            else:
                if mlp_hidden_layer_sizes is None:
                    mlp_hidden_layer_sizes = [100]
                if mlp_alpha is None:
                    mlp_alpha = 0.0001
                if mlp_learning_rate_init is None:
                    mlp_learning_rate_init = 0.001
                if mlp_max_iter is None:
                    mlp_max_iter = 200
                if mlp_batch_size is None:
                    mlp_batch_size = int(
                        min(
                            2000, min(
                                x_matrix.shape[0], max(
                                    200, x_matrix.shape[0] / 100
                                )
                            )
                        )
                    )
                cfg.logger.log.debug(
                    'hidden_layer_sizes: %s; alpha: %s; learning_rate_init: '
                    '%s; max_iter: %s; batch_size: %s;  activation: %s' % (
                        str(mlp_hidden_layer_sizes), str(mlp_alpha),
                        str(mlp_learning_rate_init), str(mlp_max_iter),
                        str(mlp_batch_size), str(mlp_activation))
                )
                # build classifier
                model_classifier = MLPClassifier(
                    hidden_layer_sizes=mlp_hidden_layer_sizes, alpha=mlp_alpha,
                    activation=mlp_activation,
                    learning_rate_init=mlp_learning_rate_init,
                    max_iter=mlp_max_iter, batch_size=mlp_batch_size,
                    verbose=0, random_state=0,
                    validation_fraction=1 - mlp_training_portion
                )
                # perform cross validation if cross_validation is True
                _perform_cross_validation_scikit(
                    cross_validation=cross_validation,
                    classifier=model_classifier, x_matrix=x_matrix, y=y,
                    n_processes=n_processes, available_ram=available_ram
                )
                # fit classifier
                cfg.progress.update(message='fitting')
                model_classifier.fit(x_matrix, y)
        # multilayer perceptron pytorch
        elif algorithm_name == cfg.pytorch_multi_layer_perceptron:
            if mlp_training_portion is None:
                mlp_training_portion = 0.9
            elif mlp_training_portion > 1:
                mlp_training_portion = 0.9
            if mlp_max_iter is None:
                mlp_max_iter = 200
            if mlp_batch_size is None or mlp_batch_size == 'auto':
                mlp_batch_size = int(
                    min(
                        2000, min(
                            x_matrix.shape[0],
                            max(200, x_matrix.shape[0] / 100)
                        )
                    )
                )
            cfg.progress.update(message='fitting')
            (model_classifier, training_loss, test_loss,
             accuracy) = train_pytorch_model(
                x_matrix=x_matrix, y_matrix=y, pytorch_model=pytorch_model,
                activation=mlp_activation, batch_size=mlp_batch_size,
                n_processes=n_processes, training_portion=mlp_training_portion,
                pytorch_optimizer=pytorch_optimizer,
                hidden_layer_sizes=mlp_hidden_layer_sizes,
                loss_function=pytorch_loss_function,
                learning_rate_init=mlp_learning_rate_init,
                optimization_n_iter_no_change=(
                    pytorch_optimization_n_iter_no_change
                ), optimization_tol=pytorch_optimization_tol,
                weight_decay=mlp_alpha, max_iterations=mlp_max_iter,
                device=pytorch_device, min_progress=min_progress,
                max_progress=max_progress
            )
        cfg.logger.log.info('end')
        # return classifier
        return cls(
            algorithm_name=algorithm_name,
            spectral_signatures=spectral_signatures,
            covariance_matrices=covariance_matrices,
            model_classifier=model_classifier,
            input_normalization=input_normalization,
            normalization_values=normalization_values
        )

    def run_prediction(
            self, input_raster_list, output_raster_path,
            n_processes: Optional[int] = None,
            available_ram: Optional[int] = None,
            macroclass: Optional[bool] = True,
            threshold: Optional[bool] = False,
            signature_raster: Optional[bool] = False,
            classification_confidence: Optional[bool] = False,
            virtual_raster: Optional[bool] = None,
            min_progress: Optional[int] = 1, max_progress: Optional[int] = 100
    ):
        """Runs prediction.

        Performs multiprocess classification using a trained classifier using 
        input bands.

        Args:
            input_raster_list: list of paths of input rasters.
            output_raster_path: path of output file.
            n_processes: number of parallel processes.
            available_ram: number of megabytes of RAM available to processes.
            macroclass: if True, use macroclass ID from ROIs or spectral signatures; if False use class ID.
            threshold: if False, classification without threshold; if True,
                use single threshold for each signature; if float, use this value 
                as threshold for all the signature.
            classification_confidence: if True, write also additional
                classification confidence rasters as output.
            signature_raster: if True, write additional rasters for each
                spectral signature as output.
            virtual_raster: if True, create virtual raster output.
            min_progress: minimum progress value for :func:`~remotior_sensus.core.progress.Progress`.
            max_progress: maximum progress value for :func:`~remotior_sensus.core.progress.Progress`.
                
                
        Returns:
            OutputManager object with
                - path = [output path]

        Examples:
            Save a trainied classifier
                >>> classifier = Classifier()
                >>> # after the training
                >>> prediction = classifier.run_prediction(
                ... input_raster_list=['file1.tif', 'file2.tif', 'file3.tif'], 
                ... output_raster_path='file_path')
        """  # noqa: E501
        # create virtual raster of input
        vrt_check = raster_vector.create_temporary_virtual_raster(
            input_raster_list
        )
        if n_processes is None:
            n_processes = cfg.n_processes
        # dummy bands for memory calculation
        dummy_bands = 5
        if self.classification_function is not None:
            cfg.multiprocess.run(
                raster_path=vrt_check, function=self.classification_function,
                function_argument=self.function_argument,
                n_processes=n_processes,
                available_ram=available_ram, dummy_bands=dummy_bands,
                function_variable=[macroclass, threshold],
                output_raster_path=output_raster_path, classification=True,
                classification_confidence=classification_confidence,
                signature_raster=signature_raster,
                virtual_raster=virtual_raster,
                progress_message='classification', min_progress=min_progress,
                max_progress=max_progress
            )
            cfg.logger.log.debug('output_path: %s' % str(output_raster_path))
            return OutputManager(path=output_raster_path)
        else:
            cfg.logger.log.error('classification function not available')
            return OutputManager(check=False)


def band_classification(
        input_bands: Union[list, int, BandSet],
        output_path: Optional[str] = None, overwrite: Optional[bool] = False,
        spectral_signatures: Optional[SpectralSignaturesCatalog] = None,
        algorithm_name: Optional[str] = None,
        bandset_catalog: Optional[BandSetCatalog] = None,
        macroclass: Optional[bool] = True,
        threshold: Optional[Union[bool, float]] = False,
        classification_confidence: Optional[bool] = False,
        signature_raster: Optional[bool] = False,
        n_processes: Optional[int] = None,
        available_ram: Optional[int] = None,
        cross_validation: Optional[bool] = True,
        x_input: Optional[np.array] = None, y_input: Optional[np.array] = None,
        covariance_matrices: Optional[dict] = None,
        input_normalization: Optional[str] = None,
        load_classifier: Optional[str] = None,
        save_classifier: Optional[bool] = False,
        only_fit: Optional[bool] = False,
        class_weight: Optional[Union[None, str, dict]] = None,
        find_best_estimator=False, rf_max_features=None,
        rf_number_trees: Optional[int] = 100,
        rf_min_samples_split: Optional[Union[None, int, float]] = None,
        svm_c: Optional[float] = None,
        svm_gamma: Optional[Union[str, float]] = None,
        svm_kernel: Optional[str] = None,
        mlp_training_portion: Optional[Union[None, float]] = None,
        mlp_hidden_layer_sizes: Optional[Union[None, tuple, list]] = None,
        mlp_alpha: Optional[float] = None,
        mlp_learning_rate_init: Optional[float] = None,
        mlp_max_iter: Optional[float] = None,
        mlp_batch_size: Optional[Union[None, int, str]] = None,
        mlp_activation: Optional[Union[None, str]] = None,
        pytorch_model: Optional = None, pytorch_optimizer: Optional = None,
        pytorch_loss_function: Optional = None,
        pytorch_optimization_n_iter_no_change: Optional[
            Union[None, int]] = None,
        pytorch_optimization_tol: Optional[Union[None, int]] = None,
        pytorch_device: Optional[Union[None, str]] = None,
        progress_message: Optional[bool] = True
) -> OutputManager:
    """Performs band classification.

    This tool allows for classification of raster bands using the selected
    algorithm.

    Args:
        input_bands: list of input raster paths, or a BandSet number,
            or a previously defined BandSet.
        output_path: path of output file.
        overwrite: if True, output overwrites existing files.
        spectral_signatures: a SpectralSignaturesCatalog containing spectral signatures.
        algorithm_name: algorithm name selected from cfg.classification_algorithms; 
            if None, minimum distance is used.
        bandset_catalog: BandSetCatalog object.
        macroclass: if True, use macroclass ID from ROIs or spectral signatures; if False use class ID.
        threshold: if False, classification without threshold; if True,
            use single threshold for each signature; if float, use this value 
            as threshold for all the signature.
        classification_confidence: if True, write also additional
            classification confidence rasters as output.
        signature_raster: if True, write additional rasters for each
            spectral signature as output.
        n_processes: number of parallel processes.
        available_ram: number of megabytes of RAM available to processes.
        cross_validation: if True, perform cross validation for algorithms
            provided through scikit-learn (random_forest, random_forest_ovr, 
            support_vector_machine, multi_layer_perceptron).
        load_classifier: path to a previously saved classifier.
        x_input: optional previously saved x matrix.
        y_input: optional previously saved y matrix.
        covariance_matrices: dictionary of previously calculated covariance matrices 
            (used in maximum_likelihood).
        input_normalization: perform input normalization; options are z_score or linear_scaling.
        only_fit: perform only classifier fitting.
        save_classifier: save classifier to file.
        class_weight: specific for random forest and support_vector_machine, if None each class
            has equal weight 1, if 'balanced' weight is computed inversely
            proportional to class frequency.
        find_best_estimator: specific for scikit classifiers, if True,
            find automatically the best parameters and fit the model, if
            integer the greater the value the more are the tested combinations.
        rf_max_features: specific for random forest, if None all features
            are considered in node splitting, available options are 'sqrt' as
            square root of all the features, an integer number, or
            a float number for a fraction of all the features.
        rf_number_trees: specific for random forest, number of trees in the forest.
        rf_min_samples_split: specific for random forest through scikit, 
            sets the minimum number of samples required to split an internal node; default = 2.
        svm_c: specific for support_vector_machine through scikit,
            sets the regularization parameter C; default = 1.
        svm_gamma: specific for support_vector_machine through scikit, 
            sets the kernel coefficient; default = scale.
        svm_kernel: specific for support_vector_machine through scikit, 
            sets the kernel; default = rbf.
        mlp_training_portion: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
            the proportion of data to be used as training (default = 0.9) and the remaining part
            as test (default = 0.1).
        mlp_hidden_layer_sizes: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
            list of values where each value defines the number of neurons in a hidden layer 
            (e.g., [200, 100] for two hidden layers of 200 and 100 neurons respectively); default = [100].
        mlp_alpha: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
            weight decay (also L2 regularization term) for Adam optimizer (default = 0.0001).
        mlp_learning_rate_init: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
            sets initial learning rate (default = 0.001).
        mlp_max_iter: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
            sets the maximum number of iterations (default = 200).
        mlp_batch_size: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
            sets the number of samples per batch for optimizer; if "auto", the batch is the 
            minimum value between 200 and the number of samples (default = auto).
        mlp_activation: specific for multi_layer_perceptron and pytorch_multi_layer_perceptron,
            sets the activation function (default relu).
        pytorch_model: specific for pytorch_multi_layer_perceptron,
            custom pytorch nn.Module.
        pytorch_optimizer: specific for pytorch_multi_layer_perceptron,
            custom pytorch optimizer.
        pytorch_loss_function: specific for pytorch_multi_layer_perceptron,
            sets a custom loss function (default CrossEntropyLoss).
        pytorch_optimization_n_iter_no_change: specific for pytorch_multi_layer_perceptron,
            sets the maximum number of epochs where the loss is not improving by 
            at least the value pytorch_optimization_tol (default 5).
        pytorch_optimization_tol: specific for pytorch_multi_layer_perceptron,
            sets the tolerance of optimization (default = 0.0001).
        pytorch_device: specific for pytorch_multi_layer_perceptron,
            processing device 'cpu' (default) or 'cuda' if available.
        progress_message: progress message.

    Returns:
        If only_fit is True returns :func:`~remotior_sensus.core.output_manager.OutputManager` object with
            - extra = {'classifier': classifier, 'model_path': output model path}
        
        If only_fit is False returns :func:`~remotior_sensus.core.output_manager.OutputManager` object with
            - path = classification path
            - extra = {'model_path': output model path}
    """  # noqa: E501
    cfg.logger.log.info('start')
    if progress_message:
        cfg.progress.update(
            process=__name__.split('.')[-1].replace('_', ' '),
            message='starting', start=True
        )
    if algorithm_name is None:
        algorithm_name = cfg.minimum_distance
    elif algorithm_name not in cfg.classification_algorithms:
        cfg.logger.log.error('unknown algorithm name')
        cfg.messages.error('unknown algorithm name')
        return OutputManager(check=False)
    cfg.progress.update(message='starting the classifier', step=1)
    cfg.logger.log.debug('algorithm_name: %s' % str(algorithm_name))
    # prepare process files
    prepared = shared_tools.prepare_process_files(
        input_bands=input_bands, output_path=output_path, overwrite=overwrite,
        n_processes=n_processes, bandset_catalog=bandset_catalog
    )
    input_raster_list = prepared['input_raster_list']
    out_path = prepared['output_path']
    vrt_r = prepared['virtual_output']
    n_processes = prepared['n_processes']
    if load_classifier is not None:
        loaded_classifier = _load_model(load_classifier)
        if loaded_classifier.check:
            classifier = loaded_classifier.extra['classifier']
        else:
            cfg.logger.log.error('failed loading classifier')
            cfg.messages.error('failed loading classifier')
            return OutputManager(check=False)
    else:
        # collect x and y matrices
        x_y_matrices = _collect_x_y_matrices(
            x_matrix=x_input, y=y_input,
            covariance_matrices=covariance_matrices,
            input_raster_list=input_raster_list,
            spectral_signatures=spectral_signatures,
            input_normalization=input_normalization, macroclass=macroclass,
            n_processes=n_processes, available_ram=available_ram,
            min_progress=1, max_progress=5, algorithm_name=algorithm_name
        )
        x_matrix = x_y_matrices.extra['x']
        y = x_y_matrices.extra['y']
        covariance_matrices_dict = x_y_matrices.extra['covariance_matrices']
        normalization = x_y_matrices.extra['normalization_values']
        # fit classifier
        classifier = Classifier.train(
            spectral_signatures=spectral_signatures,
            algorithm_name=algorithm_name,
            covariance_matrices=covariance_matrices_dict,
            svc_classification_confidence=classification_confidence,
            n_processes=n_processes, available_ram=available_ram,
            cross_validation=cross_validation, x_matrix=x_matrix, y=y,
            class_weight=class_weight, input_normalization=input_normalization,
            normalization_values=normalization,
            find_best_estimator=find_best_estimator,
            rf_max_features=rf_max_features, rf_number_trees=rf_number_trees,
            rf_min_samples_split=rf_min_samples_split, svm_c=svm_c,
            svm_gamma=svm_gamma, svm_kernel=svm_kernel,
            pytorch_model=pytorch_model, pytorch_optimizer=pytorch_optimizer,
            mlp_training_portion=mlp_training_portion,
            pytorch_loss_function=pytorch_loss_function,
            mlp_hidden_layer_sizes=mlp_hidden_layer_sizes, mlp_alpha=mlp_alpha,
            mlp_learning_rate_init=mlp_learning_rate_init,
            mlp_max_iter=mlp_max_iter, mlp_batch_size=mlp_batch_size,
            mlp_activation=mlp_activation,
            pytorch_optimization_n_iter_no_change=(
                pytorch_optimization_n_iter_no_change),
            pytorch_optimization_tol=pytorch_optimization_tol,
            pytorch_device=pytorch_device, min_progress=5, max_progress=40
        )
    # save model
    if not save_classifier:
        output_model = None
    else:
        saved_model = classifier.save_model(output_path=output_path)
        output_model = saved_model.path
    # perform multiprocess classification
    if not only_fit:
        prediction = classifier.run_prediction(
            input_raster_list=input_raster_list, output_raster_path=out_path,
            n_processes=n_processes, available_ram=available_ram,
            macroclass=macroclass,
            threshold=threshold, signature_raster=signature_raster,
            classification_confidence=classification_confidence,
            virtual_raster=vrt_r, min_progress=40, max_progress=90
        )
        if prediction.check:
            cfg.logger.log.info('end; prediction: %s' % str(prediction.path))
            return OutputManager(
                path=prediction.path, extra={'model_path': output_model}
            )
    else:
        cfg.logger.log.info(
            'end; only_fit: %s; output_model: %s' % (
                str(only_fit), str(output_model))
        )
        return OutputManager(
            extra={'classifier': classifier, 'model_path': output_model}
        )


def _perform_cross_validation_scikit(
        cross_validation, classifier, x_matrix, y, n_processes: int = None,
        available_ram: int = None
):
    """Performs cross validation with scikit."""
    if cross_validation and cross_validation is not None:
        # calculate score by cross-validation with stratification
        classifier_list, train_arg_dict_list, test_arg_dict_list = \
            _create_stratified_k_fold_scikit(
                classifier=classifier, x_matrix=x_matrix, y=y
            )
        # fit training
        cfg.multiprocess.run_scikit(
            function=fit_classifier, classifier_list=classifier_list,
            list_train_argument_dictionaries=train_arg_dict_list,
            min_progress=5, max_progress=40, n_processes=n_processes,
            available_ram=available_ram
        )
        classifiers_fitted = cfg.multiprocess.output
        fitted_classifier_list = shared_tools.expand_list(classifiers_fitted)
        # score test
        cfg.multiprocess.run_scikit(
            function=score_classifier, classifier_list=fitted_classifier_list,
            list_train_argument_dictionaries=test_arg_dict_list,
            n_processes=n_processes, available_ram=available_ram
        )
        test_scores = cfg.multiprocess.output
        score_list = shared_tools.expand_list(test_scores)
        scores = np.array(score_list)
        cfg.logger.log.debug(
            'cross validation score mean: %s; cross validation score std: '
            '%s' % (str(scores.mean()), str(scores.std()))
        )
        return scores


def _get_x_y_arrays_from_rois(
        raster_paths, roi_path, spectral_signatures, macroclass=True,
        n_processes: int = None, available_ram: int = None,
        algorithm_name=None, input_normalization=None,
        min_progress=None, max_progress=None
):
    """Gets x y arrays from rois."""
    cfg.logger.log.debug('raster_paths: %s' % str(raster_paths))
    if n_processes is None:
        n_processes = cfg.n_processes
    min_x, max_x, min_y, max_y = raster_vector.get_layer_extent(roi_path)
    virtual_path_list = []
    for p in raster_paths:
        temp_path = cfg.temp.temporary_file_path(name_suffix=cfg.vrt_suffix)
        virtual = raster_vector.create_virtual_raster(
            input_raster_list=[p], output=temp_path,
            box_coordinate_list=[min_x, max_y, max_x, min_y]
        )
        virtual_path_list.append(virtual)
    # get band arrays
    cfg.multiprocess.run_separated(
        raster_path_list=virtual_path_list, function=get_band_arrays,
        function_argument=[[roi_path, spectral_signatures.table]] * len(
            raster_paths
        ), function_variable=virtual_path_list, n_processes=n_processes,
        available_ram=available_ram, keep_output_argument=True,
        progress_message='get band arrays', min_progress=min_progress,
        max_progress=max_progress
    )
    # array for each roi
    cfg.multiprocess.multiprocess_roi_arrays()
    array_dictionary = cfg.multiprocess.output
    x_matrix = y = normalization = means = stds = max_s = min_s = None
    covariance_matrices_dict = {}
    for s in array_dictionary:
        if macroclass:
            class_value = spectral_signatures.table[
                spectral_signatures.table.signature_id == s].macroclass_id[0]
        else:
            class_value = spectral_signatures.table[
                spectral_signatures.table.signature_id == s].class_id[0]
        matrix = np.stack(array_dictionary[s])
        if x_matrix is None:
            x_matrix = matrix
            y = np.ones(matrix.shape[1]) * class_value
        else:
            x_matrix = np.hstack([x_matrix, matrix])
            y = np.concatenate([y, np.ones(matrix.shape[1]) * class_value])
    # normalization
    if input_normalization is not None:
        normalization = []
        means = []
        stds = []
        max_s = []
        min_s = []
        if input_normalization == cfg.z_score:
            for b in range(0, x_matrix.shape[0]):
                mean = x_matrix[b, ::].mean()
                means.append(mean)
                std = x_matrix[b, ::].std()
                stds.append(std)
                expression = '(%s[::, ::, n] - %s) / (%s + 0.000001)' % (
                    cfg.array_function_placeholder, str(mean), str(std))
                x_matrix[b, ::] = (x_matrix[b, ::] - mean) / (std + 0.000001)
                normalization.append(expression)
        elif input_normalization == cfg.linear_scaling:
            for b in range(0, x_matrix.shape[0]):
                minimum = x_matrix[b, ::].min()
                min_s.append(minimum)
                maximum = x_matrix[b, ::].max()
                max_s.append(maximum)
                expression = '(%s[::, ::, n] - %s) / (%s - %s)' % (
                    cfg.array_function_placeholder, str(minimum), str(maximum),
                    str(minimum))
                x_matrix[b, ::] = (x_matrix[b, ::] - minimum) / (
                        maximum - minimum)
                normalization.append(expression)
    if algorithm_name == cfg.maximum_likelihood:
        for s in array_dictionary:
            cfg.logger.log.error('s: %s' % str(s))
            matrix = np.stack(array_dictionary[s])
            # normalization
            if input_normalization is not None:
                if input_normalization == cfg.z_score:
                    for b in range(0, matrix.shape[0]):
                        matrix[b, ::] = (matrix[b, ::] - means[b]) / (
                                stds[b] + 0.000001)
                elif input_normalization == cfg.linear_scaling:
                    for b in range(0, matrix.shape[0]):
                        matrix[b, ::] = (matrix[b, ::] - min_s[b]) / (
                                max_s[b] - min_s[b])
            # covariance matrix (degree of freedom = 1 for unbiased estimate)
            cov_matrix = np.ma.cov(np.ma.masked_invalid(matrix), ddof=1)
            try:
                # inverse
                inv = np.linalg.inv(cov_matrix)
                if np.isnan(inv[0, 0]):
                    pass
                else:
                    covariance_matrices_dict[s] = cov_matrix
            except Exception as err:
                cfg.logger.log.error(str(err))
                cfg.messages.error('covariance matrix')
    cfg.logger.log.error(
        'covariance_matrices_dict: %s'
        % str(covariance_matrices_dict)
    )
    return OutputManager(
        extra={
            'x': x_matrix.T, 'y': y,
            'covariance_matrices': covariance_matrices_dict,
            'normalization_values': [normalization, means, stds, min_s, max_s]
        }
    )


def _collect_x_y_matrices(
        x_matrix=None, y=None, covariance_matrices=None,
        input_raster_list=None, spectral_signatures=None,
        input_normalization=None, macroclass=None, n_processes: int = None,
        available_ram: int = None,
        min_progress=0, max_progress=100, algorithm_name=None
):
    """Collects x and y matrices."""
    normalization_values = None
    if x_matrix is None:
        # calculate x y arrays
        x_y_arrays = _get_x_y_arrays_from_rois(
            raster_paths=input_raster_list,
            roi_path=spectral_signatures.geometry_file,
            spectral_signatures=spectral_signatures, macroclass=macroclass,
            n_processes=n_processes, available_ram=available_ram,
            algorithm_name=algorithm_name,
            input_normalization=input_normalization, min_progress=min_progress,
            max_progress=max_progress
        )
        x_matrix = x_y_arrays.extra['x']
        y = x_y_arrays.extra['y']
        covariance_matrices = x_y_arrays.extra['covariance_matrices']
        normalization_values = x_y_arrays.extra['normalization_values']
    return OutputManager(
        extra={
            'x': x_matrix, 'y': y, 'covariance_matrices': covariance_matrices,
            'normalization_values': normalization_values
        }
    )


def _create_stratified_k_fold_scikit(classifier, x_matrix, y):
    """Creates stratified k fold for classifier with scikit."""
    classifier_list = []
    train_arg_dict_list = []
    test_arg_dict_list = []
    # stratification
    stratification = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, test_index in stratification.split(x_matrix, y):
        classifier_list.append(clone(classifier))
        train_arg_dict_list.append(
            {'X': x_matrix[train_index], 'y': y[train_index]}
        )
        test_arg_dict_list.append(
            {'X': x_matrix[test_index], 'y': y[test_index]}
        )
    return classifier_list, train_arg_dict_list, test_arg_dict_list


def _normalize_signatures(
        spectral_signatures, normalization_values, input_normalization
):
    """Normalizes spectral signatures."""
    if input_normalization is not None:
        x_signatures = deepcopy(spectral_signatures.signatures)
        for s in x_signatures:
            if input_normalization == cfg.z_score:
                x_signatures[s].value = ((x_signatures[s].value - np.array(
                    normalization_values[1]
                )) / (np.array(
                    normalization_values[2]
                ) + 0.000001))
            elif input_normalization == cfg.linear_scaling:
                x_signatures[s].value = (x_signatures[s].value - np.array(
                    normalization_values[3]
                )) / (np.array(
                    normalization_values[4]
                ) - np.array(
                    normalization_values[3]
                ))
        spectral_signatures_catalog = deepcopy(spectral_signatures)
        spectral_signatures_catalog.signatures = x_signatures
    else:
        spectral_signatures_catalog = spectral_signatures
    return spectral_signatures_catalog


def _load_model(model_path):
    """Loads a model."""
    cfg.logger.log.debug('model_path: %s' % str(model_path))
    temp_dir = cfg.temp.create_temporary_directory()
    file_list = files_directories.unzip_file(model_path, temp_dir)
    algorithm_name = framework_name = spectral_signatures = \
        covariance_matrices = model_classifier = None
    input_normalization = normalization_values = None
    # open classification framework
    for f in file_list:
        f_name = files_directories.file_name(f)
        if f_name == cfg.classification_framework:
            classification_framework = read_write_files.open_text_file(f)
            lines = classification_framework.split(cfg.new_line)
            for line in lines:
                variable = line.split('=')
                if variable[0] == cfg.algorithm_name_framework:
                    algorithm_name = variable[1]
                elif variable[0] == cfg.input_normalization_framework:
                    input_normalization = variable[1]
            if input_normalization is not None and \
                    input_normalization.lower() == 'none':
                input_normalization = None
            # scikit framework
            if (
                    algorithm_name == cfg.random_forest or algorithm_name ==
                    cfg.random_forest_ovr or algorithm_name ==
                    cfg.support_vector_machine or algorithm_name ==
                    cfg.multi_layer_perceptron):
                framework_name = cfg.scikit_framework
            # pytorch framework
            elif algorithm_name == cfg.pytorch_multi_layer_perceptron:
                framework_name = cfg.pytorch_framework
        elif f_name == cfg.spectral_signatures_framework:
            spectral_signatures = pickle.load(open(f, 'rb'))
        elif f_name == cfg.normalization_values_framework:
            normalization_values = pickle.load(open(f, 'rb'))
        elif f_name == cfg.covariance_matrices_framework:
            covariance_matrices = pickle.load(open(f, 'rb'))
    # open model files
    for f in file_list:
        f_name = files_directories.file_name(f)
        if f_name == cfg.model_classifier_framework:
            if framework_name == cfg.scikit_framework:
                model_classifier = pickle.load(open(f, 'rb'))
            elif framework_name == cfg.pytorch_framework:
                model_classifier = torch.load(f)
    classifier = Classifier.load_classifier(
        algorithm_name=algorithm_name, spectral_signatures=spectral_signatures,
        covariance_matrices=covariance_matrices,
        model_classifier=model_classifier,
        input_normalization=input_normalization,
        normalization_values=normalization_values
    )
    return OutputManager(extra={'classifier': classifier})
