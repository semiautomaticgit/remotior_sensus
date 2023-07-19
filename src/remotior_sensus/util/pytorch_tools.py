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
PyTorch tools
"""

import numpy as np
from sklearn.model_selection import train_test_split

from remotior_sensus.core import configurations as cfg

nn_module = None
try:
    import torch
    from torch import nn
    nn_module = nn.Module
    from torch.utils.data import DataLoader, TensorDataset
except Exception as error:
    # empty class
    class Module:
        pass
    nn_module = Module
    if cfg.logger is not None:
        cfg.logger.log.error(str(error))
    else:
        print(str(error))


class PyTorchNeuralNetwork(nn_module):

    def __init__(
            self, columns, classes_number, hidden_layer_sizes=None,
            activation=None
    ):
        # call parent class
        super(PyTorchNeuralNetwork, self).__init__()
        if hidden_layer_sizes is not None:
            if activation == 'logistic' or activation == 'sigmoid':
                activation_function = nn.Sigmoid()
            elif activation == 'tanh':
                activation_function = nn.Tanh()
            elif activation == 'relu':
                activation_function = nn.ReLU()
            else:
                activation_function = nn.ReLU()
            layers = []
            for x in range(0, len(hidden_layer_sizes)):
                if x == 0:
                    layers.append(nn.Linear(columns, hidden_layer_sizes[x]))
                    layers.append(activation_function)
                else:
                    layers.append(
                        nn.Linear(
                            hidden_layer_sizes[x - 1], hidden_layer_sizes[x]
                        )
                    )
                    layers.append(activation_function)
            layers.append(
                nn.Linear(hidden_layer_sizes[-1], int(classes_number))
            )
            self.sequential = nn.Sequential(*layers)
        else:
            self.sequential = nn.Sequential(
                nn.Linear(columns, 100), nn.ReLU(),
                nn.Linear(100, classes_number)
            )

    # forward
    def forward(self, training_data):
        logits = self.sequential(training_data)
        return logits


# noinspection PyTypeChecker,PyUnresolvedReferences
def train_pytorch_model(
        x_matrix, y_matrix, pytorch_model=None, activation='relu',
        batch_size=None, n_processes=0, training_portion=None,
        pytorch_optimizer=None, hidden_layer_sizes=None,
        loss_function=None, learning_rate_init=None,
        optimization_n_iter_no_change=None, optimization_tol=None,
        weight_decay=None, max_iterations=None, device=None, min_progress=0,
        max_progress=100
):
    cfg.logger.log.debug('start')
    # get device
    if device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    if device == 'cpu':
        torch.set_num_threads(n_processes)
    if optimization_n_iter_no_change is None:
        optimization_n_iter_no_change = 5
    if optimization_tol is None:
        optimization_tol = 0.0001
    if max_iterations is None:
        max_iterations = 200
    if batch_size is None or batch_size == 'auto':
        batch_size = int(
            min(
                2000, min(x_matrix.shape[0], max(200, x_matrix.shape[0] / 100))
            )
        )
    cfg.logger.log.debug(
        'device: %s; n_processes: %s; batch_size: %s' % (
            device, n_processes, batch_size)
    )
    # number of classes
    classes_number = np.max(y_matrix) + 1
    if pytorch_model is None:
        # move model to device
        model = PyTorchNeuralNetwork(
            columns=x_matrix.shape[1], classes_number=classes_number,
            hidden_layer_sizes=hidden_layer_sizes, activation=activation
        ).to(device)
    else:
        model = pytorch_model(
            columns=x_matrix.shape[1], classes_number=classes_number,
            hidden_layer_sizes=hidden_layer_sizes, activation=activation
        ).to(device)
    data_type = eval('torch.%s' % x_matrix.dtype)
    if training_portion is not None and training_portion < 1:
        x_train, x_test, y_train, y_test = train_test_split(
            x_matrix, y_matrix, test_size=1 - training_portion,
            train_size=training_portion, random_state=0,
            stratify=y_matrix
        )
        x_train = torch.tensor(x_train, dtype=data_type)
        y_train = torch.tensor(y_train, dtype=torch.long)
        x_test = torch.tensor(x_test, dtype=data_type)
        y_test = torch.tensor(y_test, dtype=torch.long)
        training_data = DataLoader(
            TensorDataset(x_train, y_train), batch_size=batch_size,
            shuffle=True
        )
        test_data = DataLoader(
            TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=True
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_matrix, y_matrix, test_size=0.1,
            train_size=0.9, random_state=0, stratify=y_matrix
        )
        x_train = torch.tensor(x_matrix, dtype=data_type)
        y_train = torch.tensor(y_matrix, dtype=torch.long)
        x_test = torch.tensor(x_test, dtype=data_type)
        y_test = torch.tensor(y_test, dtype=torch.long)
        training_data = DataLoader(
            TensorDataset(x_train, y_train), batch_size=batch_size,
            shuffle=True
        )
        test_data = DataLoader(
            TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=True
        )
    if learning_rate_init is None:
        learning_rate_init = 0.001
    if weight_decay is None:
        weight_decay = 0.0001
    # optimizer
    if pytorch_optimizer is None:
        pytorch_optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate_init,
            weight_decay=weight_decay
        )
    # loss function
    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()
    epoch = 0
    test_loss_list = []
    while True:
        if cfg.action is True:
            epoch += 1
            # train
            model.train()
            training_size = len(training_data.dataset)
            training_loss = 0
            for batch, (x, y) in enumerate(training_data):
                x, y = x.to(device), y.to(device)
                # compute prediction loss
                prediction = model(x)
                loss = loss_function(prediction, y)
                # backpropagation
                pytorch_optimizer.zero_grad()
                loss.backward()
                pytorch_optimizer.step()
                training_loss += loss.item()
                # progress
                count_progress = batch * len(x)
                cfg.progress.update(
                    percentage=int(100 * count_progress / training_size)
                )
            training_loss /= training_size
            # test
            model.eval()
            test_size = len(test_data.dataset)
            test_loss, correct = 0, 0
            with torch.no_grad():
                for x, y in test_data:
                    x, y = x.to(device), y.to(device)
                    prediction = model(x)
                    test_loss += loss_function(prediction, y).item()
                    correct += (prediction.argmax(1) == y).type(
                        torch.float
                    ).sum().item()
            test_loss /= test_size
            test_loss_list.append(test_loss)
            accuracy = correct / test_size * 100
            # progress
            if max_iterations is not None:
                increment = (max_progress - min_progress) / max_iterations
                step = int(min_progress + epoch * increment)
            else:
                step = None
            cfg.progress.update(
                message=f'epoch: {epoch}, training loss: {training_loss:>8f}, '
                        f'test loss: {test_loss:>8f}, '
                        f'accuracy: {accuracy :>0.1f}%', step=step
            )
            cfg.logger.log.debug(
                f'epoch: {epoch}, training loss: {training_loss:>8f}, '
                f'test loss: {test_loss:>8f}, '
                f'accuracy: {accuracy :>0.1f}%'
            )
            # check optimization tolerance
            if epoch > optimization_n_iter_no_change:
                loss_difference = []
                for o in range(1, optimization_n_iter_no_change + 1):
                    diff = test_loss_list[-1 * o] - test_loss_list[
                        -1 * (o + 1)]
                    loss_difference.append(diff ** 2 < optimization_tol ** 2)
                if all(loss_difference):
                    cfg.logger.log.debug(
                        'optimization_tol: %s' % str(optimization_tol)
                    )
                    break
            if max_iterations is not None and epoch == max_iterations:
                cfg.logger.log.debug('max_iterations: %s'
                                     % str(max_iterations))
                break
        else:
            cfg.logger.log.error('cancel')
            return None, None, None, None
    cfg.logger.log.debug('end')
    return model, training_loss, test_loss, accuracy
