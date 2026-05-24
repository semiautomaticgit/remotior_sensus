# Remotior Sensus , software to process remote sensing and GIS data.
# Copyright (C) 2022-2026 Luca Congedo.
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
from collections import OrderedDict

try:
    # noinspection PyPackageRequirements
    from sklearn.model_selection import train_test_split
except Exception as error:
    str(error)
    train_test_split = None

from remotior_sensus.core import configurations as cfg

nn_module = None
try:
    import torch
    from torch import nn
    import torch.nn.functional as functional
    nn_module = nn.Module
    from torch.utils.data import DataLoader, TensorDataset
except Exception as error:
    # empty class
    class Module:
        pass


    nn_module = Module
    if cfg.logger is not None:
        cfg.logger.log.error(str(error))

try:
    # noinspection PyPackageRequirements
    from torchvision.models import swin_v2_b, swin_v2_t
    # noinspection PyPackageRequirements
    from torchvision.models.feature_extraction import create_feature_extractor
    # noinspection PyPackageRequirements
    from torchvision.ops import FeaturePyramidNetwork
except Exception as error:
    str(error)


try:
    from .model_implementation import SatlasSegmentationModel, SSrRrDBNet
except Exception as error:
    str(error)


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
            for x, hidden_layer_x in enumerate(hidden_layer_sizes):
                if x == 0:
                    layers.append(nn.Linear(columns, hidden_layer_x))
                    # noinspection PyUnresolvedReferences
                    layers.append(activation_function)
                else:
                    layers.append(
                        nn.Linear(
                            hidden_layer_sizes[x - 1], hidden_layer_x
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


# noinspection PyTypeChecker,PyUnresolvedReferences,PyCallingNonCallable
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
    # convert to float as required
    if x_matrix.dtype != np.float32 and x_matrix.dtype != np.float64:
        x_matrix = x_matrix.astype(np.float32)
        y_matrix = y_matrix.astype(np.float32)
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
        if cfg.action:
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
                        f'accuracy: {accuracy:>0.1f}%', step=step
            )
            cfg.logger.log.debug(
                f'epoch: {epoch}, training loss: {training_loss:>8f}, '
                f'test loss: {test_loss:>8f}, '
                f'accuracy: {accuracy:>0.1f}%'
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


# implementation of pretrained swin_v2 model
def pretrained_pytorch_model_swin2(
        array_dic, weights_path, band_number, variant=None,
        zone_raster_dic=None, pytorch_device=None, stack=True, seed = None,
        n_processes=1
):
    cfg.logger.log.debug('pretrained_pytorch_model_swin2')
    if pytorch_device is None:
        pytorch_device = 'cpu'
    if pytorch_device == 'cpu':
        torch.set_num_threads(n_processes)
    # set fixed seed
    seed = 0 if seed is None else seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = None
    if variant is None or variant == cfg.variant_base:
        model = swin_v2_b()
        model.features[0][0] = nn.Conv2d(
            in_channels=band_number, out_channels=128, kernel_size=(4, 4),
            stride=(4, 4)
        )
    elif variant == cfg.variant_tiny:
        model = swin_v2_t()
        model.features[0][0] = nn.Conv2d(
            in_channels=band_number, out_channels=96, kernel_size=(4, 4),
            stride=(4, 4)
        )
    # load weights
    state_dict = torch.load(weights_path, map_location=pytorch_device,
                            weights_only=True)
    # adapt keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.backbone.'):
            new_key = k.replace('backbone.backbone.', '')
        else:
            new_key = k
        new_state_dict[new_key] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)
    model = model.to(pytorch_device).eval()
    # get stage 0 features
    extractor = create_feature_extractor(
        model, return_nodes={'features.0': 'last_feats'}
    )
    extractor.eval()
    results = {}
    with torch.no_grad():
        for s in array_dic:
            if stack:
                bands = np.stack(array_dic[s], axis=0)
            else:
                bands = np.transpose(array_dic[s], (2, 0, 1))
            # (Batch, Channels, Height, Width)
            input_tensor = torch.from_numpy(bands).unsqueeze(0).to(
                pytorch_device)
            # (B, H, W, C)
            feats = extractor(input_tensor)['last_feats']
            # (B, C, H, W)
            feats = feats.permute(0, 3, 1, 2)
            # interpolate to original resolution
            feats = nn.functional.interpolate(
                feats, size=(bands.shape[1], bands.shape[2])
            )
            # get first batch (H, W, C)
            feats = feats.permute(0, 2, 3, 1).cpu().numpy()[0]
            # mask raster zone
            if zone_raster_dic is not None:
                feats = feats[zone_raster_dic[s] == 1]
            results[s] = feats
    # ravel for training
    if zone_raster_dic is not None:
        results_ravel = np.concatenate(list(results.values()), axis=0)
        results = results_ravel.T
    return results


# segmentation from pretrained swin_v2 model
def segmentation_pytorch_model_swin2(
        array_dic, weights_path, band_number, variant=None,
        pytorch_device=None, stack=True, seed = None, num_classes=None,
        replace_keys: list = None, n_processes=1
):
    cfg.logger.log.debug('segmentation_pytorch_model_swin2')
    if pytorch_device is None:
        pytorch_device = 'cpu'
    if pytorch_device == 'cpu':
        torch.set_num_threads(n_processes)
    # set fixed seed
    seed = 0 if seed is None else seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SatlasSegmentationModel(variant, num_classes=num_classes,
                                    band_number=band_number)
    # load model
    state_dict = torch.load(weights_path, map_location=pytorch_device)
    # adapt keys
    if replace_keys:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith(replace_keys[0]):
                new_key = k.replace(replace_keys[0], replace_keys[1])
            else:
                new_key = k
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=True)
    model = model.to(pytorch_device).eval()
    results = {}
    with torch.inference_mode():
        for s in array_dic:
            if stack:
                bands = np.stack(array_dic[s], axis=0)
            else:
                bands = np.transpose(array_dic[s], (2, 0, 1))
            # (Batch, Channels, Height, Width)
            input_tensor = torch.from_numpy(bands).unsqueeze(0).to(
                pytorch_device)
            # (B, H, W, C)
            feats, _ = model(input_tensor)
            feats = nn.functional.interpolate(
                feats, size=(bands.shape[1], bands.shape[2]),
                mode='bilinear',  align_corners=False
            )
            feats = torch.argmax(feats, dim=1)
            results[s] = feats.squeeze(0).cpu().numpy()
    return results


# superresolution model
def superresolution_pytorch_model_s2(
        array_dic, weights_path, band_number,
        pytorch_device=None, stack=True, seed=None, n_processes=1
):
    if pytorch_device is None:
        pytorch_device = 'cpu'

    if pytorch_device == 'cpu':
        torch.set_num_threads(n_processes)

    # fixed seed
    seed = 0 if seed is None else seed
    torch.manual_seed(seed)
    if pytorch_device != 'cpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    patch_size = 32
    scale = 4
    batch_size = 32

    # ---- Build model ----
    model = SSrRrDBNet(
        num_in_ch=band_number,
        num_out_ch=band_number,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale
    )

    # ---- Load weights (same logic as your segmentation) ----
    state_dict = torch.load(weights_path, map_location=pytorch_device)
    state_dict = state_dict['params_ema']
    model.load_state_dict(state_dict, strict=True)

    model = model.to(pytorch_device).eval()

    results = []

    with torch.inference_mode():
        for s in array_dic:
            # ---- Stack bands (same pattern as your function) ----
            if stack:
                bands = np.stack(array_dic[s], axis=0)  # (C, H, W)
            else:
                bands = np.transpose(array_dic[s], (2, 0, 1))

            C, H, W = bands.shape

            # ---- Normalize (VERY important for RRDB models) ----
            bands = bands.astype(np.float32)
            # adjust if your training used different scaling
            bands /= 255.0

            input_tensor = torch.from_numpy(bands).unsqueeze(0).to(
                pytorch_device)

            # ---- Pad to multiple of patch_size ----
            pad_h = (patch_size - H % patch_size) % patch_size
            pad_w = (patch_size - W % patch_size) % patch_size
            input_tensor = functional.pad(
                input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            _, _, Hp, Wp = input_tensor.shape

            # ---- Unfold into 32x32 patches ----
            patches = functional.unfold(
                input_tensor,
                kernel_size=patch_size,
                stride=patch_size
            )  # (1, C*P*P, N)

            patches = patches.permute(0, 2, 1).reshape(-1, C, patch_size,
                                                       patch_size)

            # ---- Batched SR inference ----
            sr_chunks = []
            for i in range(0, patches.size(0), batch_size):
                sr_chunks.append(model(patches[i:i + batch_size]))
            sr_patches = torch.cat(sr_chunks, dim=0)

            # ---- Fold back (scaled patches) ----
            out_patch = patch_size * scale
            sr_patches = sr_patches.reshape(
                1, -1, out_patch * out_patch * C).permute(0, 2, 1)

            sr_tensor = functional.fold(
                sr_patches,
                output_size=(Hp * scale, Wp * scale),
                kernel_size=out_patch,
                stride=out_patch
            )

            # ---- Crop padding & convert to numpy ----
            sr_tensor = sr_tensor[:, :, :H * scale, :W * scale]
            sr_np = sr_tensor.squeeze(0).cpu().numpy()

            # de-normalize if needed
            sr_np = np.clip(sr_np * 255.0, 0, 255).astype(np.float32)

            results.append(sr_np)
    cube = np.stack(results, axis=0)
    return cube[0]
