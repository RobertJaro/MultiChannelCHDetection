import logging
import os
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import BCELoss
from torch.utils.data import DataLoader

from chronnos.data.convert import sdo_cmaps
from chronnos.data.generator import CombinedCHDataset, MapDataset, MaskDataset, getDataSet
from chronnos.train.callback import PlotCallback, ValidationCallback


class Trainer:

    def __init__(self, base_path, data_path, device=None, channels=None, **training_args):
        """Trainer for the CHRONNOS model

        :param str base_path: path for the training results
        :param str data_path: path to the converted data
        :param device: None to use the available device
        :param channels: Subset of channels to use for model training
        :param training_args: arguments for the model training, requires 'n_dims', 'n_convs', 'image_channels', 'start_resolution'
        """
        self.base_path = base_path
        self.ds_path = data_path
        self.training_args = training_args
        self.channels = channels

        os.makedirs(base_path, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        logging.info('Using device: %s' % self.device)
        self.model = CHRONNOS(training_args['n_dims'], training_args['n_convs'], training_args['image_channels'], 1,
                              dropout=0.2)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3, betas=(0., 0.99), weight_decay=1e-8)

        self.criterion = BCELoss(reduction='none')
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma=0.5)
        self.validation_callback = ValidationCallback(path=base_path)

    def train(self):
        logging.info('====================== START TRAINING CORE ======================')
        start_resolution = self.training_args['start_resolution']
        self.model.to(self.device)
        self._trainStep(start_resolution)
        for i in range(1, len(self.training_args['n_dims'])):
            self.lr_scheduler.step()
            resolution = start_resolution * 2 ** i
            logging.info('====================== START FADE IN %04d ======================' % (resolution))
            # enter fade in mode
            self.model.createFadeIn()
            self.model.to(self.device)
            self.opt.add_param_group({'params': list(self.model.down_block.parameters()) +
                                                list(self.model.up_block.parameters()) +
                                                list(self.model.to_mask_fade.parameters()) +
                                                list(self.model.from_image_fade.parameters())})
            self._trainStep(resolution)

            logging.info('====================== START FIXED %04d ======================' % (resolution))
            self.model.createFixed()
            self._trainStep(resolution)
        torch.save(self.model, os.path.join(self.base_path, 'final_model.pt'))

    def _trainStep(self, resolution_id):
        fade_flag = "fade" if self.model.fade is True else "non-fade"
        batch_size = 512 // resolution_id
        epochs = 5120 // resolution_id
        epochs = 100 if epochs > 100 else epochs

        save_model_path = os.path.join(self.base_path, "ch_%04d_%s_model.pt" % (resolution_id, fade_flag))
        # continue training if step already complete
        if os.path.exists(save_model_path):
            state_dict = torch.load(save_model_path)
            self.model.load_state_dict(state_dict['m'])
            self.opt.load_state_dict(state_dict['o'])
            self.validation_callback.history = state_dict['history']
            return
        # Data Set Generator
        train_files_map, train_files_mask, valid_files_map, valid_files_mask = getDataSet(self.ds_path, resolution_id)
        train_ds = CombinedCHDataset(train_files_map, train_files_mask, channel=self.channels)
        valid_map_ds = MapDataset(valid_files_map, channel=self.channels)
        valid_mask_ds = MaskDataset(valid_files_mask)

        # compute norm
        train_mask_ds = MaskDataset(train_files_mask)
        masks = np.array([m for m in train_mask_ds])
        norm = np.sum(masks) / np.sum(1 - masks)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)

        valid_loader_map = DataLoader(valid_map_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        valid_loader_mask = DataLoader(valid_mask_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        pred_callback = PlotCallback(data=[(valid_map_ds[i], valid_mask_ds[i]) for i in
                                           range(0, len(valid_map_ds), len(valid_map_ds) // 10)],
                                     prefix="%04d_%s" % (resolution_id, fade_flag),
                                     path=self.base_path,
                                     model=self.model, cmaps=np.array(sdo_cmaps)[self.channels] if self.channels else None)
        self.model.eval()
        with torch.no_grad():
            pred_callback.call(-1, .0)
        epoch_alphas = np.linspace(0, 1, epochs, endpoint=False)
        for epoch, epoch_alpha in enumerate(epoch_alphas):
            alphas = np.linspace(epoch_alpha, epoch_alpha + np.diff(epoch_alphas)[0], len(train_loader))
            alpha = epoch_alpha
            self.model.train()
            for (x, y), alpha in zip(train_loader, alphas):
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad()
                if self.model.fade:
                    y_pred = self.model.forwardFadeIn(x, alpha)
                else:
                    y_pred = self.model.forward(x)
                loss = self.criterion(y_pred, y)
                loss = loss * y + loss * (1 - y) * norm  # weight 10:1
                loss = torch.mean(loss)
                loss.backward()
                self.opt.step()
            self.model.eval()
            with torch.no_grad():
                if (epoch + 1) % (epochs // 10) == 0 or epoch == 0:
                    pred_callback.call(epoch, alpha)

                iou, acc = self.validation_callback.call(self.model, valid_loader_map, valid_loader_mask, alpha)
            logging.info(
                'EPOCH %03d/%03d [IOU %.03f] [ACC %.03f]' % (epoch + 1, epochs, iou, acc))
        torch.save(
            {'m': self.model.state_dict(), 'o': self.opt.state_dict(), 'history': self.validation_callback.history},
            save_model_path)


class CHRONNOS(nn.Module):

    def __init__(self, n_dims: List, n_convs: List, image_channels, n_classes, dropout=0.2):
        super().__init__()
        self.image_channels = image_channels
        self.n_classes = n_classes
        self.step = 0
        self.n_dims = n_dims
        self.n_convs = n_convs
        self.dropout = dropout
        self.fade = False

        self.from_image = FromImageModel(image_channels, n_dims[self.step])
        self.down_sample = nn.AvgPool2d(2)
        self.core = CoreModel(n_dims[self.step], n_convs[self.step], dropout=self.dropout)
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)
        self.to_mask = ToMaskModel(n_dims[self.step], n_classes)

    def createFadeIn(self):
        assert self.fade == False, 'Model is already in fade in mode. Use createFixed first.'
        self.step += 1
        self.fade = True
        self.from_image_fade = FromImageModel(self.image_channels, self.n_dims[self.step])
        self.down_block = DownBlock(self.n_dims[self.step], self.n_dims[self.step - 1], self.n_convs[self.step],
                                    self.dropout)

        self.to_mask_fade = ToMaskModel(self.n_dims[self.step], self.n_classes)
        self.up_block = UpBlock(self.n_dims[self.step - 1], self.n_dims[self.step], self.n_convs[self.step],
                                dropout=self.dropout)

    def createFixed(self):
        assert self.fade is True, 'Model is already in fixed mode. Use createFadeIn first.'
        self.fade = False
        self.from_image = self.from_image_fade
        self.to_mask = self.to_mask_fade
        self.core = CombinedCoreModel(self.down_block, self.core, self.up_block)

    def forwardFadeIn(self, image, alpha):
        assert self.fade == True, 'Model currently in fade in mode. Use forward instead.'
        x_new = self.from_image_fade(image)
        x_new, skip = self.down_block(x_new)
        x_new *= alpha

        x_old = self.down_sample(image)
        x_old = self.from_image(x_old)
        x_old *= (1 - alpha)

        x = x_old + x_new
        x = self.core(x)

        x_new = self.up_block(x, skip)
        x_new = self.to_mask_fade(x_new)
        x_new = torch.mul(x_new, alpha)

        x_old = self.to_mask(x)
        x_old = self.up_sample(x_old)
        x_old = torch.mul(x_old, (1 - alpha))

        return x_old + x_new

    def forward(self, image):
        assert self.fade is False, 'Model currently in fixed mode. Use forwardFadeIn instead.'
        x = self.from_image(image)
        x = self.core(x)
        x = self.to_mask(x)
        return x

    def encode(self, image):
        assert self.fade is False, 'Only available in fixed mode!'
        x = self.from_image(image)
        core = self.core
        skips = []
        while isinstance(core, CombinedCoreModel):
            x, skip = core.down_block(x)
            skips.append(skip)
            core = core.core
        x = core.forward(x)
        return x, skips


class SCAN(nn.Module):

    def __init__(self, channel_models, n_classes, n_dims,
                 n_convs, dropout=0.2):
        super().__init__()
        self.fade = False

        self.channels = len(channel_models)
        self.merge_layers = [nn.Conv2d(n_dims[i] * self.channels, n_dims[i], 1, bias=False) for i in range(len(n_dims))]

        self.models = channel_models
        self.up_blocks = [UpBlock(n_dims[i - 1], n_dims[i], n_convs[i - 1], dropout=dropout) for i in
                          range(1, len(n_dims))]
        self.to_img = ToMaskModel(n_dims[-1], n_classes)

        self.module_list = nn.ModuleList([*self.models, *self.up_blocks, self.to_img, *self.merge_layers])
        self.train_modules = nn.ModuleList([*self.up_blocks, self.to_img, *self.merge_layers])

    def forward(self, image):
        encoded_features = []
        models_skips = []
        with torch.no_grad():
            for i in range(self.channels):
                x = image[:, i:i + 1]
                x, s = self.models[i].encode(x)
                encoded_features.append(x)
                models_skips.append(s)

        x = torch.cat(encoded_features, 1)
        x = self.merge_layers[0](x)

        for i in range(len(self.up_blocks)):
            up_block, merge_layer = self.up_blocks[i], self.merge_layers[i + 1]
            skip = torch.cat([skips[-(i + 1)] for skips in models_skips], dim=1)
            skip = merge_layer(skip)
            x = up_block(x, skip)

        x = self.to_img(x)
        return x

    # for evaluation
    def forwardCombined(self, image):
        combined_mask = self.forward(image)
        channel_masks = [m.forward(image[:, i:i + 1]) for i, m in enumerate(self.models)]
        return [combined_mask, *channel_masks]


class DownBlock(nn.Module):

    def __init__(self, dim, out_dim, n_convs, dropout=0.2):
        super().__init__()
        assert n_convs >= 1, 'Invalid configuration, requires at least 1 convolution block'
        self.convs = nn.Sequential(*[Conv2dBlock(dim, dim, dropout=dropout) for _ in range(n_convs)])
        self.down = Conv2dBlock(dim, out_dim, stride=2)
        self.module_list = nn.ModuleList([self.convs, self.down])

    def forward(self, x):
        x = self.convs(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):

    def __init__(self, dim, out_dim, n_convs, n_skip=1, dropout=0.2):
        super().__init__()
        assert n_convs >= 1, 'Invalid configuration, requires at least 1 convolution block'
        self.up = ConvTranspose2dBlock(dim, out_dim, stride=2)
        self.convs = nn.Sequential(
            *[Conv2dBlock(out_dim * (1 + n_skip) if i == 0 else out_dim, out_dim, dropout=dropout) for i in
              range(n_convs)])
        self.module_list = nn.ModuleList([self.up, self.convs])

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.convs(x)
        return x


class FromImageModel(nn.Module):

    def __init__(self, image_channels, dim):
        super().__init__()
        self.conv = Conv2dBlock(image_channels, dim)

    def forward(self, x):
        return self.conv(x)


class ToMaskModel(nn.Module):

    def __init__(self, dim, n_classes):
        super().__init__()
        self.conv = Conv2dBlock(dim, n_classes, activation='sigmoid', use_norm=False)

    def forward(self, x):
        return self.conv(x)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1,
                 padding=1, activation='leaky_relu', use_bias=True, use_norm=True, dropout=0.):
        super().__init__()
        self.use_bias = use_bias
        self.pad = nn.ReflectionPad2d(padding)
        self.norm = nn.BatchNorm2d(output_dim) if use_norm else None
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0. else None

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=4, stride=2,
                 padding=1, activation='leaky_relu', use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.norm = nn.BatchNorm2d(output_dim)
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        else:
            self.activation = None
        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class CoreModel(nn.Module):

    def __init__(self, dim, n_convs, dropout=0.2):
        super().__init__()
        assert n_convs > 1, 'Invalid configuration, requires at least 1 convolution block'
        module = [Conv2dBlock(dim, dim, dropout=dropout) for _ in range(n_convs)]
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)


class CombinedCoreModel(nn.Module):

    def __init__(self, down_block, core, up_block):
        super().__init__()
        self.down_block = down_block
        self.core = core
        self.up_block = up_block

    def forward(self, x):
        x, skip = self.down_block(x)
        x = self.core(x)
        x = self.up_block(x, skip)
        return x