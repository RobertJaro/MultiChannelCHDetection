from typing import List

import torch
from torch import nn


class ProModel(nn.Module):

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
