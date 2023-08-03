from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from functools import partial
from monai.networks.nets import Regressor
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(backbone, model_depth: Union[int, None] = None, **kwargs):
    assert backbone in ["mobilenetv2", "shufflenetv2", "resnet", "monai-regressor"]

    if backbone == "monai-regressor":
        model = Regressor((1, 10, 500, 500), (3,), (2, 4, 8, 16, 32, 64), (2, 2, 2, 2, 2, 2), dropout=0.5)
    elif backbone == "mobilenetv2":
        from models.mobilenetv2 import MobileNetV2
        model = MobileNetV2(**kwargs)
    elif backbone == "shufflenetv2":
        from models.shufflenetv2 import ShuffleNetV2
        model = ShuffleNetV2(**kwargs)
    elif backbone == "resnet":
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]

        if model_depth == 10:
            model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
        elif model_depth == 18:
            model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
        elif model_depth == 34:
            model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
        elif model_depth == 50:
            model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
        elif model_depth == 101:
            model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
        elif model_depth == 152:
            model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
        elif model_depth == 200:
            model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

def logistic(x, a, k, xc):
    a, k, xc = a.to(torch.float), k.to(torch.float), xc.to(torch.float)
    return a / (1 + np.exp(-k * (x - xc)))

def plot_curve_pred(preds, curves):
    preds = preds.cpu()
    for pred, strain, stress in zip(preds, curves["strain"], curves["stress"]):
        x = torch.linspace(1, 1.7, 70)
        l, = plt.plot(x, logistic(x, *pred))

        plt.scatter(strain.cpu(), stress.cpu(), color=l.get_color())
    
    plt.xlabel("Strain")
    plt.xlabel("Stress [MPa]")
    plt.xlim([1, 1.6])

    return plt.gcf()

class Skinstression(pl.LightningModule):
    def __init__(self, backbone: str = "monai-regressor", model_depth: int = 10, variables: int = 3) -> None:
        super().__init__()
        self.model = generate_model(backbone=backbone, model_depth=model_depth, n_classes=variables)
        self.validation_step_outputs_preds = []
        self.validation_step_outputs_strain = []
        self.validation_step_outputs_stress = []
    
    def _common_step(self, batch):
        x, y, curve = batch
        pred = self.model(x)
        loss = nn.functional.l1_loss(pred, y)
        return loss, pred, curve

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, curves = self._common_step(batch)
        self.log("val_loss", loss)
        self.validation_step_outputs_preds.extend(preds)
        self.validation_step_outputs_strain.extend(curves["strain"])
        self.validation_step_outputs_stress.extend(curves["stress"])
    
    def on_validation_epoch_end(self) -> None:
        preds = torch.stack(self.validation_step_outputs_preds)
        strain = torch.stack(self.validation_step_outputs_strain)
        stress = torch.stack(self.validation_step_outputs_stress)
        curves = {"strain": strain, "stress": stress}
        self.logger.experiment.add_figure("Val curves", plot_curve_pred(preds, curves), self.current_epoch)
        self.validation_step_outputs_preds.clear()
        self.validation_step_outputs_strain.clear()
        self.validation_step_outputs_stress.clear()

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, 300, 1e-5)
        return [optimizer], [scheduler]
