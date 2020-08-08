from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedBlockBN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=2):
        super(GatedBlockBN, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 2): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        conv = nn.ConvTranspose2d if deconv else nn.Conv2d
        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_conv = nn.BatchNorm2d(c_out)
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_gate = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x1 = self.bn_conv(self.conv(x))
        x2 = torch.sigmoid(self.bn_gate(self.gate(x)))
        out = x1 * x2
        return out

class GatedBlockIN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=2):
        super(GatedBlockIN, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 2): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_conv = nn.InstanceNorm2d(c_out)
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_gate = nn.InstanceNorm2d(c_out)

    def forward(self, x):
        x1 = self.bn_conv(self.conv(x))
        x2 = torch.sigmoid(self.bn_gate(self.gate(x)))
        out = x1 * x2
        return out

class GatedBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=2):
        super(GatedBlock, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 2): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.sigmoid(self.gate(x))
        out = x1 * x2
        return out

class SkipGatedBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=2):
        super(SkipGatedBlock, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 2): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.skip = True if c_in == c_out else False

    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.sigmoid(self.gate(x))
        out = x1 * x2
        if self.skip: out += x
        return out

class ReluBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=2):
        super(ReluBlock, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 2): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        bn = {1: nn.BatchNorm1d,
              2: nn.BatchNorm2d}[conv_dim]
        self.conv = nn.Sequential(
            conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            bn(c_out),
            nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class PrintShapeLayer(nn.Module):
    def __init__(self, str=None):
        super(PrintShapeLayer, self).__init__()
        self.str = str

    def forward(self, input):
        if self.str: logger.debug(f"{self.str}")
        logger.debug(f"{input.shape}")
        return input

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Encoder(nn.Module):
    def __init__(self, conv_dim=1, block_type='normal', n_layers=3):
        super(Encoder, self).__init__()
        block = {'normal': GatedBlock,
                 'skip': SkipGatedBlock,
                 'bn': GatedBlockBN,
                 'in': GatedBlockIN,
                 'relu': ReluBlock}[block_type]

        layers = [block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False)]

        for i in range(n_layers-1):
            layers.append(block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        h = self.main(x)
        return h

class CarrierDecoder(nn.Module):
    def __init__(self, conv_dim, block_type='normal', n_layers=4):
        super(CarrierDecoder, self).__init__()
        block = {'normal': GatedBlock,
                 'skip': SkipGatedBlock,
                 'bn': GatedBlockBN,
                 'in': GatedBlockIN,
                 'relu': ReluBlock}[block_type]

        layers = [block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False)]

        for i in range(n_layers-2):
            layers.append(block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False))

        layers.append(block(c_in=64, c_out=1, kernel_size=1, stride=1, padding=0, deconv=False))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        h = self.main(x)
        return h

class MsgDecoder(nn.Module):
    def __init__(self, conv_dim=1, block_type='normal'):
        super(MsgDecoder, self).__init__()
        block = {'normal': GatedBlock,
                 'skip': SkipGatedBlock,
                 'bn': GatedBlockBN,
                 'in': GatedBlockIN,
                 'relu': ReluBlock}[block_type]

        self.main = nn.Sequential(
                block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=1, kernel_size=3, stride=1, padding=1, deconv=False)
                )

    def forward(self, x):
        h = self.main(x)
        return h

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
                GatedBlockBN(1,16,3,1,1),
                GatedBlockBN(16,32,3,1,1),
                GatedBlockBN(32,64,3,1,1),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
                )
        self.linear = nn.Linear(64,1)

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x = self.conv(x)
        x = x.squeeze(2).squeeze(2)
        x = self.linear(x)
        return x
