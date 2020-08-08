import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from hparams import *

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class STFT(torch.nn.Module, metaclass=Singleton):
    def __init__(self, filter_length=1024, hop_length=512):
        super(STFT, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        input_data = input_data.view(num_batches, 1, num_samples)
        forward_transform = F.conv1d(input_data,
                                     Variable(self.forward_basis, requires_grad=False),
                                     stride = self.hop_length,
                                     padding = self.filter_length)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-10)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude*torch.cos(phase),
                                               magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase,
                                               Variable(self.inverse_basis, requires_grad=False),
                                               stride=self.hop_length,
                                               padding=0)
        inverse_transform = inverse_transform[:, :, self.filter_length:]
        inverse_transform = inverse_transform[:, :, :self.num_samples]
        return inverse_transform
