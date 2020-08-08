import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from hparams import *
import soundfile as sf
from loguru import logger

filter_length = N_FFT
hop_length = HOP_LENGTH
scale = filter_length / hop_length
fourier_basis = np.fft.fft(np.eye(filter_length))

cutoff = int((filter_length / 2 + 1))
fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                           np.imag(fourier_basis[:cutoff, :])])
forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

def stft(input_data):
    num_batches = input_data.size(0)
    num_samples = input_data.size(1)

    input_data = input_data.view(num_batches, 1, num_samples)
    forward_transform = F.conv1d(input_data,
                                 Variable(forward_basis, requires_grad=False),
                                 stride = hop_length,
                                 padding = filter_length)
    cutoff = int((filter_length / 2) + 1)
    real_part = forward_transform[:, :cutoff, :]
    imag_part = forward_transform[:, cutoff:, :]

    magnitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-10)
    phase = torch.atan2(imag_part.data, real_part.data)
    return magnitude, phase


def istft(magnitude, phase):
    forward_transform = None
    scale = filter_length / hop_length
    fourier_basis = np.fft.fft(np.eye(filter_length))

    cutoff = int((filter_length / 2 + 1))
    fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                               np.imag(fourier_basis[:cutoff, :])])
    forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
    inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

    recombine_magnitude_phase = torch.cat([magnitude*torch.cos(phase),
                                           magnitude*torch.sin(phase)], dim=1)

    inverse_transform = F.conv_transpose1d(recombine_magnitude_phase,
                                           Variable(inverse_basis, requires_grad=False),
                                           stride=hop_length,
                                           padding=0)
    inverse_transform = inverse_transform[:, :, filter_length:]
    return inverse_transform


def griffin_lim(spectrogram, n_iter=300, zero_phase=False):
    logger.info(f"starting GL")
    if zero_phase:
        phase = torch.zeros(spectrogram.shape)
    else:
        phase = torch.rand(spectrogram.shape).clamp(-1, 1) * np.pi
    phase_len = phase.shape[-1]
    for i in range(n_iter):
        wav = istft(spectrogram, phase)[:, 0, :]
        garbage_spect, phase = stft(wav)
        phase = phase[:, :, :phase_len]
        garbage_spect = garbage_spect[:, :, :phase_len]

    wav = istft(spectrogram, phase)
    logger.info(f"done!")
    return wav
