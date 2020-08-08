from loguru import logger
from boltons import fileutils
import os
import os.path
from collections import defaultdict
import numpy as np
import torch
import torch.utils.data as data
import random
import soundfile
from hparams import *
from stft.stft import STFT
from add_noise import inject_noise_sample
from tqdm import tqdm, trange


def spect_loader(path, trim_start, return_phase=False, num_samples=16000, crop=True):
    y, sr = soundfile.read(path)

    if crop:
        y = y[trim_start: trim_start + num_samples]  # trim 'trim_start' from start and crop 1 sec
        y = np.hstack((y, np.zeros((num_samples - len(y)))))

    stft = STFT(N_FFT, HOP_LENGTH)
    y = torch.FloatTensor(y).unsqueeze(0)
    spect, phase = stft.transform(y)

    if return_phase:
        return spect, phase

    return spect


class BaseLoader(data.Dataset):
    def __init__(self, root,
                       n_messages=1,
                       n_pairs=100000,
                       transform=None,
                       trim_start=0,
                       num_samples=16000,
                       test=False):
        random.seed(0)
        self.spect_pairs = self.make_pairs_dataset(root, n_messages, n_pairs)
        self.root = root
        self.transform = transform
        self.loader = spect_loader
        self.trim_start = int(trim_start)
        self.num_samples = num_samples
        self.test = test

    def __getitem__(self, index):
        carrier_file, msg_files = self.spect_pairs[index]
        carrier_spect, carrier_phase = self.loader(carrier_file,
                                                   self.trim_start,
                                                   return_phase=True,
                                                   num_samples=self.num_samples)
        msg = [self.loader(msg_file,
                           self.trim_start,
                           return_phase=True,
                           num_samples=self.num_samples)
                           for msg_file in msg_files]
        msg_spects = list(map(lambda x: x[0], msg))
        msg_phases = list(map(lambda x: x[1], msg))

        if self.transform is not None:
            carrier_spect = self.transform(carrier_spect)
            carrier_phase= self.transform(carrier_phase)
            msg_spects = [self.transform(msg_spect) for msg_spect in msg_spects]

        if self.test:
            return carrier_spect, carrier_phase, msg_spects, msg_phases
        else:
            return carrier_spect, carrier_phase, msg_spects

    def __len__(self):
        return len(self.spect_pairs)


class YohoLoader(BaseLoader):
    def __init__(self, root,
                       n_messages=1,
                       n_pairs=100000,
                       transform=None,
                       trim_start=0,
                       num_samples=8000,
                       test=False):
        super(YohoLoader, self).__init__(root,
                                          n_messages,
                                          n_pairs,
                                          transform,
                                          trim_start,
                                          num_samples,
                                          test)

    def make_pairs_dataset(self, path, n_hidden_messages, n_pairs):
        pairs = []
        files_by_speaker = defaultdict(list)
        unfiltered_wav_files = list(fileutils.iter_find_files(path, "*.wav"))
        wav_files = []
        for wav in unfiltered_wav_files:
            # filter out short files
            try:
                if soundfile.read(wav)[0].shape[0] > 3*8000: wav_files.append(wav)
            except:
                pass

        for wav in wav_files:
            speaker = int(wav.split('/')[-3])
            files_by_speaker[speaker].append(wav)

        for i in range(n_pairs):
            speaker = random.sample(files_by_speaker.keys(), 1)[0]
            sampled_files = random.sample(files_by_speaker[speaker], 1+n_hidden_messages)
            carrier_file, hidden_message_files = sampled_files[0], sampled_files[1:]
            pairs.append((carrier_file, hidden_message_files))

        return pairs


class TimitLoader(BaseLoader):
    def __init__(self, root,
                       n_messages=1,
                       n_pairs=100000,
                       transform=None,
                       trim_start=0,
                       num_samples=16000,
                       test=False):
        super(TimitLoader, self).__init__(root,
                                          n_messages,
                                          n_pairs,
                                          transform,
                                          trim_start,
                                          num_samples,
                                          test)

    def make_pairs_dataset(self, path, n_hidden_messages, n_pairs):
        pairs = []
        wav_files = list(fileutils.iter_find_files(path, "*.wav"))

        for i in range(n_pairs):
            sampled_files = random.sample(wav_files, 1+n_hidden_messages)
            carrier_file, hidden_message_files = sampled_files[0], sampled_files[1:]
            pairs.append((carrier_file, hidden_message_files))
        return pairs
