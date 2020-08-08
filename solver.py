from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from os.path import exists, join
from os import makedirs
from dataloader import TimitLoader, YohoLoader, spect_loader
from convert_ceps import convert
from model import Encoder, CarrierDecoder, MsgDecoder, Discriminator
from tqdm import tqdm, trange
from hparams import *
from stft.stft import STFT
import os
from os.path import join, basename
from collections import defaultdict
from experiment import Experiment

class Solver(object):
    def __init__(self, config):
        self.config = config
        # optimization hyperparams
        self.lr = config.lr
        self.lambda_carrier_loss = config.lambda_carrier_loss
        self.lambda_msg_loss = config.lambda_msg_loss

        # training config
        self.num_iters = config.num_iters
        self.cur_iter = 0
        self.loss_type = config.loss_type
        self.train_path = config.train_path
        self.val_path = config.val_path
        self.test_path = config.test_path
        self.batch_size = config.batch_size
        self.n_pairs = config.n_pairs
        self.n_messages = config.n_messages
        self.model_type = config.model_type
        self.dataset= config.dataset
        self.trim_start = {'yoho': int(2.0*8000),
                           'timit': int(0.6*16000)}[self.dataset]
        if config.mode == 'sample':
            self.trim_start = 0
        self.num_samples = int({'yoho': AUDIO_LEN * 8000,
                                'timit': AUDIO_LEN * 16000}[self.dataset])
        self.carrier_detach = config.carrier_detach
        self.add_stft_noise = config.add_stft_noise
        self.add_carrier_noise = config.add_carrier_noise
        self.carrier_noise_norm = config.carrier_noise_norm
        self.adv = config.adv
        self.block_type = config.block_type
        self.opt_type = {'adam': torch.optim.Adam,
                         'sgd':  torch.optim.SGD,
                         'rms':  torch.optim.RMSprop}[config.opt]

        # model dimensions
        self.enc_conv_dim     = 16
        self.enc_num_repeat   = 3
        self.dec_c_conv_dim   = self.enc_conv_dim * (2 ** self.enc_num_repeat)
        self.dec_c_num_repeat = self.enc_num_repeat
        self.dec_m_conv_dim   = 1
        self.dec_m_num_repeat = 8

        # create experiment
        self.experiment    = Experiment(config.run_dir, use_comet=False, use_wandb=False)
        self.run_dir       = self.experiment.dir
        self.ckpt_dir      = self.experiment.ckpt_dir
        self.code_dir      = self.experiment.code_dir
        self.load_ckpt_dir = config.load_ckpt
        self.samples_dir   = join(self.run_dir, 'samples')
        self.experiment.save_hparams(config)

        self.num_workers        = config.num_workers
        self.device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_model_every   = config.save_model_every
        self.sample_every       = config.sample_every
        self.print_every        = 10
        self.mode               = 'test'

        self.create_dirs()
        self.load_data()
        self.build_models()
        torch.manual_seed(10)

        self.stft = STFT(N_FFT, HOP_LENGTH)
        self.stft.num_samples = self.num_samples

        torch.autograd.set_detect_anomaly(True)

        # logging
        logger.add(join(self.run_dir, "stdout.log"))
        if self.add_stft_noise == -1:
            logger.warning("not using stft noise in training!")

    def log_losses(self, losses, iteration=None):
        if iteration is None:
            iteration = self.cur_iter

        self.experiment.log_metric(losses, step=iteration)

    def create_dirs(self):
        makedirs(self.samples_dir, exist_ok=True)
        logger.info("created dirs")

    def load_data(self):
        loader = {'yoho': YohoLoader,
                  'timit': TimitLoader}[self.dataset]

        train = loader(self.train_path,
                       n_messages=self.n_messages,
                       n_pairs=self.n_pairs,
                       trim_start=self.trim_start,
                       num_samples=self.num_samples)
        self.train_loader = DataLoader(train,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=self.num_workers)

        val = loader(self.val_path,
                     n_messages=self.n_messages,
                     n_pairs=1000,
                     trim_start=self.trim_start,
                     num_samples=self.num_samples,
                     test=True)
        self.val_loader = DataLoader(val,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers)

        test = loader(self.test_path,
                      n_messages=self.n_messages,
                      n_pairs=1000,
                      trim_start=self.trim_start,
                      num_samples=self.num_samples,
                      test=True)
        self.test_loader = DataLoader(test,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      num_workers=0)

        logger.info(f"loaded train ({len(train)}), val ({len(val)}), test ({len(test)})")

    def build_models(self):
        if self.adv:
            self.discriminator = Discriminator()
            self.discriminator = nn.DataParallel(self.discriminator)
            self.discriminator.to(self.device)
            self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

    def save_models(self, suffix=''):
        raise NotImplementedError

    def load_models(self, ckpt_dir):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def reset_grad(self):
        raise NotImplementedError

    def incur_loss(self, carrier, carrier_reconst, msg, msg_reconst):
        raise NotImplementedError

    def forward(self, carrier, msg):
        raise NotImplementedError

    def train_mode(self):
        logger.debug("train mode")
        self.mode = 'train'

    def eval_mode(self):
        logger.debug("eval mode")
        self.mode = 'test'

    def reconstruction_loss(self, input, target, type='mse'):
        if type == 'mse':
            loss = F.mse_loss(input, target)
        elif type == 'abs':
            loss = F.l1_loss(input, target)
        else:
            logger.error("unsupported loss function! reverting to MSE...")
            loss = F.mse_loss(input, target)
        return loss

    def train(self):
        self.eval_mode()
        #  self.sample_examples(subdir=f"epoch_0")

        # start of training loop
        logger.info("start training...")
        epoch_it = trange(self.num_iters)

        for epoch in epoch_it:
            lr = self.opt.param_groups[0]['lr']
            epoch_it.set_description(f"Epoch {epoch}, LR={lr}")
            epoch_loss = defaultdict(list)
            it = tqdm(self.train_loader)
            self.train_mode()

            # inner epoch loop
            for carrier, carrier_phase, msg in it:
                self.cur_iter += 1
                i = self.cur_iter
                batch_size, _, h, w = carrier.shape

                # feedforward and suffer loss
                carrier_reconst, msg_reconst = self.forward(carrier, carrier_phase, msg)
                loss, losses_log = self.incur_loss(carrier, carrier_reconst, msg, msg_reconst)

                if self.adv:
                    g_target_label_encoded = torch.full((batch_size, 1), 1, device=self.device)
                    d_on_encoded_for_enc = self.discriminator(carrier_reconst)
                    g_loss_adv = F.binary_cross_entropy_with_logits(d_on_encoded_for_enc, g_target_label_encoded)
                    loss += g_loss_adv

                self.reset_grad()
                loss.backward()
                self.step()

                if self.adv:
                    self.discriminator_opt.zero_grad()
                    d_target_label_cover = torch.full((batch_size, 1), 1, device=self.device)
                    d_on_cover = self.discriminator(carrier)
                    d_loss_on_cover = F.binary_cross_entropy_with_logits(d_on_cover, d_target_label_cover)
                    d_loss_on_cover.backward()

                    d_target_label_encoded = torch.full((batch_size, 1), 0, device=self.device)
                    d_on_encoded = self.discriminator(carrier_reconst.detach())
                    d_loss_on_encoded = F.binary_cross_entropy_with_logits(d_on_encoded, d_target_label_encoded)
                    d_loss_on_encoded.backward()
                    self.discriminator_opt.step()

                    losses_log['d_real'] = d_loss_on_cover.item()
                    losses_log['d_fake'] = d_loss_on_encoded.item()
                    losses_log['g_fake'] = g_loss_adv.item()

                # log stuff
                if i % self.print_every == 0:
                    log = f"[{i}/{len(self.train_loader)}]"
                    for loss_name, loss_value in losses_log.items():
                        log += f", {loss_name}: {loss_value:.4f}"
                    it.set_description(log)
                self.log_losses(losses_log, iteration=self.cur_iter)

                # log epoch losses
                for k,v in losses_log.items():
                    epoch_loss[k].append(v)

            # calc epoch stats
            for k,v in list(epoch_loss.items()):
                epoch_loss["epoch_" + k] = np.mean(v)
                epoch_loss.pop(k)
            epoch_loss['lr'] = lr
            self.log_losses(epoch_loss, iteration=epoch)

            # put everything in eval mode for sampling
            self.eval_mode()

            # save model every epoch
            self.save_models(suffix=str(epoch+1) + "_epoch")

            # sample every epoch
            #  self.sample_examples(subdir=f"epoch_{epoch+1}")

            # run validation and log losses
            self.log_losses(self.test(data='val'), iteration=epoch)

        logger.info("finished training!")

    def snr(self, orig, recon):
        N = orig.shape[-1] * orig.shape[-2]
        orig, recon = orig.cpu(), recon.cpu()
        rms1 = ((torch.sum(orig ** 2) / N) ** 0.5)
        rms2 = ((torch.sum((orig - recon) ** 2) / N) ** 0.5)
        snr = 10 * torch.log10((rms1 / rms2) ** 2)
        return snr

    def test(self, data='test'):
        self.eval_mode()

        with torch.no_grad():
            avg_carrier_loss, avg_msg_loss = 0, 0
            carrier_snr_list = []
            msg_snr_list = []

            logger.info(f"phase: {'test' if data == 'test' else 'validation'}")
            data = self.test_loader if data == 'test' else self.val_loader
            # start of training loop
            logger.info("start testing...")
            for carrier, carrier_phase, msg, msg_phase in tqdm(data):
                # feedforward and incur loss
                carrier_reconst, msg_reconst = self.forward(carrier, carrier_phase, msg)
                loss, losses_log = self.incur_loss(carrier, carrier_reconst, msg, msg_reconst)
                avg_carrier_loss += losses_log['carrier_loss']
                avg_msg_loss += losses_log['avg_msg_loss']

                # calculate SnR for msg
                msg_snr = 0
                for m_spect, m_reconst in zip(msg, msg_reconst):
                    msg_snr += self.snr(m_spect, m_reconst)
                msg_snr_list.append(msg_snr / self.n_messages)

                # calculate SnR for carrier
                carrier_snr = self.snr(carrier, carrier_reconst)
                carrier_snr_list.append(carrier_snr)

            logger.info("finished testing!")
            logger.info(f"carrier loss: {avg_carrier_loss/len(data)}")
            logger.info(f"carrier SnR: {np.mean(carrier_snr_list)}")
            logger.info(f"message loss: {avg_msg_loss/len(data)}")
            logger.info(f"message SnR: {np.mean(msg_snr_list)}")

        return {'val epoch carrier loss': avg_carrier_loss/len(data),
                'val epoch msg loss': avg_msg_loss/len(data),
                'val epoch carrier SnR': np.mean(carrier_snr_list),
                'val epoch msg SnR': np.mean(msg_snr_list)}

    def sample_examples(self, n_examples=1, subdir=None):
        if self.mode != 'test':
            logger.warning("generating audio not in test mode!")

        examples_dir = self.samples_dir
        if subdir is not None:
            examples_dir = join(examples_dir, subdir)
        makedirs(examples_dir, exist_ok=True)

        logger.debug(f"generating {n_examples} examples in '{subdir}'")
        for i in range(n_examples):
            examples_subdir = join(examples_dir, f'{i}')
            makedirs(examples_subdir, exist_ok=True)
            carrier_path, msg_path = self.val_loader.dataset.spect_pairs[i]
            convert(self,
                    carrier_path,
                    msg_path,
                    trg_dir=examples_subdir,
                    epoch=i,
                    trim_start=self.trim_start,
                    num_samples=self.num_samples)
        logger.debug("done")
