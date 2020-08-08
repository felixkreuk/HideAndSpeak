from loguru import logger
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os.path import exists, join
from os import makedirs
from convert_ceps import convert
from model import Encoder, CarrierDecoder, MsgDecoder
from solver import Solver
from collections import defaultdict
from hparams import *

class SolverNMsgCond(Solver):
    def __init__(self, config):
        super(SolverNMsgCond, self).__init__(config)
        print("==> running conditional solver!")

        # ------ create models ------
        self.dec_c_conv_dim = self.n_messages * (self.n_messages+1) + 1 + 64
        # self.dec_c_conv_dim = (self.n_messages+1) * 64
        self.build_models()

        # ------ make parallel ------
        self.enc_c = nn.DataParallel(self.enc_c)
        self.enc_m = nn.DataParallel(self.enc_m)
        self.dec_m = nn.DataParallel(self.dec_m)
        self.dec_c = nn.DataParallel(self.dec_c)

        # ------ create optimizers ------
        params = list(self.enc_m.parameters()) \
               + list(self.enc_c.parameters()) \
               + list(self.dec_c.parameters()) \
               + list(self.dec_m.parameters())
        self.opt = self.opt_type(params, lr=self.lr)
        self.lr_sched = StepLR(self.opt, step_size=20, gamma=0.5)

        # ------ send to cuda ------
        self.enc_c.to(self.device)
        self.enc_m.to(self.device)
        self.dec_m.to(self.device)
        self.dec_c.to(self.device)

        if self.load_ckpt_dir:
            self.load_models(self.load_ckpt_dir)

        logger.debug(self.enc_m)
        logger.debug(self.dec_c)
        logger.debug(self.dec_m)

    def build_models(self):
        super(SolverNMsgCond, self).build_models()

        self.enc_m = Encoder(conv_dim   = 1 + self.n_messages,
                             block_type = self.block_type,
                             n_layers   = self.config.enc_n_layers)

        self.enc_c = Encoder(conv_dim   = 1,
                             block_type = self.block_type,
                             n_layers   = self.config.enc_n_layers)

        self.dec_c = CarrierDecoder(conv_dim   = self.dec_c_conv_dim,
                                    block_type = self.block_type,
                                    n_layers   = self.config.dec_c_n_layers)

        self.dec_m = MsgDecoder(conv_dim   = self.dec_m_conv_dim + self.n_messages,
                                block_type = self.block_type)


    def save_models(self, suffix=''):
        logger.info(f"saving model to: {self.ckpt_dir}\n==> suffix: {suffix}")
        makedirs(join(self.ckpt_dir, suffix), exist_ok=True)
        torch.save(self.enc_c.state_dict(), join(self.ckpt_dir, suffix, "enc_c.ckpt"))
        torch.save(self.enc_m.state_dict(), join(self.ckpt_dir, suffix, "enc_m.ckpt"))
        torch.save(self.dec_c.state_dict(), join(self.ckpt_dir, suffix, "dec_c.ckpt"))
        torch.save(self.dec_m.state_dict(), join(self.ckpt_dir, suffix, "dec_m.ckpt"))

    def load_models(self, ckpt_dir):
        self.enc_c.load_state_dict(torch.load(join(ckpt_dir, "enc_c.ckpt")))
        self.enc_m.load_state_dict(torch.load(join(ckpt_dir, "enc_m.ckpt")))
        self.dec_c.load_state_dict(torch.load(join(ckpt_dir, "dec_c.ckpt")))
        self.dec_m.load_state_dict(torch.load(join(ckpt_dir, "dec_m.ckpt")))
        logger.info("loaded models")

    def reset_grad(self):
        self.opt.zero_grad()

    def train_mode(self):
        super(SolverNMsgCond, self).train_mode()
        self.enc_m.train()
        self.enc_c.train()
        self.dec_c.train()
        self.dec_m.train()

    def eval_mode(self):
        super(SolverNMsgCond, self).eval_mode()
        self.enc_m.train()
        self.enc_c.train()
        self.dec_c.train()
        self.dec_m.train()

    def step(self):
        self.opt.step()
        if self.cur_iter % len(self.train_loader) == 0:
            self.lr_sched.step()

    def incur_loss(self, carrier, carrier_reconst, msg, msg_reconst):
        n_messages = len(msg)
        losses_log = defaultdict(int)
        carrier, msg = carrier.to(self.device), [msg_i.to(self.device) for msg_i in msg]
        all_msg_loss = 0
        carrier_loss = self.reconstruction_loss(carrier_reconst, carrier, type=self.loss_type)
        for i in range(n_messages):
            msg_loss = self.reconstruction_loss(msg_reconst[i], msg[i], type=self.loss_type)
            all_msg_loss += msg_loss
        losses_log['carrier_loss'] = carrier_loss.item()
        losses_log['avg_msg_loss'] = all_msg_loss.item() / self.n_messages
        loss = self.lambda_carrier_loss * carrier_loss + self.lambda_msg_loss * all_msg_loss

        return loss, losses_log

    def forward(self, carrier, carrier_phase, msg):
        assert type(carrier) == torch.Tensor and type(msg) == list
        batch_size = carrier.shape[0]
        carrier, carrier_phase, msg = carrier.to(self.device), carrier_phase.to(self.device), [msg_i.to(self.device) for msg_i in msg]
        msg_encoded_list = []
        msg_reconst_list = []

        # encode carrier
        carrier_enc = self.enc_c(carrier)

        # encoder mesasges
        for i in range(self.n_messages):
            # create one-hot vectors for msg index
            cond = torch.tensor(()).new_full((batch_size,), i)
            cond = self.label2onehot(cond, self.n_messages).to(self.device)
            # concat conditioning vectors to input
            msg_i = self.concat_cond(msg[i], cond)
            msg_encoded_list.append(msg_i)

        # merge encodings and reconstruct carrier
        msg_enc = torch.cat(msg_encoded_list, dim=1)
        merged_enc = torch.cat((carrier, carrier_enc, msg_enc), dim=1)  # concat encodings on features axis
        carrier_reconst = self.dec_c(merged_enc)

        if self.carrier_detach != -1 and self.cur_iter > self.carrier_detach:
            carrier_reconst = carrier_reconst.detach()

        # add stft noise to carrier
        if (self.add_stft_noise != -1 and self.cur_iter > self.add_stft_noise) or self.mode == 'test':
            self.stft.to(self.device)
            y = self.stft.inverse(carrier_reconst.squeeze(1), carrier_phase.squeeze(1))
            carrier_reconst_tag, _ = self.stft.transform(y.squeeze(1))
            carrier_reconst_tag = carrier_reconst_tag.unsqueeze(1)
            self.stft.to('cpu')
        else:
            carrier_reconst_tag = carrier_reconst

        # decode messages from carrier
        for i in range(self.n_messages):
            cond = torch.tensor(()).new_full((batch_size,), i)
            cond = self.label2onehot(cond, self.n_messages).to(self.device)
            msg_reconst = self.dec_m(self.concat_cond(carrier_reconst_tag, cond))
            msg_reconst_list.append(msg_reconst)

        return carrier_reconst, msg_reconst_list

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def concat_cond(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        return torch.cat([x, c], dim=1)
