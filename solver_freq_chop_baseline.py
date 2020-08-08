from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from solver import Solver
from collections import defaultdict
from hparams import *


class SolverBaseline(Solver):
    def __init__(self, config):
        super(SolverBaseline, self).__init__(config)
        logger.info("running baseline 4k chopping solver!")

    def save_models(self, suffix=''):
        pass

    def load_models(self, ckpt_dir):
        pass

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
        carrier, carrier_phase, msg = carrier.to(self.device), carrier_phase.to(self.device), [msg_i.to(self.device) for msg_i in msg]

        msg = msg[0]  # work just on first msg for this baseline
        mid_freq = int(msg.shape[2]/2)
        carrier_first_half = carrier[:,:,:mid_freq+1,:]  # chop first half of freqs
        msg_first_half = msg[:,:,:mid_freq,:]  # chop first half of freqs
        carrier_reconst = torch.cat([carrier_first_half, msg_first_half], dim=2)  # concat msg freqs above carrier freqs

        self.stft.to(self.device)
        y = self.stft.inverse(carrier_reconst.squeeze(1), carrier_phase.squeeze(1))
        carrier_reconst_tag, _ = self.stft.transform(y.squeeze(1))
        carrier_reconst_tag = carrier_reconst_tag.unsqueeze(1)
        self.stft.to('cpu')

        # decode messages from carrier
        msg_reconst = carrier_reconst_tag[:,:,mid_freq+1:,:]
        msg_padding = torch.zeros(carrier_first_half.shape).to(self.device)
        msg_reconst = torch.cat([msg_reconst, msg_padding], dim=2)

        return carrier_reconst, [msg_reconst]
