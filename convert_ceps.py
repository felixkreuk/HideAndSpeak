from loguru import logger
from os.path import join, basename
from hparams import *
from stft.stft import STFT
from dataloader import spect_loader
import soundfile as sf
from griffin_lim import griffin_lim

def convert(solver, carrier_wav_path, msg_wav_paths, trg_dir, epoch, trim_start, num_samples):
    if solver.mode != 'test':
        logger.warning("generating audio not in test mode!")

    _, sr = sf.read(carrier_wav_path)
    carrier_basename = basename(carrier_wav_path).split(".")[0]
    msg_basenames = [basename(msg_wav_path).split(".")[0] for msg_wav_path in msg_wav_paths]

    spect_carrier, phase_carrier = spect_loader(carrier_wav_path, trim_start, return_phase=True, num_samples=num_samples)
    spect_carrier, phase_carrier = spect_carrier.unsqueeze(0), phase_carrier.unsqueeze(0)
    magphase_msg = [spect_loader(path, trim_start, return_phase=True, num_samples=num_samples) for path in msg_wav_paths]
    spects_msg, phases_msg = [D[0].unsqueeze(0) for D in magphase_msg], [D[1].unsqueeze(0) for D in magphase_msg]

    spect_carrier = spect_carrier.to('cuda')
    spects_msg = [spect_msg.to('cuda') for spect_msg in spects_msg]
    spect_carrier_reconst, spects_msg_reconst = solver.forward(spect_carrier, phase_carrier, spects_msg)
    spect_carrier_reconst = spect_carrier_reconst.cpu().squeeze(0)
    spects_msg_reconst = [spect_msg_reconst.cpu().squeeze(0) for spect_msg_reconst in spects_msg_reconst]

    stft = STFT(N_FFT, HOP_LENGTH)
    out_carrier = stft.inverse(spect_carrier_reconst, phase_carrier.squeeze(0)).squeeze(0).squeeze(0).detach().numpy()
    orig_out_carrier = stft.inverse(spect_carrier.cpu().squeeze(0), phase_carrier.squeeze(0)).squeeze(0).squeeze(0).detach().numpy()

    outs_msg = [stft.inverse(spect_msg_reconst, phase_msg.squeeze(0)).squeeze(0).squeeze(0).detach().numpy() for spect_msg_reconst, phase_msg in zip(spects_msg_reconst, phases_msg)]
    orig_outs_msg = [stft.inverse(spect_msg.cpu().squeeze(0), phase_msg.squeeze(0)).squeeze(0).squeeze(0).detach().numpy() for spect_msg, phase_msg in zip(spects_msg, phases_msg)]
    outs_msg_gl = [griffin_lim(m.cpu(), n_iter=50)[0, 0].detach().numpy() for m in spects_msg_reconst]

    sf.write(join(trg_dir, f'{epoch}_{carrier_basename}_carrier_embedded.wav'), out_carrier, sr)
    sf.write(join(trg_dir, f'{epoch}_{carrier_basename}_carrier_orig.wav'), orig_out_carrier, sr)
    for i in range(len(outs_msg)):
        sf.write(join(trg_dir, f'{epoch}_{msg_basenames[i]}_msg_recovered_orig_phase.wav'), outs_msg[i], sr)
        sf.write(join(trg_dir, f'{epoch}_{msg_basenames[i]}_msg_orig.wav'), orig_outs_msg[i], sr)
        sf.write(join(trg_dir, f'{epoch}_{msg_basenames[i]}_msg_recovered_gl_phase.wav'), outs_msg_gl[i], sr)
