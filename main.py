import argparse
from solver import Solver
from solver_n_msg_conditional import SolverNMsgCond
from solver_n_msg_deepsteg import SolverNMsgMultipleDecodersDeepSteg
from solver_freq_chop_baseline import SolverBaseline
import torch
torch.manual_seed(0)

def main(config):
    torch.set_num_threads(1000)
    if config.model_type == 'n_msg':
        solver = SolverNMsgMultipleDecodersDeepSteg(config)
    elif config.model_type == 'n_msg_cond':
        solver = SolverNMsgCond(config)
    elif config.model_type == 'baseline':
        solver = SolverBaseline(config)
    else:
        print("dataset type not supported!")
        return -1
    if config.mode == 'train':
        solver.train()
        solver.test()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'sample':
        solver.eval_mode()
        solver.sample_examples()

if __name__ == '__main__':
    tunable = True
    parser = argparse.ArgumentParser(description='Hide and Speak')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--num_iters', type=int, default=100, help='number of epochs')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'abs'], help='loss function used for training')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'sample'], help='`train` will initiate training, `test` should be used in conjunction with `load_ckpt` to run a test epoch, `sample` should be used in conjunction with `load_ckpt` to sample examples from dataset')
    parser.add_argument('--train_path', required=True, type=str, help='path to training set. should be a folder containing .wav files for training')
    parser.add_argument('--val_path', required=True, type=str, help='')
    parser.add_argument('--test_path', required=True, type=str, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--n_pairs', type=int, default=50000, help='number of training examples generated from wav files')
    parser.add_argument('--n_messages', type=int, default=1, help='number of hidden messages')
    parser.add_argument('--dataset', type=str, default='timit', help='select dataset', choices=['timit', 'yoho'])
    parser.add_argument('--model_type', type=str, default='n_msg', help='`n_msg` default model type, `n_msg_cond` conditional message decoding, `baseline` is the frequency-chop baseline', choices=['n_msg', 'n_msg_cond', 'baseline'])
    parser.add_argument('--carrier_detach', default=-1, type=int, help='flag that stops gradients from the generated carrier and back. if -1 will not be used, if set to k!=-1 then gradients will be stopped from the kth iteration (used for fine-tuning the message decoder)')
    parser.add_argument('--add_stft_noise', default=-1, type=int, help='flag that trasforms the generated carrier spectrogram back to the time domain to simulate real-world conditions. if -1 will not be used, if set to k!=-1 will be used from the kth iteration')
    parser.add_argument('--add_carrier_noise', default=None, type=str, choices=['gaussian', 'snp', 'salt', 'pepper', 'speckle'], help='add different types of noise the the carrier spectrogram')
    parser.add_argument('--carrier_noise_norm', default=0.0, type=float, help='strength of carrier noise')
    parser.add_argument('--adv', action='store_true',default=False, help='flag that indicates if adversarial training should be used')

    parser.add_argument('--block_type', type=str, default='normal', choices=['normal', 'skip', 'bn', 'in', 'relu'], help='type of block for encoder/decoder')
    parser.add_argument('--enc_n_layers', default=3, type=int, help='number of layers in encoder')
    parser.add_argument('--dec_c_n_layers', default=4, type=int, help='number of layers in decoder')
    parser.add_argument('--lambda_carrier_loss', type=float, default=3.0, help='coefficient for carrier loss term')
    parser.add_argument('--lambda_msg_loss', type=float, default=1.0, help='coefficient for message loss term')

    parser.add_argument('--num_workers', type=int, default=20, help='number of data loading workers')
    parser.add_argument('--load_ckpt', type=str, default=None, help='path to checkpoint (used for test epoch or for sampling)')
    parser.add_argument('--run_dir', type=str, default='.', help='output directory for logs, samples and checkpoints')
    parser.add_argument('--save_model_every', type=int, default=None, help='')
    parser.add_argument('--sample_every', type=int, default=None, help='')
    args = parser.parse_args()

    main(args)
