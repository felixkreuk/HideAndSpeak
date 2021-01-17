# Hide and Speak: Towards Deep Neural Networks for Speech Steganography (INTERSPEECH 2020)

## Paper
[Hide and Speak: Towards Deep Neural Networks for Speech Steganography](https://arxiv.org/pdf/1902.03083.pdf)
</br>
Felix Kreuk, Yossi Adi, Bhiksha Raj, Rita Singh, Joseph Keshet
</br>
INTERSPEECH 2020

Steganography is the science of hiding a secret message within an ordinary public message, which is referred to as Carrier. Traditionally, digital signal processing techniques, such as least significant bit encoding, were used for hiding messages. In this paper, we explore the use of deep neural networks as steganographic functions for speech data. We showed that steganography models proposed for vision are less suitable for speech, and propose a new model that includes the short-time Fourier transform and inverse-short-time Fourier transform as differentiable layers within the network, thus imposing a vital constraint on the network outputs.
We empirically demonstrated the effectiveness of the proposed method comparing to deep learning based on several speech datasets and analyzed the results quantitatively and qualitatively. Moreover, we showed that the proposed approach could be applied to conceal multiple messages in a single carrier using multiple decoders or a single conditional decoder. Lastly, we evaluated our model under different channel distortions. Qualitative experiments suggest that modifications to the carrier are unnoticeable by human listeners and that the decoded messages are highly intelligible.

If you find this paper and implementation useful, please consider citing our work:
```
@article{kreuk2019hide,
  title={Hide and speak: Deep neural networks for speech steganography},
  author={Kreuk, Felix and Adi, Yossi and Raj, Bhiksha and Singh, Rita and Keshet, Joseph},
  journal={arXiv preprint arXiv:1902.03083},
  year={2019}
}
```
## Audio samples
Audio samples are available [here](https://felixkreuk.github.io/Hide-and-Speak-Towards-Deep-Neural-Networks-for-Speech-Steganography/).

## Clone repository
```
git clone https://github.com/felixkreuk/HideAndSpeak.git
cd HideAndSpeak
```

## Setup environment
```
conda create --name HideAndSpeak --file requirements.txt
conda activate HideAndSpeak
```

## Usage
Please refer to the `--help` section to see possible hyperparameters:
```
python main.py --help                                                                                                                                                                                                                                                        reb -> public_release ! ?usage: main.py [-h] [--lr LR] [--num_iters NUM_ITERS] [--loss_type {mse,abs}]
               [--opt OPT] [--mode {train,test,sample}] --train_path
               TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH
               [--batch_size BATCH_SIZE] [--n_pairs N_PAIRS]
               [--n_messages N_MESSAGES] [--dataset {timit,yoho}]
               [--model_type {n_msg,n_msg_cond,baseline}]
               [--carrier_detach CARRIER_DETACH]
               [--add_stft_noise ADD_STFT_NOISE]
               [--add_carrier_noise {gaussian,snp,salt,pepper,speckle}]
               [--carrier_noise_norm CARRIER_NOISE_NORM] [--adv]
               [--block_type {normal,skip,bn,in,relu}]
               [--enc_n_layers ENC_N_LAYERS] [--dec_c_n_layers DEC_C_N_LAYERS]
               [--lambda_carrier_loss LAMBDA_CARRIER_LOSS]
               [--lambda_msg_loss LAMBDA_MSG_LOSS] [--num_workers NUM_WORKERS]
               [--load_ckpt LOAD_CKPT] [--run_dir RUN_DIR]
               [--save_model_every SAVE_MODEL_EVERY]
               [--sample_every SAMPLE_EVERY]

Hide and Speak

optional arguments:
  -h, --help            show this help message and exit
  --lr LR
  --num_iters NUM_ITERS
                        number of epochs
  --loss_type {mse,abs}
                        loss function used for training
  --opt OPT             optimizer
  --mode {train,test,sample}
                        `train` will initiate training, `test` should be used
                        in conjunction with `load_ckpt` to run a test epoch,
                        `sample` should be used in conjunction with
                        `load_ckpt` to sample examples from dataset
  --train_path TRAIN_PATH
                        path to training set. should be a folder containing
                        .wav files for training
  --val_path VAL_PATH
  --test_path TEST_PATH
  --batch_size BATCH_SIZE
                        batch size
  --n_pairs N_PAIRS     number of training examples generated from wav files
  --n_messages N_MESSAGES
                        number of hidden messages
  --dataset {timit,yoho}
                        select dataset
  --model_type {n_msg,n_msg_cond,baseline}
                        `n_msg` default model type, `n_msg_cond` conditional
                        message decoding, `baseline` is the frequency-chop
                        baseline
  --carrier_detach CARRIER_DETACH
                        flag that stops gradients from the generated carrier
                        and back. if -1 will not be used, if set to k!=-1 then
                        gradients will be stopped from the kth iteration (used
                        for fine-tuning the message decoder)
  --add_stft_noise ADD_STFT_NOISE
                        flag that trasforms the generated carrier spectrogram
                        back to the time domain to simulate real-world
                        conditions. if -1 will not be used, if set to k!=-1
                        will be used from the kth iteration
  --add_carrier_noise {gaussian,snp,salt,pepper,speckle}
                        add different types of noise the the carrier
                        spectrogram
  --carrier_noise_norm CARRIER_NOISE_NORM
                        strength of carrier noise
  --adv                 flag that indicates if adversarial training should be
                        used
  --block_type {normal,skip,bn,in,relu}
                        type of block for encoder/decoder
  --enc_n_layers ENC_N_LAYERS
                        number of layers in encoder
  --dec_c_n_layers DEC_C_N_LAYERS
                        number of layers in decoder
  --lambda_carrier_loss LAMBDA_CARRIER_LOSS
                        coefficient for carrier loss term
  --lambda_msg_loss LAMBDA_MSG_LOSS
                        coefficient for message loss term
  --num_workers NUM_WORKERS
                        number of data loading workers
  --load_ckpt LOAD_CKPT
                        path to checkpoint (used for test epoch or for
                        sampling)
  --run_dir RUN_DIR     output directory for logs, samples and checkpoints
  --save_model_every SAVE_MODEL_EVERY
  --sample_every SAMPLE_EVERY

```
### Training
The `--train_path,--val_path,--test_path` should point to a directory containing `.wav` files. For default settings, use `--dataset timit`, this option will create training examples from all existing `.wav` files in the given directory.
```
python main.py --mode train --train_path /datasets/timit/train --val_path /datasets/timit/val --test_path /datasets/timit/test --dataset timit
python main.py --mode train ... --n_messages 3  # train with 3 hidden messages
python main.py --mode train ... --model_type n_msg_cond  # to train a conditional message decoder
python main.py --mode train ... --add_stft_noise 0  # add stst/istft conversion in training from first epoch
python main.py --mode train ... --carrier_detach 10  # train whole system for 10 epochs, then fine-tune only the message decoder
```
### Testing
```
python main.py --mode test --load_ckpt /path/to/model.ckpt --train_path /datasets/timit/train --val_path /datasets/timit/val --test_path /datasets/timit/test --dataset timit
```
### Sampling
To sample generated examples from the given data (specified using the `--train_path,--val_path,--test_path` flags):
```
python main.py --mode sample --run_dir output_dir --load_ckpt /path/to/model.ckpt --train_path /datasets/timit/train --val_path /datasets/timit/val --test_path /datasets/timit/test --dataset timit
```
