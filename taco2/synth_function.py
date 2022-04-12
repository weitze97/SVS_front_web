import os
import time
import argparse
from numpy import finfo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import pdb
import soundfile as sf
import torch
from taco2.distributed import apply_gradient_allreduce
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from taco2.model import Tacotron2 
from taco2.data_utils import TextMelLoader, TextMelCollate
from taco2.hparams import create_hparams
from taco2.utils import get_mask_from_lengths, write_hdf5
from taco2.audio_processing import griffin_lim
from taco2.stft import STFT
from taco2.layers import TacotronSTFT


# command:
# python synth.py -o exp/pwg

def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    valset = TextMelLoader(hparams.testing_files, hparams, phase='infer')
    collate_fn = TextMelCollate(hparams.n_frames_per_step, phase='infer')

    return valset, collate_fn


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def mel2wave_gl(hparams, mel, n_iters=30, scaling=1000, compress_factor='log10'):
    # parameterize some functions
    stft_fn = STFT(
        filter_length=hparams.filter_length, 
        hop_length=hparams.hop_length, 
        win_length=hparams.win_length
        )

    taco_stft = TacotronSTFT(
        filter_length=hparams.filter_length, 
        hop_length=hparams.hop_length, 
        win_length=hparams.win_length, 
        sampling_rate=hparams.sampling_rate, 
        mel_fmin=hparams.mel_fmin, 
        mel_fmax=hparams.mel_fmax
        )

    # mel (torch): [80, T_out] --> [1, T_out, 80]
    mel_decompress = mel.unsqueeze(0)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()

    if compress_factor == 'log10':
        mel_decompress = 10**mel_decompress
    elif compress_factor == 'ln':
        mel_decompress = torch.exp(mel_decompress)
    else:
        raise 'Error: compress_factor error!'

    # Project from Mel-Spectrogram to Linear-Spectrogram
    spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1)
    spec_from_mel = spec_from_mel * scaling
    spec_from_mel = spec_from_mel.unsqueeze(0)
    x_reconstruct = griffin_lim(spec_from_mel, stft_fn, n_iters)

    max_sample = abs(x_reconstruct).max()
    x_reconstruct = (x_reconstruct / max_sample).detach().numpy().T

    return x_reconstruct


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def plot_data(data, outdir, iter, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')
    plt.savefig(os.path.join(outdir, 'val_spec_{}.png'.format(iter)), dpi=600)
    plt.close()

def plot_f0_contour(lf0_targets, lf0_outputs, vuv_targets, vuv_outputs, save_dir):
    f0_targets = lf0_targets.copy()
    f0_targets[vuv_targets < 0.5] = 0
    f0_targets[np.nonzero(f0_targets)] = 10 ** (f0_targets[np.nonzero(f0_targets)])

    f0_outputs = lf0_outputs.copy()
    f0_outputs[vuv_outputs < 0.5] = 0
    f0_outputs[np.nonzero(f0_outputs)] = 10 ** (f0_outputs[np.nonzero(f0_outputs)])

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(f0_targets, color='green', label='target')
    ax.plot(f0_outputs, color='red', label='output')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("F0")
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, 'f0.png'), dpi=600)
    plt.close()


def validate(args, model, valset, batch_size, n_gpus,
             collate_fn, distributed_run, rank, hparams):
    """Handles all the validation scoring and printing"""
    model.eval()
      
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        count = 0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            
            text_padded, note_padded, notelen_padded, input_lengths, mel_padded, max_len, output_lengths = x
            mel_targets, gate_targets = y

            if output_lengths[0] != 50 and output_lengths[-1] != 50:
                mask = get_mask_from_lengths(output_lengths)
            else:
                mask = None

            for j in range(mel_targets.shape[0]):

                # parse inference input
                x_infer = [text_padded[j, :input_lengths[j].cpu().numpy()].cpu().numpy(), 
                            note_padded[j, :input_lengths[j].cpu().numpy()].cpu().numpy(), 
                            notelen_padded[j, :input_lengths[j].cpu().numpy()].cpu().numpy()]

                sequence = np.array(x_infer)
                sequence = sequence.reshape(3, 1, -1)  
                sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

                # parse inference output
                y_pred = model.inference(sequence)
                mel_outputs, mel_outputs_postnet, gate_out, alignments = y_pred
                
                # for validation (given targets)
                if mask is not None:
                    mel_pred = mel_outputs_postnet[0, :, :mask[j].sum()] # (80, T_out)
                    mel_gt = mel_targets[j, :, :mask[j].sum()]
                    mel_targets_plot = mel_targets.float().data.cpu().numpy()[j, :, :mask[j].sum()]
                    mel_outputs_plot = mel_outputs_postnet.float().data.cpu().numpy()[0, :, :mask[j].sum()]
                    alignments_plot = alignments[0, :mask[j].sum(), :].float().data.cpu().numpy().T
                # for testing (without given targets)
                else:
                    mel_pred = mel_outputs_postnet[0, :, :]
                    mel_gt = mel_targets[j, :, :]
                    mel_targets_plot = mel_targets.float().data.cpu().numpy()[j, :, :]
                    mel_outputs_plot = mel_outputs_postnet.float().data.cpu().numpy()[0, :, :]
                    alignments_plot = alignments[0, :, :].float().data.cpu().numpy().T


                # Griffin-Lim
                x_reconstruct = mel2wave_gl(hparams, mel_pred)
                x_reconstruct_target = mel2wave_gl(hparams, mel_gt)
                               
                if j == 0:
                    X = x_reconstruct
                    X_target = x_reconstruct_target
                else:
                    X = np.concatenate((X, x_reconstruct), axis=0)
                    X_target = np.concatenate((X_target, x_reconstruct_target), axis=0)

                count += 1
                print('finish:{}/{}'.format(count, len(valset)))

                # save: (1) mel for neural vocoders
                save_mel_dir = os.path.join(args.output_directory, 'mel')
                if not os.path.isdir(save_mel_dir):
                    os.makedirs(save_mel_dir)
                mel_pred = mel_pred.detach().cpu().numpy().T  # save to hdf5: (T_out, 80)          
                write_hdf5(os.path.join(save_mel_dir, 'mel_{}.h5'.format(count)), 
                            "feats", mel_pred.astype(np.float32))
              
                # save: (2) figure
                save_fig_dir = os.path.join(args.output_directory, 'figure')
                if not os.path.isdir(save_fig_dir):
                    os.makedirs(save_fig_dir)

                # plot mel-spectrogram of one of the segments
                plot_data((mel_targets_plot,
                        mel_outputs_plot,
                        alignments_plot),
                        save_fig_dir,
                        count)

                # save: (3) soundfile generated by Griffin-Lim
                save_wav_dir = os.path.join(args.output_directory, 'soundfile')
                if not os.path.isdir(save_wav_dir):
                    os.makedirs(save_wav_dir)
            
            print('Saving Generated Audio for Griffin-Lim......')    
            sf.write(os.path.join(save_wav_dir, 'GL_test_batch{}.wav'.format(i+1)), X, hparams.sampling_rate)
            sf.write(os.path.join(save_wav_dir, 'GL_target_batch{}.wav'.format(i+1)), X_target, hparams.sampling_rate)




def train(args, output_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):


    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1

    model.eval()

    validate(args, model, valset, hparams.batch_size, n_gpus, collate_fn, 
            hparams.distributed_run, rank, hparams)



def synth():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', default="./taco2/exp/pwg", type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--checkpoint_path', 
                        default="./taco_f1",
                        type=str, required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=4,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args, args.output_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)

if __name__ == '__main__':
    synth()