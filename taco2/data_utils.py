import random
import numpy as np
import torch
import torch.utils.data
import os
import taco2.layers as layers
from taco2.utils import load_wav_to_torch, load_filepaths_and_text, read_hdf5
from taco2.utils import phone2id, note2id, notelen2id, pl2fl
from taco2.hparams import create_hparams
import argparse
import pdb


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, phase='train'):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.hop_length = hparams.hop_length
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)

        self.phase = phase
        if self.phase == 'train':
            random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        if len(audiopath_and_text) == 5:
            audiopath, phon, note, note_len,\
            pho_dur = audiopath_and_text[0],\
                    audiopath_and_text[1],\
                    audiopath_and_text[2],\
                    audiopath_and_text[3],\
                    audiopath_and_text[4]
            mel = self.get_mel(audiopath)

        elif len(audiopath_and_text) == 4:
            phon, note, note_len,\
            pho_dur = audiopath_and_text[0],\
                    audiopath_and_text[1],\
                    audiopath_and_text[2],\
                    audiopath_and_text[3]
            '''
            Need not to give mels in inference phase; 
            in order not to modify too much,
            assign an arbitrary number (50) to T_out.
            '''
            mel = torch.zeros(80, 50)

        else:
            raise 'Error: load_filepaths_and_text error!'

        text = self.get_text(phon, phone2id)  # replace this function.
        note = self.get_text(note, note2id)
        note_len = self.get_text(note_len, notelen2id)

        # ---------------------------------------- pl2fl -------------------------------------------------
        # parse phoneme level to frame level by phoneme duration
        text = pl2fl(text, pho_dur, self.sampling_rate, self.hop_length)
        note = pl2fl(note, pho_dur, self.sampling_rate, self.hop_length)
        note_len = pl2fl(note_len, pho_dur, self.sampling_rate, self.hop_length)

        return (text, note, note_len, mel)
            
            
    # 用Griffin-Lim的code
    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)  
        else:
            
            filename = filename.replace('.wav', '.h5')
            filename = filename.split('/')[-1]
            dirs = [
                "/home/ahgmuse/database/mel/f1/train", 
                "/home/ahgmuse/database/mel/f1/dev"
                ]
            for dir in dirs:
                for path in os.listdir(dir):
                    if filename == path:
                        new_filename = os.path.join(dir, path)
                        break
                    else:
                        continue
            
            melspec = read_hdf5(new_filename, 'feats').T
            melspec = torch.from_numpy(melspec)


            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec


    def get_text(self, text, text2id_fn):
        phone_seq = text.split()
        id_seq, _ = text2id_fn(phone_seq)  # phone2id, note2id, notelen2id
        text_norm = torch.IntTensor(id_seq)
        return text_norm


    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step, phase='train'):
        self.n_frames_per_step = n_frames_per_step
        self.phase = phase

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        # 模仿text_padded做note, notelen padding
        note_padded = torch.LongTensor(len(batch), max_input_len)
        note_padded.zero_()
        notelen_padded = torch.LongTensor(len(batch), max_input_len)
        notelen_padded.zero_()
        duration_padded = torch.LongTensor(len(batch), max_input_len)
        duration_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):

            if self.phase == 'train':
                text = batch[ids_sorted_decreasing[i]][0]
                note = batch[ids_sorted_decreasing[i]][1]
                notelen = batch[ids_sorted_decreasing[i]][2]

            elif self.phase == 'infer':
                text = batch[i][0]
                note = batch[i][1]
                notelen = batch[i][2]
            
            else:
                raise 'Error: phase error!'

            text_padded[i, :text.size(0)] = text
            note_padded[i, :note.size(0)] = note
            notelen_padded[i, :notelen.size(0)] = notelen
      

        # Right zero-pad mel-spec
        # num_mels = batch[0][1].size(0)
        num_mels = batch[0][3].size(0)
        max_target_len = max([x[3].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):

            if self.phase == 'train':
                mel = batch[ids_sorted_decreasing[i]][3]
            elif self.phase == 'infer':
                mel = batch[i][3]
            else:
                raise 'Error: phase error!'

            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, note_padded, notelen_padded, \
            input_lengths, mel_padded, gate_padded, output_lengths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
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

    trainset = TextMelLoader('/media/harddrive/svs/dataset/f1/input/audio_text_val.txt', hparams)
    pdb.set_trace()
