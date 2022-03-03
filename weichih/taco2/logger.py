import random
import torch
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_spectrogram_to_numpy_dur
from plotting_utils import plot_spectrogram_to_numpy_dur_v2
from plotting_utils import plot_gate_outputs_to_numpy
from utils import onehot2phone


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')

    

    def log_validation_dur(self, reduced_loss, model, y, y_pred, onehot_phone_seq, note_len, mask, iteration, spec, sampling_rate, hop_length):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        ph_du_output = y_pred.squeeze(-1)
        ph_du_target = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, ph_du_output.size(0) - 1)
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy_dur(
            spec[idx].data.cpu().numpy(),
            onehot2phone(onehot_phone_seq[idx].data.cpu().numpy()),
            note_len.data.cpu().numpy()[idx, :mask[idx].sum()],
            ph_du_output.data.cpu().numpy()[idx, :mask[idx].sum()],
            sampling_rate, hop_length),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy_dur(
            spec[idx].data.cpu().numpy(),
            onehot2phone(onehot_phone_seq[idx].data.cpu().numpy()),
            note_len.data.cpu().numpy()[idx, :mask[idx].sum()],
            ph_du_target.data.cpu().numpy()[idx, :mask[idx].sum()],
            sampling_rate, hop_length),
            iteration, dataformats='HWC')

    def log_validation_dur_v2(self, reduced_loss, model, y, y_pred, phone_seq, note_len, mask, iteration, spec, sampling_rate, hop_length):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        ph_du_output = y_pred.squeeze(-1)
        ph_du_target = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, ph_du_output.size(0) - 1)
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy_dur_v2(
            spec[idx].data.cpu().numpy(),
            phone_seq[idx],
            note_len[idx, :mask[idx].sum()],
            ph_du_output.data.cpu().numpy()[idx, :mask[idx].sum()],
            sampling_rate, hop_length),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy_dur_v2(
            spec[idx].data.cpu().numpy(),
            phone_seq[idx],
            note_len[idx, :mask[idx].sum()],
            ph_du_target.data.cpu().numpy()[idx, :mask[idx].sum()],
            sampling_rate, hop_length),
            iteration, dataformats='HWC')