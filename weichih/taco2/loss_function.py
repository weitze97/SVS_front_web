from torch import nn
from utils import get_mask_from_lengths
import pdb 
import torch

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets, output_lengths):

        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        
        mask = get_mask_from_lengths(output_lengths)
        mask = mask.expand(80, mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        num_non_padding = mask.sum()
        

        mel_out, mel_out_postnet, gate_out, alignment = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss(reduction='sum')(mel_out, mel_target) + \
            nn.MSELoss(reduction='sum')(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss(reduction='sum')(gate_out, gate_target)
        total_loss = (mel_loss + gate_loss)/num_non_padding

        # --------------- guided attention -------------------
        guided_loss = self.guided_loss(alignment)

        return total_loss, guided_loss

    def guided_loss(self, alignment):

        loss = torch.sum(torch.abs(alignment * self.get_att_weight(alignment[0,...])))
        return loss / (alignment.shape[0] * alignment.shape[2])
        
    
    @staticmethod
    def get_att_weight(A_nt, g=0.2):
        n_idx = torch.arange(0., A_nt.shape[0]).float()
        t_idx = torch.arange(0., A_nt.shape[1]).float()
        W_n, W_t = torch.meshgrid(n_idx, t_idx)

        W = 1 - torch.exp(-(W_n/float(A_nt.shape[0]) -  W_t / float(A_nt.shape[1])) ** 2 / (2 * g * g))
        return W.reshape([1, W.shape[0], W.shape[1]]).cuda()