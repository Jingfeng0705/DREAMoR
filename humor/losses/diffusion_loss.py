
import time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(self,  # if > 0 will anneal KL loss cyclicly
                ddpm=1.0,):
        super(DiffusionLoss, self).__init__()
        # self.kl_loss_weight = kl_loss
        # self.kl_loss_anneal_start = kl_loss_anneal_start
        # self.kl_loss_anneal_end = kl_loss_anneal_end
        # self.use_kl_anneal = self.kl_loss_anneal_end > self.kl_loss_anneal_start

        # self.kl_loss_cycle_len = kl_loss_cycle_len
        # self.use_kl_cycle = False
        # if self.kl_loss_cycle_len > 0:
        #     self.use_kl_cycle = True
        #     self.use_kl_anneal = False
            
        self.ddpm_weight = ddpm

    def forward(self, pred_dict, gt_dict, cur_epoch, gender=None, betas=None):
        loss = 0.0
        stats_dict = dict()
        pred_noise = pred_dict['pred_noise']
        noise = pred_dict['noise']
        
        loss = F.mse_loss(pred_noise, noise)
        
        stats_dict['loss'] = loss.item()
        
        return loss, stats_dict
        
        