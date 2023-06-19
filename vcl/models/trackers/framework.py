import numbers
from os import stat_result
import os.path as osp
from collections import *
from tkinter.messagebox import NO

import mmcv
from mmcv.runner import auto_fp16, load_state_dict, load_checkpoint

from ..base import BaseModel
from ..builder import build_backbone, build_loss, build_model
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.utils import *
from vcl.models.common import *


import torch.nn as nn
import torch
import torch.nn.functional as F


@MODELS.register_module()
class Framework(BaseModel):

    def __init__(self,
                 backbone,
                 backbone_t,
                 weight=20,
                 num_stage=2,
                 feat_size=[64, 32],
                 radius=[12, 6],
                 downsample_rate=[4, 8],
                 temperature=1.0,
                 temperature_t=0.07,
                 momentum=-1,
                 T=0.7,
                 loss=None,
                 loss_weight=None,
                 scaling=True,
                 norm=False,
                 detach=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):
        """_summary_

        Args:
            backbone (_type_): backbone type
            backbone_t (_type_): backbone type of teacher model
            weight (int, optional): weight in rec loss. Defaults to 20.
            num_stage (int, optional):  number of the pyramid levels. Defaults to 2.
            feat_size (list, optional):  feature size at different pyramid levels. Defaults to [64, 32].
            radius (list, optional):  radius for reconstruction at different levels. Defaults to [12, 6].
            downsample_rate (list, optional):  stride of different pyramid levels. diDefaults to [4, 8].
            temperature (float, optional):  Defaults to 1.0.
            temperature_t (float, optional):  Defaults to 0.07.
            momentum (int, optional):  Defaults to -1.
            T (float, optional): the threshold for entropy-based selection. Defaults to 0.7.
            loss (_type_, optional):  Defaults to None.
            loss_weight (_type_, optional):  Defaults to None.
            scaling (bool, optional):  Defaults to True.
            norm (bool, optional):  Defaults to False.
            detach (bool, optional):  Defaults to False.
            train_cfg (_type_, optional):  Defaults to None.
            test_cfg (_type_, optional):  Defaults to None.
            pretrained (_type_, optional):  Defaults to None.
        """

        super().__init__()
        from mmcv.ops import Correlation
        
        self.num_stage = num_stage
        self.feat_size = feat_size

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.downsample_rate = downsample_rate

        self.momentum = momentum
        self.pretrained = pretrained
        self.temperature = temperature
        self.temperature_t = temperature_t
        self.scaling = scaling
        self.norm = norm
        self.T = T
    
        self.detach = detach
        self.loss_weight = loss_weight
        self.logger = get_root_logger()

        # build backbone
        self.backbone = build_backbone(backbone)
        self.backbone_t = build_backbone(backbone_t) if backbone_t != None else None
        
        # loss
        self.loss = build_loss(loss) if loss != None else None
        self.mask = [ make_mask(feat_size[i], radius[i]) for i in range(len(radius))]   
        self.corr = [Correlation(max_displacement=R) for R in radius ]
        
        self.weight = weight
    
    def init_weights(self):
        
        self.backbone.init_weights()
        if self.backbone_t is not None:  
            self.backbone_t.init_weights()
        
        if self.pretrained is not None:
            _ = load_checkpoint(self, self.pretrained, map_location='cpu')
    
    def forward_train(self, imgs, images_lab=None):
            
        bsz, num_clips, t, c, h, w = imgs.shape
        
        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs = [f.reshape(bsz, t, *f.shape[-3:]) for f in fs]
        
        tar_pyramid, refs_pyramid = [f[:, -1] for f in fs], [ f[:, :-1] for f in fs]
        
        losses = {}
        
        c1s = []
        c2s = []
        for idx, (tar, refs) in enumerate(zip(tar_pyramid, refs_pyramid)):
            
            # get global correlation for distillation
            if idx == len(fs) - 1 and self.backbone_t != None:
                _, c_g = non_local_correlation(tar, refs, scaling=self.scaling)
                break
                        
            # get local correlation map            
            _, c1 = non_local_correlation(tar, refs, mask=self.mask[idx], scaling=True)      
                  
            # for frame reconstruction loss
            ref_gt = [self.prep(gt[:,ch], downsample_rate=self.downsample_rate[idx]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(c1, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, self.feat_size[idx], self.feat_size[idx])     
            losses[f'stage{idx}_l1_loss'] = self.compute_lphoto(images_lab_gt, ch, outputs)[0] * self.loss_weight[f'stage{idx}_l1_loss']
            
            # get local correlation map for calculating entropy
            c2 = self.corr[idx](tar, refs[:,0])
            
            c2s.append(c2)
            c1s.append(c1)
            
        # get weight based on local correlation entropy
        if self.T != -1:
            corr_feat = c2s[-1].reshape(bsz, -1, self.feat_size[-1], self.feat_size[-1])
            corr = corr_feat.softmax(1)
            corr_en = (-torch.log(corr+1e-12)).sum(1)
            # corr_en = self.zero_boundary(corr_en).flatten(-2)
            corr_sorted, _ = torch.sort(corr_en, dim=-1, descending=True)
            idx = int(corr_en.shape[-1] * self.T) - 1
            T = corr_sorted[:, idx:idx+1]
            sparse_mask = (corr_en > T).reshape(bsz, 1, *corr_feat.shape[-2:]).float().detach()
            weight = sparse_mask.flatten(-2).permute(0,2,1).repeat(1, 1, c1s[-1].shape[-1])
        else:
            corr_feat = c2s[-1].reshape(bsz, -1, self.feat_size[-1], self.feat_size[-1])
            corr = corr_feat.softmax(1)
            corr_en = (-torch.log(corr+1e-12)).sum(1)
            corr_en = self.zero_boundary(corr_en)
            corr_en = corr_en.flatten(-2)
            corr_en_no = (corr_en - corr_en.min(-1, keepdim=True)[0]) / (corr_en.max(1,keepdim=True)[0] - corr_en.min(1,keepdim=True)[0])
            
            weight = corr_en_no.unsqueeze(-1).repeat(1, 1, c1s[-1].shape[-1])
        
        # for layer distillation loss
        c1_ = c1s[0].reshape(bsz, -1, *fs[0].shape[-2:])
        c1_ = F.avg_pool2d(c1_, 2, stride=2).flatten(-2).permute(0,2,1).reshape(bsz, -1, *fs[0].shape[-2:])
        target = F.avg_pool2d(c1_, 2, stride=2).flatten(-2).permute(0,2,1)

        
        if not self.detach:
            losses['local_corrdis_loss'] = self.loss_weight['local_corrdis_loss'] * self.loss(c1s[-1][:,0], target, weight=weight)
        else:
            losses['local_corrdis_loss'] = self.loss_weight['local_corrdis_loss'] * self.loss(c1s[-1][:,0], target.detach(), weight=weight)
            
        # for correlation distillation loss
        if self.backbone_t is not None:
            with torch.no_grad():
                self.backbone_t.eval()
                fs_t = self.backbone_t(imgs.flatten(0,2))
                fs_t = fs_t.reshape(bsz, t, *fs_t.shape[-3:])
                tar_t, refs_t = fs_t[:, -1], fs_t[:, :-1]
                _, target_c_g = non_local_correlation(tar_t, refs_t, temprature=self.temperature_t, norm=self.norm)
                
            losses['global_corrdis_loss'] = self.loss_weight['global_corrdis_loss'] * self.loss(c_g, target_c_g)
            
        return losses
    
    
    def prep(self, image, downsample_rate):
        _,c,_,_ = image.size()
        x = image.float()[:,:,::downsample_rate,::downsample_rate]

        return x
    
    def zero_boundary(self, x, num=3):
        
        x[:,-num:, :] = 0
        x[:, :, -num:] = 0
        
        x[:,:num, :] = 0
        x[:,:, :num] = 0
        
        return x
    
    def dropout2d_lab(self, arr): # drop same layers for all images
        if not self.training:
            return arr

        drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
        drop_ch_ind = np.random.choice(np.arange(1,3), drop_ch_num, replace=False)

        for a in arr:
            for dropout_ch in drop_ch_ind:
                a[:, dropout_ch] = 0
            a *= (3 / (3 - drop_ch_num))

        return arr, drop_ch_ind # return channels not masked
    
    def compute_lphoto(self, images_lab_gt, ch, outputs, upsample=True):
        b, c, h, w = images_lab_gt[0].size()

        tar_y = images_lab_gt[-1][:,ch]  # y4

        if upsample:
            outputs = F.interpolate(outputs, (h, w), mode='bilinear')
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='mean')
        else:
            tar_y = self.prep(images_lab_gt[-1])[:,ch]
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='mean')

        err_maps = torch.abs(outputs - tar_y).sum(1).detach()

        return loss, err_maps
    
    