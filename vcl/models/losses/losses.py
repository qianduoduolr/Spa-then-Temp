# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import *

@masked_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated L1 loss.
    """
    return F.l1_loss(pred, target, reduction='none')

@masked_loss
def mse_loss(pred, target):
    """MSE loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated MSE loss.
    """
    return F.mse_loss(pred, target, reduction='none')

@masked_loss
def charbonnier_loss(pred, target, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return torch.sqrt((pred - target)**2 + eps)

@masked_loss
def kl_loss(pred, target, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return F.kl_div(F.log_softmax(pred, dim=-1), target.softmax(dim=-1), reduction='none')

@LOSSES.register_module()
class Ce_Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * F.cross_entropy(pred, target.squeeze(1), reduction=self.reduction)

@LOSSES.register_module()
class Soft_Ce_Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
  
        log_likelihood = -F.log_softmax(pred, dim=-1)
        bsz = pred.shape[0]
        if self.reduction == 'mean':
            loss = torch.sum(torch.mul(log_likelihood, target.softmax(dim=-1))) / bsz
        else:
            loss = torch.sum(torch.mul(log_likelihood, target.softmax(dim=-1)), dim=-1)

        if weight is not None:
            weight = weight.view(-1)
            loss = (loss * weight).sum() / weight.sum()

        return loss * self.loss_weight


@LOSSES.register_module()
class Kl_Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * kl_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # if self.sample_wise:
        #     loss = F.mse_loss(pred, target, reduction='none')
        #     if self.reduction == 'mean':
        #         loss = loss.mean(-1)
        #     else:
        #         loss = loss.sum(-1)
        # else:
        #     assert self.reduction != 'none'
        #     loss = F.mse_loss(pred, target, reduction=self.reduction)
        # return self.loss_weight * loss
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)



@LOSSES.register_module()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 sample_wise=False,
                 eps=1e-12):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class CosineSimLoss(nn.Module):
    """NLL Loss.

    It will calculate Cosine Similarity loss given cls_score and label.
    """

    def __init__(self,
                 loss_weight=1.0,
                 with_norm=True,
                 negative=False,
                 pairwise=False,
                 reduction='mean',
                 **kwargs):
        super().__init__()
        self.with_norm = with_norm
        self.negative = negative
        self.pairwise = pairwise
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, cls_score, label, mask=None, **kwargs):
        if self.with_norm:
            cls_score = F.normalize(cls_score, p=2, dim=1)
            label = F.normalize(label, p=2, dim=1)
        if mask is not None:
            assert self.pairwise
        if self.pairwise:
            cls_score = cls_score.flatten(2)
            label = label.flatten(2)
            prod = torch.einsum('bci,bcj->bij', cls_score, label)
            if mask is not None:
                assert prod.shape == mask.shape
                prod *= mask.float()
            prod = prod.flatten(1)
        else:
            prod = torch.sum(
                cls_score * label, dim=1).view(cls_score.size(0), -1)
        if self.negative:
            loss = -prod.mean(dim=-1)
        else:
            loss = 2 - 2 * prod.mean(dim=-1)
        
        # if self.reduction == 'mean':
        #     loss = loss.mean()
        # elif self.reduction == 'sum':
        #     loss = loss.sum()
        # else:
        #     pass
            
        return loss * self.loss_weight
    
    

@LOSSES.register_module()
class DiscreteLoss(nn.Module):
    """NLL Loss.

    It will calculate Cosine Similarity loss given cls_score and label.
    """

    def __init__(self,
                 nbins,
                 fmax,
                 loss_weight=1.0,
                 reduction='mean',
                 **kwargs):
        super().__init__()
      
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss()
        assert nbins % 2 == 1, "nbins should be odd"
        self.nbins = nbins
        self.fmax = fmax
        self.step = 2 * fmax / float(nbins)
        
    def tobin(self, target):
        target = torch.clamp(target, -self.fmax + 1e-3, self.fmax - 1e-3)
        quantized_target = torch.floor((target + self.fmax) / self.step)
        return quantized_target.type(torch.cuda.LongTensor)

    def __call__(self, input, target):
        size = target.shape[2:4]
        if input.shape[2] != size[0] or input.shape[3] != size[1]:
            input = nn.functional.interpolate(input, size=size, mode="bilinear", align_corners=True)
        target = self.tobin(target)
        assert input.size(1) == self.nbins * 2
        loss = self.loss(input[:,:self.nbins,...], target[:,0,...]) + self.loss(input[:,self.nbins:,...], target[:,1,...])
        return loss * self.loss_weight