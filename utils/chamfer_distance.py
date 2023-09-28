import os
import numpy as np
from time import time
import cv2
import pdb
import scipy
import scipy.ndimage
import torch
import torchmetrics

black_threshold = 255.0 * 0.99


def batch_edt(img, block=1024):
    expand = False
    bs,h,w = img.shape
    diam2 = h**2 + w**2
    odtype = img.dtype
    grid = (img.nelement()+block-1) // block

    # cupy implementation

    # default to scipy cpu implementation

    sums = img.sum(dim=(1,2))
    ans = torch.tensor(np.stack([
        scipy.ndimage.morphology.distance_transform_edt(i)
        if s!=0 else  # change scipy behavior for empty image
        np.ones_like(i) * np.sqrt(diam2)
        for i,s in zip(1-img, sums)
    ]), dtype=odtype)

    if expand:
        ans = ans.unsqueeze(1)
    return ans


############### DERIVED DISTANCES ###############

# input: (bs,h,w) or (bs,1,h,w)
# returns: (bs,)
# normalized s.t. metric is same across proportional image scales

# average of two asymmetric distances
# normalized by diameter and area
def batch_chamfer_distance(gt, pred, block=1024, return_more=False):
    t = batch_chamfer_distance_t(gt, pred, block=block)
    p = batch_chamfer_distance_p(gt, pred, block=block)
    cd = (t + p) / 2
    return cd
def batch_chamfer_distance_t(gt, pred, block=1024, return_more=False):
    #pdb.set_trace()
    assert gt.device==pred.device and gt.shape==pred.shape
    bs,h,w = gt.shape[0], gt.shape[-2], gt.shape[-1]
    dpred = batch_edt(pred, block=block)
    cd = (gt*dpred).float().mean((-2,-1)) / np.sqrt(h**2+w**2)
    if len(cd.shape)==2:
        assert cd.shape[1]==1
        cd = cd.squeeze(1)
    return cd
def batch_chamfer_distance_p(gt, pred, block=1024, return_more=False):
    assert gt.device==pred.device and gt.shape==pred.shape
    bs,h,w = gt.shape[0], gt.shape[-2], gt.shape[-1]
    dgt = batch_edt(gt, block=block)
    cd = (pred*dgt).float().mean((-2,-1)) / np.sqrt(h**2+w**2)
    if len(cd.shape)==2:
        assert cd.shape[1]==1
        cd = cd.squeeze(1)
    return cd

# normalized by diameter
# always between [0,1]
def batch_hausdorff_distance(gt, pred, block=1024, return_more=False):
    assert gt.device==pred.device and gt.shape==pred.shape
    bs,h,w = gt.shape[0], gt.shape[-2], gt.shape[-1]
    dgt = batch_edt(gt, block=block)
    dpred = batch_edt(pred, block=block)
    hd = torch.stack([
        (dgt*pred).amax(dim=(-2,-1)),
        (dpred*gt).amax(dim=(-2,-1)),
    ]).amax(dim=0).float() / np.sqrt(h**2+w**2)
    if len(hd.shape)==2:
        assert hd.shape[1]==1
        hd = hd.squeeze(1)
    return hd


############### TORCHMETRICS ###############

class ChamferDistance2dMetric(torchmetrics.Metric):
    full_state_update=False
    def __init__(
            self, block=1024, convert_dog=True, k=1.6, epsilon=0.01, kernel_factor=4, clip=False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.block = block
        self.convert_dog = convert_dog

        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        dist = batch_chamfer_distance(target, preds, block=self.block)
        self.running_sum += dist.sum()
        self.running_count += len(dist)
        return
        
    def compute(self):
        return self.running_sum.float() / self.running_count

class ChamferDistance2dTMetric(ChamferDistance2dMetric):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.convert_dog:
            preds = (usketchers.batch_dog(preds, **self.dog_params)>0.5).float()
            target = (usketchers.batch_dog(target, **self.dog_params)>0.5).float()
        dist = batch_chamfer_distance_t(target, preds, block=self.block)
        self.running_sum += dist.sum()
        self.running_count += len(dist)
        return
class ChamferDistance2dPMetric(ChamferDistance2dMetric):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.convert_dog:
            preds = (usketchers.batch_dog(preds, **self.dog_params)>0.5).float()
            target = (usketchers.batch_dog(target, **self.dog_params)>0.5).float()
        dist = batch_chamfer_distance_p(target, preds, block=self.block)
        self.running_sum += dist.sum()
        self.running_count += len(dist)
        return

class HausdorffDistance2dMetric(torchmetrics.Metric):
    def __init__(
            self, block=1024, convert_dog=True,
            t=2.0, sigma=1.0, k=1.6, epsilon=0.01, kernel_factor=4, clip=False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.block = block
        self.convert_dog = convert_dog
        self.dog_params = {
            't': t, 'sigma': sigma, 'k': k, 'epsilon': epsilon,
            'kernel_factor': kernel_factor, 'clip': clip,
        }
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.convert_dog:
            preds = (usketchers.batch_dog(preds, **self.dog_params)>0.5).float()
            target = (usketchers.batch_dog(target, **self.dog_params)>0.5).float()
        dist = batch_hausdorff_distance(target, preds, block=self.block)
        self.running_sum += dist.sum()
        self.running_count += len(dist)
        return
    def compute(self):
        return self.running_sum.float() / self.running_count




def rgb2sketch(img, black_threshold):
    #pdb.set_trace()
    img[img < black_threshold] = 1
    img[img >= black_threshold] = 0
    #cv2.imwrite("grey.png",img*255)
    return torch.tensor(img)
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def cd_score(img1, img2):


    img1 = rgb2gray(img1.astype(float))
    img2 = rgb2gray(img2.astype(float))
    
    img1_sketch = rgb2sketch(img1, black_threshold)
    img2_sketch = rgb2sketch(img2, black_threshold)

    img1_sketch = img1_sketch.unsqueeze(0)
    img2_sketch = img2_sketch.unsqueeze(0)

    CD = ChamferDistance2dMetric()
    cd = CD(img1_sketch,img2_sketch)
    return cd

