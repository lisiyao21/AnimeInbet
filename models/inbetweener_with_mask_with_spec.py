from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
# from seg_desc import seg_descriptor
import argparse
import numpy as np
import torch.nn.functional as F
from sknetwork.embedding import Spectral

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]

class ThreeLayerEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, enc_dim):
        super().__init__()
        # input must be 3 channel (r, g, b)
        self.layer1 = nn.Conv2d(3, enc_dim//4, 7, padding=3)
        self.non_linear1 = nn.ReLU()
        self.layer2 = nn.Conv2d(enc_dim//4, enc_dim//2, 3, padding=1)
        self.non_linear2 = nn.ReLU()
        self.layer3 = nn.Conv2d(enc_dim//2, enc_dim, 3, padding=1)

        self.norm1 = nn.InstanceNorm2d(enc_dim//4)
        self.norm2 = nn.InstanceNorm2d(enc_dim//2)
        self.norm3 = nn.InstanceNorm2d(enc_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, img):
        x = self.non_linear1(self.norm1(self.layer1(img)))
        x = self.non_linear2(self.norm2(self.layer2(x)))
        x = self.norm3(self.layer3(x))
        # x = self.non_linear1(self.layer1(img))
        # x = self.non_linear2(self.layer2(x))
        # x = self.layer3(x)
        return x


class VertexDescriptor(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, enc_dim):
        super().__init__()
        self.encoder = ThreeLayerEncoder(enc_dim)
        # self.super_pixel_pooling = 
        # use scatter
        # nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, img, vtx):
        x = self.encoder(img)
        n, c, h, w = x.size()
        assert((h, w) == img.size()[2:4])
        return x[:, :, torch.round(vtx[0, :, 1]).long(), torch.round(vtx[0, :, 0]).long()]
        # return super_pixel_pooling(x.view(n, c, -1), seg.view(-1).long(), reduce='mean')
        # here return size is [1]xCx|Seg|


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([2] + layers + [feature_dim])
        # for m in self.encoder.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        inputs = kpts.transpose(1, 2)

        x = self.encoder(inputs)
        return x

class TopoEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([64] + layers + [feature_dim])
        # for m in self.encoder.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        inputs = kpts.transpose(1, 2)
        x = self.encoder(inputs)
        return x


def attention(query, key, value, mask=None):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf'))

    prob = torch.nn.functional.softmax(scores, dim=-1)

    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, mask=None):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value, mask)
        # self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, mask=None):
        message = self.attn(x, source, source, mask)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1, mask00=None, mask11=None, mask01=None, mask10=None):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
                mask0, mask1 = mask01[:, None], mask10[:, None] 
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
                mask0, mask1 = mask00[:, None], mask11[:, None]

            delta0, delta1 = layer(desc0, src0, mask0), layer(desc1, src1, mask1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int, ms=None, ns=None):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    if ms is  None or ns is  None:
        ms, ns = (m*one).to(scores), (n*one).to(scores)
    # else:
    #     ms, ns = ms.to(scores)[:, None], ns.to(scores)[:, None]
    # here m,n should be parameters not shape

    # ms, ns: (b, )
    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    # pad additional scores for unmatcheed (to -1)
    # alpha is the learned threshold
    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log() # (b, )

    if ms.size()[0] > 0:
        norm = norm[:, None]
        log_mu = torch.cat([norm.expand(b, m), ns.log()[:, None] + norm], dim=-1) # (m + 1)
        log_nu = torch.cat([norm.expand(b, n), ms.log()[:, None] + norm], dim=-1)
    else:
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm]) # (m + 1)
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)

    if ms.size()[0] > 1:
        norm = norm[:, :, None]
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlueT(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    # default_config = {
    #     'descriptor_dim': 128,
    #     'weights': 'indoor',
    #     'keypoint_encoder': [32, 64, 128],
    #     'GNN_layers': ['self', 'cross'] * 9,
    #     'sinkhorn_iterations': 100,
    #     'match_threshold': 0.2,
    # }

    def __init__(self, config=None):
        super().__init__()

        default_config = argparse.Namespace()
        default_config.descriptor_dim = 128

        default_config.keypoint_encoder = [32, 64, 128]
        default_config.GNN_layers = ['self', 'cross'] * 9
        default_config.sinkhorn_iterations = 100
        default_config.match_threshold = 0.2
        self.spectral = Spectral(64,  normalized=False)


        if config is None:
            self.config = default_config
        else:
            self.config = config   
            self.config.GNN_layers = ['self', 'cross'] * self.config.GNN_layer_num

        self.kenc = KeypointEncoder(
            self.config.descriptor_dim, self.config.keypoint_encoder)

        self.tenc = TopoEncoder(
            self.config.descriptor_dim, [96])


        self.gnn = AttentionalGNN(
            self.config.descriptor_dim, self.config.GNN_layers)

        self.final_proj = nn.Conv1d(
            self.config.descriptor_dim, self.config.descriptor_dim,
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        self.vertex_desc = VertexDescriptor(self.config.descriptor_dim)
       

    def forward(self, data):
        kpts0, kpts1 = data['keypoints0'].float(), data['keypoints1'].float()

        ori_mask0, ori_mask1 = data['mask0'].float(), data['mask1'].float()
        dim_m, dim_n = data['ms'].float(), data['ns'].float()

        # spectual embedding of adj matrices
        # here I find that online computation of spectrals are too slow during training
        # so the spectrual embedding is moved to dataset pipeline 
        # such that it can be computed in data preparation by multi-processing cpus
        spec0, spec1 = data['spec0'], data['spec1']
        # spec0, spec1 = np.abs(self.spectral.fit_transform(adj_mat0[0].cpu().numpy())), np.abs(self.spectral.fit_transform(adj_mat1[0].cpu().numpy()))

        mmax = dim_m.int().max()
        nmax = dim_n.int().max()

        mask0 = ori_mask0[:, :mmax]
        mask1 = ori_mask1[:, :nmax]

        kpts0 = kpts0[:, :mmax]
        kpts1 = kpts1[:, :nmax]

        # image context embedding
        desc0, desc1 = self.vertex_desc(data['image0'], kpts0.float()), self.vertex_desc(data['image1'], kpts1.float())

        # add topological embedding
        desc0 = desc0 + self.tenc(desc0.new_tensor(spec0))
        desc1 = desc1 + self.tenc(desc1.new_tensor(spec1))

        # masks here were prepared for synchronized training with bach size > 1, but seems not to work well
        # so the current framework still uses grad accumulation 
        mask00 = torch.ones_like(mask0)[:, :, None] * mask0[:, None, :]
        
        mask11 = torch.ones_like(mask1)[:, :, None] * mask1[:, None, :]
        mask01 = torch.ones_like(mask0)[:, :, None] * mask1[:, None, :]
        mask10 = torch.ones_like(mask1)[:, :, None] * mask0[:, None, :]
        

        if kpts0.shape[1] < 2 or kpts1.shape[1] < 2:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            # print(data['file_name'])
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                # 'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                # 'matching_scores1': kpts1.new_zeros(shape1)[0],
                'skip_train': True
            }

        all_matches = data['all_matches'] if 'all_matches' in data else None# shape = (1, K1)

        # positional embedding
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        pos0 = self.kenc(kpts0)
        pos1 = self.kenc(kpts1)

        desc0 = desc0 + pos0
        desc1 = desc1 + pos1


        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1, mask00, mask11, mask01, mask10)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores0 = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc0)
        scores1 = torch.einsum('bdn,bdm->bnm', mdesc1, mdesc1)


        # b k1 k2
        scores = scores / self.config.descriptor_dim**.5


        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config.sinkhorn_iterations,
            ms=dim_m, ns=dim_n)


        # Get the matches with score above "match_threshold".
        return scores[:, :-1, :-1], scores0, scores1, mdesc0, mdesc1
       
def tensor_erode(bin_img, ksize=5):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k

    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded

class InbetweenerTM(nn.Module):
    """AnimeInbet
    The whole pipeline includes
    1. vertex correspondence (vertex embedding + correspondence transformer)
    2. repositioning propagation
    3. vis mask

    vertex corr code is modified from SUPER GLUE 

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """


    def __init__(self, config=None):
        super().__init__()
        # vertex correspondence
        self.corr = SuperGlueT(config.corr_model)
        self.mask_map = MLP([config.corr_model.descriptor_dim, 32, 1])
        self.pos_weight = config.pos_weight
        

    def forward(self, data):
        # if in the mode of video generating
        if 'gen_vid' in data:
            dim_m, dim_n = data['ms'].float(), data['ns'].float()
            mmax = dim_m.int().max()
            nmax = dim_n.int().max()
            with torch.no_grad():
                self.corr.eval()
                score01, score0, score1, dec0, dec1 = self.corr(data)
                kpts0, kpts1 = data['keypoints0'][:,:mmax].float(), data['keypoints1'][:,:nmax].float() # BM2, BN2 

                motion_pred0 = torch.softmax(score01, dim=-1) @ kpts1 - kpts0
                motion_pred1 = torch.softmax(score01.transpose(1, 2), dim=-1) @ kpts0 - kpts1

            motion_pred0 = torch.softmax(score0, dim=-1) @ motion_pred0
            motion_pred1 = torch.softmax(score1, dim=-1) @ motion_pred1
            
            self.mask_map.eval()
            vb0 = self.mask_map(dec0)[:, 0]
            vb1 = self.mask_map(dec1)[:, 0]

            motion_output0, motion_output1 =  motion_pred0.clone(), motion_pred1.clone()
            kpt0t = kpts0 + motion_output0 
            kpt1t = kpts1 + motion_output1 
            if 'topo0' in data and 'topo1' in data:
            # print(len(data['topo0'][0]), len(data['topo1']), flush=True)
                for node, nbs in enumerate(data['topo0'][0]):
                    for nb in nbs:
                        if vb0[0, nb] and vb0[0, node] and ((kpt0t[0, node] - kpt0t[0, nb]) ** 2).sum() / (((kpts0[0, node] - kpts0[0, nb]) ** 2).sum() + 1e-7) > 3:
                            vb0[0, nb] = 0
                            vb0[0, node] = 0
                for node, nbs in enumerate(data['topo1'][0]):
                    for nb in nbs:
                        if vb1[0, nb] and vb1[0, node] and ((kpt1t[0, node] - kpt1t[0, nb]) ** 2).sum() / (((kpts1[0, node] - kpts1[0, nb]) ** 2).sum() + 1e-7) > 3:
                            vb1[0, nb] = 0
                            vb1[0, node] = 0
            return {'r0': motion_output0, 'r1': motion_output1, 'vb0':vb0, 'vb1':vb1,}

        # in the normal train/test mode
        dim_m, dim_n = data['ms'].float(), data['ns'].float()
        mmax = dim_m.int().max()
        nmax = dim_n.int().max()
        # with torch.no_grad():
        #     self.corr.eval()
        score01, score0, score1, dec0, dec1 = self.corr(data)


        kpts0, kpts1 = data['keypoints0'][:,:mmax].float(), data['keypoints1'][:,:nmax].float() # BM2, BN2 

        motion_pred0 = torch.softmax(score01, dim=-1) @ kpts1 - kpts0
        motion_pred1 = torch.softmax(score01.transpose(1, 2), dim=-1) @ kpts0 - kpts1

        motion_pred0 = torch.softmax(score0, dim=-1) @ motion_pred0
        motion_pred1 = torch.softmax(score1, dim=-1) @ motion_pred1


        max0, max1 = score01.max(2), score01.max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = score01.new_tensor(0)

        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        # valid0 = mutual0 & (mscores0 > self.config.match_threshold)
        # valid1 = mutual1 & valid0.gather(1, indices1)
        
        valid0 = mscores0 > 0.2
        valid1 = valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))




        # motion_pred1[0][indices1[0]==-1] = 0




        
        vb0 = self.mask_map(dec0)[:, 0]
        vb1 = self.mask_map(dec1)[:, 0]
        motion_output0, motion_output1 =  motion_pred0.clone(), motion_pred1.clone()

        if not self.training:
            motion_pred0[0][indices0[0]!=-1] = kpts1[0][indices0[0][indices0[0]!=-1]] - kpts0[0][indices0[0]!=-1]
        # # motion_pred0[0][indices0[0]==-1] = 0
            motion_pred1[0][indices1[0]!=-1] = kpts0[0][indices1[0][indices1[0]!=-1]] - kpts1[0][indices1[0]!=-1]
            vb0[:] = vb0[:] + 0.7
            vb1[:] = vb1[:] + 0.7

            # motion0_pred, vb0 = pred0[:, :2].permute(0, 2, 1), pred0[:, 2:][:, 0]
            # motion1_pred, vb1 = pred1[:, :2].permute(0, 2, 1), pred1[:, 2:][:, 0]
            
            # delta0, delta1 = motion_delta[:, :, :mmax].permute(0, 2, 1), motion_delta[:, :, mmax:].permute(0, 2, 1)
            # motion_output0, motion_output1 =  motion0 + delta0, motion1 + delta1
            motion_output0, motion_output1 =  motion_pred0.clone(), motion_pred1.clone()

            im0_erode =  data['image0']
            im1_erode =  data['image1']
            im0_erode[im0_erode > 0] = 1
            im0_erode[im0_erode <= 0] = 0
            im1_erode[im1_erode > 0] = 1
            im1_erode[im1_erode <= 0] = 0
            
            im0_erode = tensor_erode(im0_erode, 7)
            im1_erode = tensor_erode(im1_erode, 7)


            
            kpt0t = kpts0 + motion_output0 / 2
            kpt1t = kpts1 + motion_output1 / 2
        
        
            ##################################################
            ##  Note Here the mini batch size is 1!!!!!!!!  ##
            ##################################################

            if 'topo0' in data and 'topo1' in data:
                # print(len(data['topo0'][0]), len(data['topo1']), flush=True)
                for node, nbs in enumerate(data['topo0'][0]):
                    for nb in nbs:
                        if vb0[0, nb] > 0 and vb0[0, node] > 0 and ((kpt0t[0, node] - kpt0t[0, nb]) ** 2).sum() / (((kpts0[0, node] - kpts0[0, nb]) ** 2).sum() + 1e-7) > 5:
                            vb0[0, nb] = -1
                            vb0[0, node] = -1
                for node, nbs in enumerate(data['topo1'][0]):
                    for nb in nbs:
                        if vb1[0, nb] > 0 and vb1[0, node] > 0 and ((kpt1t[0, node] - kpt1t[0, nb]) ** 2).sum() / (((kpts1[0, node] - kpts1[0, nb]) ** 2).sum() + 1e-7) > 5:
                            vb1[0, nb] = -1
                            vb1[0, node] = -1
            
            
            kpt0t = kpts0 + motion_output0 * 1
            kpt1t = kpts1 + motion_output1 * 1
            if 'topo0' in data and 'topo1' in data:
                ##  print(len(data['topo0'][0]), len(data['topo1']), flush=True)
                for node, nbs in enumerate(data['topo0'][0]):
                    for nb in nbs:

                        center = ((kpt0t[0, node] + kpt0t[0, nb]) * 0.5).int()[0]
                        if center[0] >= 720 or center[1] >= 720:
                            continue

                        if vb0[0, nb] > 0 and vb0[0, node] > 0 and im1_erode[0,:, center[1], center[0]].mean() > 0.8:
                            vb0[0, nb] = -1
                            vb0[0, node] = -1

                for node, nbs in enumerate(data['topo1'][0]):
                    for nb in nbs:
                        
                        center = ((kpt1t[0, node] + kpt1t[0, nb]) * 0.5).int()[0]
                        if vb1[0, nb] > 0  and vb1[0, node] > 0 and im0_erode[0,:, center[1], center[0]].mean() > 0.8:
                            vb1[0, nb] = -1
                            vb1[0, node] = -1

        

        kpt0t = kpts0 + motion_output0 / 2
        kpt1t = kpts1 + motion_output1 / 2

        

        if 'motion0' in data and 'motion1' in data:
            loss_motion = torch.nn.functional.l1_loss(motion_pred0, data['motion0'][:, :mmax]) +\
                torch.nn.functional.l1_loss(motion_pred1, data['motion1'][:, :nmax])
            

            EPE0 = ((motion_pred0 - data['motion0'][:, :mmax]) ** 2).sum(dim=-1).sqrt()
            EPE1 = ((motion_pred1 - data['motion1'][:, :nmax]) ** 2).sum(dim=-1).sqrt()
            # print(EPE0.size(), 'fdsafdsa')

            EPE = (EPE0.mean() + EPE1.mean()) * 0.5

            if 'visibility0' in data and 'visibility1' in data:
                loss_visibility = torch.nn.functional.binary_cross_entropy_with_logits(vb0[:, :mmax].view(-1, 1), data['visibility0'][:, :mmax].view(-1, 1), pos_weight=vb0.new_tensor([self.pos_weight])) + \
                torch.nn.functional.binary_cross_entropy_with_logits(vb1[:, :nmax].view(-1, 1), data['visibility1'][:, :nmax].view(-1, 1), pos_weight=vb0.new_tensor([self.pos_weight]))
            
                VB_Acc = ((((vb0 > 0).float() == data['visibility0'][:, :mmax]).float().sum() + ((vb1 > 0).float() == data['visibility1'][:, :nmax]).float().sum()) * 1.0 / (mmax + nmax))
            else:
                loss_visibility = 0
                VB_Acc = EPE.new_zeros([1])
            loss = loss_motion + 10 * loss_visibility

            loss_mean = torch.mean(loss)

            b, _, _ = motion_pred0.size()

            return {
                'keypoints0t': kpt0t,
                'keypoints1t': kpt1t,
                'vb0': (vb0 > 0).float(),
                'vb1': (vb1 > 0).float(),
                'r0': motion_output0,
                'r1': motion_output1,
                'loss': loss_mean,
                'EPE': EPE,
                'Visibility Acc': VB_Acc
            }
        else:
            return {
                'loss': -1,
                'skip_train': True,
                'keypointst0': kpts0 + motion_output0,
                'keypointst1': kpts1 + motion_output1,
                'vb0': vb0,
                'vb1': vb1,
            }


if __name__ == '__main__':

    args = argparse.Namespace()
    args.batch_size = 2
    args.gap = 5
    args.type = 'train'
    args.model = None
    args.action = None
    ss = Refiner()


    loader = fetch_dataloader(args)
    # #print(len(loader))
    for data in loader:
        # p1, p2, s1, s2, mi = data
        dict1 = data

        kp1 = dict1['keypoints0']
        kp2 = dict1['keypoints1']
        p1 = dict1['image0']
        p2 = dict1['image1']  

        # #print(s1)
        # #print(s1.type)
        mi = dict1['m01']
        fname = dict1['file_name'] 
        print(dict1['keypoints0'].size(), dict1['keypoints1'].size(), dict1['m01'].size(), dict1['motion0'].size(), dict1['mask0'].size())
        # print(kp1.shape, p1.shape, mi.shape)  
        # #print(mi.size())  
        # #print(mi)
        # break

        a = ss(data)
        print(dict1['file_name'])
        print(a['loss'])
        print(a['EPE'], a['Visibility Acc'],flush=True)
        a['loss'].backward()