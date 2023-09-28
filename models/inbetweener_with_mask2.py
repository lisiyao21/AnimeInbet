from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
# from seg_desc import seg_descriptor
import argparse
import torch.nn.functional as F

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
        # print(inputs.size(), 'wula!')
        x = self.encoder(inputs)
        # print(x.size())
        return x


def attention(query, key, value, mask=None):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    if mask is not None:
        # print(mask, flush=True)
        scores = scores.masked_fill(mask==0, float('-inf'))

    # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
    # att = F.softmax(att, dim=-1)
    prob = torch.nn.functional.softmax(scores, dim=-1)

    # print(scores[1][1], prob[1][1], flush=True)
    # while True:
    #     pass 
    # prob = torch.exp(scores) /((torch.sum(torch.exp(scores), dim=-1)[:, :, :, None]) + 1e-7)
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
    # print(scores.min(), flush=True)
    if ms.size()[0] > 0:
        norm = norm[:, None]
        log_mu = torch.cat([norm.expand(b, m), ns.log()[:, None] + norm], dim=-1) # (m + 1)
        log_nu = torch.cat([norm.expand(b, n), ms.log()[:, None] + norm], dim=-1)
        # print(log_nu.min(), log_mu.min(), flush=True)
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


class SuperGlueM(nn.Module):
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
        # default_config.weights = 
        default_config.keypoint_encoder = [32, 64, 128]
        default_config.GNN_layers = ['self', 'cross'] * 9
        default_config.sinkhorn_iterations = 100
        default_config.match_threshold = 0.2
        # self.config = {**self.default_config, **config}

        if config is None:
            self.config = default_config
        else:
            self.config = config   
            self.config.GNN_layers = ['self', 'cross'] * self.config.GNN_layer_num
            # print('WULA!', self.config.GNN_layer_num)

        self.kenc = KeypointEncoder(
            self.config.descriptor_dim, self.config.keypoint_encoder)

        self.gnn = AttentionalGNN(
            self.config.descriptor_dim, self.config.GNN_layers)

        self.final_proj = nn.Conv1d(
            self.config.descriptor_dim, self.config.descriptor_dim,
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        self.vertex_desc = VertexDescriptor(self.config.descriptor_dim)

        # assert self.config.weights in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config.weights)
        # self.load_state_dict(torch.load(path))
        # print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #     self.config.weights))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        # print(data['segment0'].size())
        # desc0, desc1 = data['descriptors0'].float()(), data['descriptors1'].float()()
         # print(desc0.size())
        kpts0, kpts1 = data['keypoints0'].float(), data['keypoints1'].float()

        ori_mask0, ori_mask1 = data['mask0'].float(), data['mask1'].float()
        dim_m, dim_n = data['ms'].float(), data['ns'].float()

        mmax = dim_m.int().max()
        nmax = dim_n.int().max()

        mask0 = ori_mask0[:, :mmax]
        mask1 = ori_mask1[:, :nmax]

        kpts0 = kpts0[:, :mmax]
        kpts1 = kpts1[:, :nmax]

        desc0, desc1 = self.vertex_desc(data['image0'], kpts0.float()), self.vertex_desc(data['image1'], kpts1.float())
        
       
        # print(desc0.size(), flush=True)

        mask00 = torch.ones_like(mask0)[:, :, None] * mask0[:, None, :]
        # print(mask00[1], flush=True)
        
        mask11 = torch.ones_like(mask1)[:, :, None] * mask1[:, None, :]
        mask01 = torch.ones_like(mask0)[:, :, None] * mask1[:, None, :]
        mask10 = torch.ones_like(mask1)[:, :, None] * mask0[:, None, :]
        
        # desc0 = desc0.transpose(0,1)
        # desc1 = desc1.transpose(0,1)
        # kpts0 = torch.reshape(kpts0, (1, -1, 2))
        # kpts1 = torch.reshape(kpts1, (1, -1, 2))

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

        # file_name = data['file_name']
        all_matches = data['all_matches'] if 'all_matches' in data else None# shape = (1, K1)
        # .permute(1,2,0) # shape=torch.Size([1, 87,])
        
        # positional embedding
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        # print(data['file_name'])
        # print(kpts0.size())
    
        pos0 = self.kenc(kpts0)
        pos1 = self.kenc(kpts1)
        # print(desc0.size(), pos0.size())
        # print(desc0.size(), pos0.size())
        desc0 = desc0 + pos0
        desc1 = desc1 + pos1

        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                    #  .view(1, 1, config.block_size, config.block_size))
        # mask0 = ...
        # mask1 = ...

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1, mask00, mask11, mask01, mask10)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        # print(mdesc0.size(), mdesc1.size())
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores0 = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc0)
        scores1 = torch.einsum('bdn,bdm->bnm', mdesc1, mdesc1)
        # #print('here1!!', scores.size())

        # b k1 k2
        scores = scores / self.config.descriptor_dim**.5
        # print(scores.size(), mask01.size())
        # mask01 = mask0[:, :, None] * mask1[:, None, :]
        # scores = scores.masked_fill(mask01 == 0, float('-inf'))

        # print(scores.size())
        # Run the optimal transport.
        # print(dim_m.size(), dim_m, flush=True)
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config.sinkhorn_iterations,
            ms=dim_m, ns=dim_n)

        # print(scores)
        # print(scores.sum())
        # print(scores.sum(1))
        # print(scores.sum(0))

        # Get the matches with score above "match_threshold".
        return scores[:, :-1, :-1], scores0, scores1, mdesc0, mdesc1
       

def tensor_erode(bin_img, ksize=5):
    # 首先为原图加入 padding，防止腐蚀后图像尺寸缩小
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    # 将原图 unfold 成 patch
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k

    # 取每个 patch 中最小的值，i.e., 0
    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded

class InbetweenerM(nn.Module):
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
        self.corr = SuperGlueM(config.corr_model)
        self.mask_map = MLP([config.corr_model.descriptor_dim, 32, 1])
        self.pos_weight = config.pos_weight
        # self.motion_propagation = 
        
        # assert self.config.weights in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config.weights)
        # self.load_state_dict(torch.load(path))
        # print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #     self.config.weights))

    def forward(self, data):
        if 'gen_vid' in data:
            dim_m, dim_n = data['ms'].float(), data['ns'].float()
            mmax = dim_m.int().max()
            nmax = dim_n.int().max()
            # with torch.no_grad():
            #     self.corr.eval()
            score01, score0, score1, dec0, dec1 = self.corr(data)
            kpts0, kpts1 = data['keypoints0'][:,:mmax].float(), data['keypoints1'][:,:nmax].float() # BM2, BN2 
          ##  print(kpts0.mean(), kpts1.mean(), flush=True)

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

            adj0, adj1 = data['adj_mat0'].float(), data['adj_mat1'].float()

            motion_pred0 = torch.softmax(score01, dim=-1) @ kpts1 - kpts0
            motion_pred1 = torch.softmax(score01.transpose(1, 2), dim=-1) @ kpts0 - kpts1

            # score0.mask_off()

            motion_pred0 = torch.softmax(score0.masked_fill(adj0==0, float('-inf')), dim=-1) @ motion_pred0
            motion_pred1 = torch.softmax(score1.masked_fill(adj1==0, float('-inf')), dim=-1) @ motion_pred1
            
            vb0 = self.mask_map(dec0)[:, 0]
            vb1 = self.mask_map(dec1)[:, 0]
            vb0[:] = 1
            vb1[:] = 1

            im0_erode =  data['image0']
            im1_erode =  data['image1']
            im0_erode[im0_erode > 0] = 1
            im0_erode[im0_erode <= 0] = 0
            im1_erode[im1_erode > 0] = 1
            im1_erode[im1_erode <= 0] = 0
            
            im0_erode = tensor_erode(im0_erode, 3)
            im1_erode = tensor_erode(im1_erode, 3)

            motion_output0, motion_output1 =  motion_pred0.clone(), motion_pred1.clone()
          ##  print('>>>>> here', motion_pred0.mean(), motion_pred1.mean(), flush=True)
            kpt0t = kpts0 + motion_output0 * 1
            kpt1t = kpts1 + motion_output1 * 1
            if 'topo0' in data and 'topo1' in data:
              ##  print(len(data['topo0'][0]), len(data['topo1']), flush=True)
                for node, nbs in enumerate(data['topo0'][0]):
                    for nb in nbs:
                        # print(nb, flush=True)
                        # print(kpt0t.size(), 'fDsafdsafds', flush=True)
                        # if vb0[0, nb] and vb0[0, node] and ((kpt0t[0, node] - kpt0t[0, nb]) ** 2).sum() / (((kpts0[0, node] - kpts0[0, nb]) ** 2).sum() + 1e-7) > 3:
                        #     vb0[0, nb] = -1
                        #     vb0[0, node] = -1
                        # print(node.size())
                        center = ((kpt0t[0, node] + kpt0t[0, nb]) * 0.5).int()[0]
                        # print(center.size(), flush=True)
                        if vb0[0, nb] and vb0[0, node] and im1_erode[0,:, center[1], center[0]].mean() > 0.8:
                            vb0[0, nb] = -1
                            vb0[0, node] = -1
                        # center = ((kpt0t[0, node] + kpt0t[0, nb]) * 0.25).int()[0]
                        # # print(center.size(), flush=True)
                        # if vb0[0, nb] and vb0[0, node] and center[1] < 720 and center[0] < 720 and im1_erode[0,:, center[1], center[0]].mean() > 0.8:
                        #     vb0[0, nb] = -1
                        #     vb0[0, node] = -1
                        # center = ((kpt0t[0, node] + kpt0t[0, nb]) * 0.75).int()[0]
                        # # print(center.size(), flush=True)
                        # if vb0[0, nb] and vb0[0, node] and center[1] < 720 and center[0] < 720 and im1_erode[0,:, center[1], center[0]].mean() > 0.8:
                        #     vb0[0, nb] = -1
                        #     vb0[0, node] = -1
                for node, nbs in enumerate(data['topo1'][0]):
                    for nb in nbs:
                        
                        # if vb1[0, nb] and vb1[0, node] and ((kpt1t[0, node] - kpt1t[0, nb]) ** 2).sum() / (((kpts1[0, node] - kpts1[0, nb]) ** 2).sum() + 1e-7) >3:
                        #     vb1[0, nb] = -1
                        #     vb1[0, node] = -1
                        center = ((kpt1t[0, node] + kpt1t[0, nb]) * 0.5).int()[0]
                        if vb1[0, nb] and vb1[0, node] and im0_erode[0,:, center[1], center[0]].mean() > 0.95:
                            vb1[0, nb] = -1
                            vb1[0, node] = -1
                        # center = ((kpt1t[0, node] + kpt1t[0, nb]) * 0.25).int()[0]
                        # if vb1[0, nb] and vb1[0, node] and center[1] < 720 and center[0] < 720 and im0_erode[0,:, center[1], center[0]].mean() > 0.95:
                        #     vb1[0, nb] = -1
                        #     vb1[0, node] = -1
                        # center = ((kpt1t[0, node] + kpt1t[0, nb]) * 0.75).int()[0]
                        # if vb1[0, nb] and vb1[0, node] and center[1] < 720 and center[0] < 720 and im0_erode[0,:, center[1], center[0]].mean() > 0.95:
                        #     vb1[0, nb] = -1
                        #     vb1[0, node] = -1
            # print(vb0.mean(), vb1.mean(), flush=True)
            return {'r0': motion_output0, 'r1': motion_output1, 'vb0':(vb0 > 0).float(), 'vb1':(vb1 > 0).float(),}

        dim_m, dim_n = data['ms'].float(), data['ns'].float()
        mmax = dim_m.int().max()
        nmax = dim_n.int().max()
        # with torch.no_grad():
        #     self.corr.eval()
        score01, score0, score1, dec0, dec1 = self.corr(data)


        kpts0, kpts1 = data['keypoints0'][:,:mmax].float(), data['keypoints1'][:,:nmax].float() # BM2, BN2 


        adj0, adj1 = data['adj_mat0'].float(), data['adj_mat1'].float()

        motion_pred0 = torch.softmax(score01, dim=-1) @ kpts1 - kpts0
        motion_pred1 = torch.softmax(score01.transpose(1, 2), dim=-1) @ kpts0 - kpts1

        # score0.mask_off()

        motion_pred0 = torch.softmax(score0.masked_fill(adj0==0, float('-inf')), dim=-1) @ motion_pred0
        motion_pred1 = torch.softmax(score1.masked_fill(adj1==0, float('-inf')), dim=-1) @ motion_pred1
        
        vb0 = self.mask_map(dec0)[:, 0]
        vb1 = self.mask_map(dec1)[:, 0]

        # motion0_pred, vb0 = pred0[:, :2].permute(0, 2, 1), pred0[:, 2:][:, 0]
        # motion1_pred, vb1 = pred1[:, :2].permute(0, 2, 1), pred1[:, 2:][:, 0]
        
        # delta0, delta1 = motion_delta[:, :, :mmax].permute(0, 2, 1), motion_delta[:, :, mmax:].permute(0, 2, 1)
        # motion_output0, motion_output1 =  motion0 + delta0, motion1 + delta1
        motion_output0, motion_output1 =  motion_pred0.clone(), motion_pred1.clone()

        # print(delta0.max(), delta1.max())
        # vb0 = kpts0.new_ones(motion_pred0[:, :, 0].size()) + 1.0
        # vb1 = kpts1.new_ones(motion_pred1[:, :, 0].size()) + 1.0

        # vb0, vb1 = visibility[:, 0, :mmax], visibility[:, 0, mmax:]
        # mask0, mask1 = mask[:, :mmax].bool(), mask[:, mmax:].bool()
        # vb0_output = vb0.clone()
        # vb1_output = vb1.clone()

        # vb1_output[batch, corr01[corr01 != -1]] = 1.0

        # motion_output0[valid0.bool()] = motion0[valid0.bool()]
        # motion_output1[valid1.bool()] = motion1[valid1.bool()]

        # vb0_output[vb0_output >= 0] = 1.0
        # vb0_output[vb0_output < 0] = 0.0
        # vb1_output[vb1_output >= 0] = 1.0
        # vb1_output[vb1_output < 0 ] = 0.0

        

        kpt0t = kpts0 + motion_output0 / 2
        kpt1t = kpts1 + motion_output1 / 2
        # kpt1t[batch, corr01[corr01 != -1]] = kpt0t[corr01 != -1]
        
        
        ##################################################
        ##  Note Here the mini batch size is 1!!!!!!!!  ##
        ##################################################

        if 'topo0' in data and 'topo1' in data:
            # print(len(data['topo0'][0]), len(data['topo1']), flush=True)
            for node, nbs in enumerate(data['topo0'][0]):
                for nb in nbs:
                    if vb0[0, nb] and vb0[0, node] and ((kpt0t[0, node] - kpt0t[0, nb]) ** 2).sum() / (((kpts0[0, node] - kpts0[0, nb]) ** 2).sum() + 1e-7) > 5:
                        vb0[0, nb] = -1
                        vb0[0, node] = -1
            for node, nbs in enumerate(data['topo1'][0]):
                for nb in nbs:
                    if vb1[0, nb] and vb1[0, node] and ((kpt1t[0, node] - kpt1t[0, nb]) ** 2).sum() / (((kpts1[0, node] - kpts1[0, nb]) ** 2).sum() + 1e-7) > 5:
                        vb1[0, nb] = -1
                        vb1[0, node] = -1

        if 'motion0' in data and 'motion1' in data:
            # valid_motion0 = motion_output0[mask0[:, :, None].repeat(1, 1, 2)]
            # gt_valid_motion0 = data['motion0'][:, :mmax][mask0[:, :, None].repeat(1, 1, 2)].float()
            # valid_motion1 = motion_output1[mask1[:, :, None].repeat(1, 1, 2)]
            # gt_valid_motion1 = data['motion1'][:, :nmax][mask1[:, :, None].repeat(1, 1, 2)].float()

            loss_motion = torch.nn.functional.l1_loss(motion_pred0, data['motion0'][:, :mmax]) +\
                torch.nn.functional.l1_loss(motion_pred1, data['motion1'][:, :nmax])
            
            # loss_valid0 = ((corr01 == -1) & (mask0 == 1))
            # loss_valid1 = ((corr10 == -1) & (mask1 == 1))
            EPE0 = ((motion_pred0 - data['motion0'][:, :mmax]) ** 2).sum(dim=-1).sqrt()
            EPE1 = ((motion_pred1 - data['motion1'][:, :nmax]) ** 2).sum(dim=-1).sqrt()
            # print(EPE0.size(), 'fdsafdsa')

            EPE = (EPE0.mean() + EPE1.mean()) * 0.5
            # print(len(EPE0[mask0]), len(EPE1[mask1]))
            # print(vb0[:, :mmax][mask0], vb0[:, :mmax][mask0].shape, data['visibility0'][:, :mmax][mask0], data['visibility0'][:, :mmax][mask0].shape)
            # print(.size())
            # print((vb0[:, :mmax] > 0).float().sum(), data['visibility0'][:, :mmax].float().sum())
            # pos_weight=vb0.new_tensor([0.5])
            if 'visibility0' in data and 'visibility1' in data:
                loss_visibility = torch.nn.functional.binary_cross_entropy_with_logits(vb0[:, :mmax].view(-1, 1), data['visibility0'][:, :mmax].view(-1, 1), pos_weight=vb0.new_tensor([self.pos_weight])) + \
                torch.nn.functional.binary_cross_entropy_with_logits(vb1[:, :nmax].view(-1, 1), data['visibility1'][:, :nmax].view(-1, 1), pos_weight=vb0.new_tensor([self.pos_weight]))
            
                VB_Acc = ((((vb0 > 0).float() == data['visibility0'][:, :mmax]).float().sum() + ((vb1 > 0).float() == data['visibility1'][:, :nmax]).float().sum()) * 1.0 / (mmax + nmax))
            else:
                loss_visibility = 0
                VB_Acc = EPE.new_zeros([1])
            loss = loss_motion + 10 * loss_visibility

            loss_mean = torch.mean(loss)
            # loss_mean = torch.reshape(loss_mean, (1, -1))
            # print(loss_mean, flush=True)

            # print(all_matches[:, :mmax].size(), indices0.size(), mask0.size(), flush=True)
            #print((all_matches[0] == indices0[0]).sum())

            # print(vb1.size(),corr01.size())

            # kpt0t = torch.nn.functional.pad(kpts0 + motion_output0, (0, 0, 0, self.max_len - mmax, 0, 0), mode='constant', value=0)
            # kpt1t = torch.nn.functional.pad(kpts1 + motion_output1, (0, 0, 0, self.max_len - nmax, 0, 0), mode='constant', value=0),

            # kpt1t[:, :nmax][batch, corr01[corr01 != -1]] = kpt0t[:, :mmax][corr01 != -1]

            b, _, _ = motion_pred0.size()
            # batch = torch.arange(b)[:, None].repeat(1, mmax)[corr01 != -1].long()
            # # print(kpts0[corr01 != -1].size(), corr01[corr01 != -1].size())
            # matched_intermediate = (kpts0[(corr01 != -1)] + kpts1[batch, corr01[corr01 != -1].long(), :]) * 0.5
            # motion0[corr01 != -1] = matched_intermediate - kpts0[corr01 != -1]
            # motion1[batch, corr01[corr01 != -1].long(), :] = matched_intermediate - kpts1[batch, corr01[corr01 != -1].long(), :]

            # vb0 = torch.nn.functional.pad(vb0, (0, self.max_len - mmax, 0, 0), mode='constant', value=0),
            # vb1 = torch.nn.functional.pad(vb1, (0, self.max_len - nmax, 0, 0), mode='constant', value=0),

            # self.max_len = 3050
            # VB_Acc = ((((vb0 > 0.5).float() == data['visibility0'][:, :mmax]).float().sum() + ((vb1 > 0.5).float() == data['visibility1'][:, :nmax]).float().sum()) * 1.0 / (mmax + nmax))
                
            return {
                # 'matches0': indices0, # use -1 for invalid match
                # 'matches1': indices1[0], # use -1 for invalid match
                # 'matching_scores0': mscores0,
                # 'matching_scores1': mscores1[0],
                # 'keypointst0': torch.nn.functional.pad(kpts0 + motion_output0, (0, 0, 0, self.max_len - mmax, 0, 0), mode='constant', value=0),
                # 'keypointst1': torch.nn.functional.pad(kpts1 + motion_output1, (0, 0, 0, self.max_len - nmax, 0, 0), mode='constant', value=0),
                # 'vb0': torch.nn.functional.pad(vb0, (0, self.max_len - mmax, 0, 0), mode='constant', value=0),
                # 'vb1': torch.nn.functional.pad(vb1, (0, self.max_len - nmax, 0, 0), mode='constant', value=0),
                'keypoints0t': kpt0t,
                'keypoints1t': kpt1t,
                'vb0': (vb0 > 0).float(),
                'vb1': (vb1 > 0).float(),
                'loss': loss_mean,
                'EPE': EPE,
                'Visibility Acc': VB_Acc
                # ((((vb0[mask0] > 0).float() == data['visibility0'][:, :mmax][mask0]).float().sum() + ((vb1[mask1] > 0).float() == data['visibility1'][:, :nmax][mask1]).float().sum()) * 1.0 / (mask0.float().sum() + mask1.float().sum())),
                # 'skip_train': [False],
                # 'accuracy': (((all_matches[:, :mmax] == indices0) & mask0.bool()).sum() / mask0.sum()).item(),
                # 'valid_accuracy': (((all_matches[:, :mmax] == indices0) & (all_matches[:, :mmax] != -1) & mask0.bool()).float().sum() / ((all_matches[:, :mmax] != -1) & mask0.bool()).float().sum()).item(),
            }
        else:
            return {
                'loss': -1,
                'skip_train': True,
                'keypointst0': kpts0 + motion_output0,
                'keypointst1': kpts1 + motion_output1,
                'vb0': vb0,
                'vb1': vb1,
                # 'accuracy': -1,
                # 'area_accuracy': -1,
                # 'valid_accuracy': -1,
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