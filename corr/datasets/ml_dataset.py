import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
# import networkx as nx
import os
import math
import random
from glob import glob
import os.path as osp

import sys
import argparse
import cv2
from collections import Counter

import json
import sknetwork
from sknetwork.embedding import Spectral

def read_json(file_path):
    """
        input: json file path
        output: 2d vertex, connections, and index numbers in original 3D space
    """

    with open(file_path) as file:
        data = json.load(file)
        vertex2d = np.array(data['vertex location'])
        
        topology = data['connection']
        index = np.array(data['original index'])

    return vertex2d, topology, index

def ids_to_mat(id1, id2):
    """
        inputs are two list of vertex index in original 3D mesh
    """
    corr1 = np.zeros(len(id1)) - 1.0
    corr2 = np.zeros(len(id2)) - 1.0

    id1 = np.array(id1).astype(int)[:, None]
    id2 = np.array(id2).astype(int)
    
    mat = (id1 == id2)

    pos12 = np.arange(len(id2))[None].repeat(len(id1), 0)
    pos21 = np.arange(len(id1))[None].repeat(len(id2), 0)
    corr1[mat.astype(int).sum(1).astype(bool)] = pos12[mat]
    corr2[mat.transpose().astype(int).sum(1).astype(bool)] = pos21[mat.transpose()]

    return mat, corr1, corr2

def adj_matrix(topology):
    """
        topology is the adj table; returns adj matrix
    """
    gsize = len(topology)
    adj = np.zeros((gsize, gsize)).astype(float)
    for v in range(gsize):
        adj[v][v] = 1.0
        for nb in topology[v]:
            adj[v][nb] = 1.0
            adj[nb][v] = 1.0
    return adj

class MixamoLineArt(data.Dataset):
    def __init__(self, root, gap=0, split='train', model=None, action=None, mode='train', max_len=3050, use_vs=False):
        """
            input:
                root: the root folder of the line art data
                gap: how many frames between two frames
                split: train or test
                model: indicate a specific character (default None)
                action: indicate a specific action (default None)
        """
        super(MixamoLineArt, self).__init__()


        if model == 'None':
            model = None
        if action == 'None':
            action = None

        self.is_train = True if mode == 'train' else False
        self.is_eval = True if mode == 'eval' else False
        # self.is_train = False
        self.max_len = max_len

        self.image_list = []
        self.label_list = []
        
        if use_vs:
            label_root = osp.join(root, split, 'labels_vs')
        else:
            label_root = osp.join(root, split, 'labels')
        image_root = osp.join(root, split, 'frames')
        self.spectral = Spectral(64,  normalized=False)

        for clip in os.listdir(image_root):
            skip = False
            if model != None:
                for mm in model:
                    if mm in clip:
                        skip = True
                
            if action != None:
                for aa in action:
                    if aa in clip:
                        skip = True
            if skip:
                continue
            image_list = sorted(glob(osp.join(image_root, clip, '*.png')))
            label_list = sorted(glob(osp.join(label_root, clip, '*.json')))
            if len(image_list) != len(label_list):
                print(image_root, flush=True)
                continue
            for i in range(len(image_list) - (gap+1)):
                self.image_list += [ [image_list[i], image_list[i+gap+1]] ]
            for i in range(len(label_list) - (gap+1)):
                self.label_list += [ [label_list[i], label_list[i+gap+1]] ]
        # print(clip)
        print('Len of Frame is ', len(self.image_list))
        print('Len of Label is ', len(self.label_list))

    def __getitem__(self, index):

        # load image/label files
        # image crop to a square, 2d label same operation
        # index to index matching
        # spectral embedding

        # test does not need index matching
        
        index = index % len(self.image_list)
        file_name = self.label_list[index][0][:-4]
  
        img1 = cv2.imread(self.image_list[index][0])
        img2 = cv2.imread(self.image_list[index][1])
        v2d1, topo1, id1 = read_json(self.label_list[index][0])
        v2d2, topo2, id2 = read_json(self.label_list[index][1])
        for ii in range(len(topo1)):
            # if not len(topo1[ii]):
            topo1[ii].append(ii)
        for ii in range(len(topo2)):
            topo2[ii].append(ii)

    
        m, n = len(v2d1), len(v2d2)

        # img1, v2d1 = crop_img(img1, np.array(v2d1))
        # img2, v2d2 = crop_img(img2, np.array(v2d2))

        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
        
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() * 2 / 255.0 - 1.0 
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() * 2 / 255.0 - 1.0

        v2d1 = torch.from_numpy(v2d1)
        v2d2 = torch.from_numpy(v2d2)

        v2d1[v2d1 > 719] = 719
        v2d1[v2d1 < 0] = 0
        v2d2[v2d2 > 719] = 719
        v2d2[v2d2 < 0] = 0


        adj1 = sknetwork.data.from_adjacency_list(topo1, matrix_only=True, reindex=False).toarray()
        adj2 = sknetwork.data.from_adjacency_list(topo2, matrix_only=True, reindex=False).toarray()

        # note here we compute the spectral embedding of adj matrix in data loading period
        # since it needs cpu computation and is not friendy to our cluster's computation
        # put them here to use multi-cpu pre-computing before network forward flow
        spec0, spec1 = np.abs(self.spectral.fit_transform(adj1)), np.abs(self.spectral.fit_transform(adj2))

        mat_index, corr1, corr2 = ids_to_mat(id1, id2)
        mat_index = torch.from_numpy(mat_index).float()
        corr1 = torch.from_numpy(corr1).float()
        corr2 = torch.from_numpy(corr2).float()
        if self.is_train:
        # if False:
            v2d1 = torch.nn.functional.pad(v2d1, (0, 0, 0, self.max_len - m), mode='constant', value=0)
            v2d2 = torch.nn.functional.pad(v2d2, (0, 0, 0, self.max_len - n), mode='constant', value=0)
            corr1 = torch.nn.functional.pad(corr1, (0, self.max_len - m), mode='constant', value=0)
            corr2 = torch.nn.functional.pad(corr2, (0, self.max_len - n), mode='constant', value=0)

            mask0, mask1 = torch.zeros(self.max_len).float(), torch.zeros(self.max_len).float()
            mask0[:m] = 1
            mask1[:n] = 1
        else:
            mask0, mask1 = torch.ones(m).float(), torch.ones(n).float()

        # not return id anymore. too slow
        if self.is_eval:
            return{
                'keypoints0': v2d1,
                'keypoints1': v2d2,
                'topo0': [topo1],
                'topo1': [topo2],
                # 'id0': id1,
                # 'id1': id2,
                'adj_mat0': spec0,
                'adj_mat1': spec1,
                'image0': img1,
                'image1': img2,

                'all_matches': corr1,
                'm01': corr1,
                'm10': corr2,
                'ms': m,
                'ns': n,
                'mask0': mask0,
                'mask1': mask1,
                'file_name': file_name,
                # 'with_match': True
            } 
        if not self.is_train:
            return{
                'keypoints0': v2d1,
                'keypoints1': v2d2,
                # 'topo0': topo1,
                # 'topo1': topo2,
                # 'id0': id1,
                # 'id1': id2,
                'adj_mat0': spec0,
                'adj_mat1': spec1,
                'image0': img1,
                'image1': img2,

                'all_matches': corr1,
                'm01': corr1,
                'm10': corr2,
                'ms': m,
                'ns': n,
                'mask0': mask0,
                'mask1': mask1,
                'file_name': file_name,
                # 'with_match': True
            } 
        else:
            return{
                'keypoints0': v2d1,
                'keypoints1': v2d2,
                # 'topo0': topo1,
                # 'topo1': topo2,
                # 'id0': id1,
                # 'id1': id2,
                'adj_mat0': spec0,
                'adj_mat1': spec1,
                'image0': img1,
                'image1': img2,

                'all_matches': corr1,
                'm01': corr1,
                'm10': corr2,
                'ms': m,
                'ns': n,
                'mask0': mask0,
                'mask1': mask1,
                'file_name': file_name,
                # 'with_match': True
            } 

        

    def __rmul__(self, v):
        self.index_list = v * self.index_list
        self.seg_list = v * self.seg_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def fetch_dataloader(args, type='train',):
    lineart = MixamoLineArt(root=args.root if hasattr(args, 'root') else '/mnt/lustre/syli/inbetween/data/12by12/ml144_norm_100_44_split/', gap=args.gap, split=args.type, model=args.model, action=args.action, mode=args.mode if hasattr(args, 'mode') else 'train', use_vs=args.use_vs if hasattr(args, 'use_vs') else False)
    train_loader = data.DataLoader(lineart, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=8, drop_last=True, worker_init_fn=worker_init_fn)

    if args.mode != 'train':
        loader = data.DataLoader(lineart, batch_size=args.batch_size, 
            pin_memory=True, shuffle=False, num_workers=8)

    return train_loader


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = argparse.Namespace()
    # args.subset = 'agent'
    args.batch_size = 1
    args.gap = 5
    args.type = 'test'
    args.model = ['ganfaul', 'firlscout', 'jolleen', 'kachujin', 'knight', 'maria', 'michelle', 'peasant', 'timmy', 'uriel']
    args.action = ['hip_hop', 'slash']
    # args.model = None
    # args.action = None
    args.use_vs = False
    # args.model = ['warrok', 'police']
    args.action = ['breakdance', 'capoeira', 'chapa-', 'fist_fight', 'flying', 'climb', 'running', 'reaction', 'magic', 'tripping']
        
    args.mode = 'eval'
    args.root='/mnt/lustre/syli/inbetween/data/12by12/ml144_norm_100_44_split/'
    # args.stage = 'anime'
    # args.image_size = (368, 368)
    # lineart = MixamoLineArt(root='/mnt/lustre/syli/inbetween/data/12by12/ml144/', gap=0, split='train')
    lineart = fetch_dataloader(args)
    # lineart = MixamoLineArt(root='/mnt/cache/syli/inbetween/data/ml100_norm/', gap=args.gap, split=args.type, model=args.model, action=args.action, mode=args.mode if hasattr(args, 'mode') else 'train')
    # train_loader = data.DataLoader(lineart, batch_size=args.batch_size, 

    percentage = 0.0
    vertex_num = 0.0
    vertex_shift = 0.0
    vertex_max_shift = 0.0
    edges = 0.0
    # for data in loader:
    #     print(data)
    #     break
    unmatched_all = []
    max_node = 0
    for dict in lineart:
        # print(dict['file_name'])
        # print(dict['file_name'][0], flush=True)
        v2d1 = dict['keypoints0'].numpy().astype(int)[0]
        v2d2 = dict['keypoints1'].numpy().astype(int)[0]

        ms = dict['ms'][0]
        ns = dict['ns'][0]
        # this_edges 
        topo = dict['topo0'][0]
        for ii in range(len(topo)):
            edges += len(topo[ii])
        # print(len(topo), flush=True)


        # print(ms, ns, flush=True)
        # print(dict['keypoints0'], flush=True)
        # print(dict['image0'].size(), flush=True)
        v2d1 = v2d1[:ms]
        v2d2 = v2d2[:ns]
        m01 = dict['m01'][0][:ms]
        # print(m01.shape)
        # print(np.arange(len(m01))[m01 != -1], m01[m01 != -1])
        # print(v2d2.shape, v2d1.shape)
        shift = np.sqrt(((v2d2[m01[m01 != -1].int(), :] * 1.0 - v2d1[np.arange(len(m01))[m01 != -1],:]) ** 2).sum(-1))
        vertex_num += len(v2d1)
        vertex_shift += shift.mean()
        vertex_max_shift += shift.max()
        percentage += ((m01!=-1).float().sum() * 1.0 / len(m01) * 100.0)
    
    print('>>>> gap=', args.gap, ' percentage=', percentage / len(lineart), ' vertex num=', vertex_num*1.0/len(lineart), 'edges num=', edges*1.0/len(lineart)/2, 'vertex shift=', vertex_shift/len(lineart), ' vertex max shift=', vertex_max_shift/len(lineart), flush=True)
        

        # if len(v2d1) > max_node:
        #     max_node = len(v2d1)
        # if len(v2d2) > max_node:
        #     max_node = len(v2d2)
    # print(max_node)
        # print(v2d1.shape)
        # img1 = ((dict['image0'][0].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(np.uint8).copy()
        # img2 = ((dict['image1'][0].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(np.uint8).copy()

        # # print(v2d1.shape, img1.shape, flush=True)

        # for node, nbs in enumerate(dict['topo0']):
        #     for nb in nbs:
        #         cv2.line(img1, [v2d1[node][0], v2d1[node][1]], [v2d1[nb][0], v2d1[nb][1]], [255, 180, 180], 2)
        # colors1, colors2 = {}, {}

        # id1 = dict['id0'][0].numpy()
        # id2 = dict['id1'][0].numpy()
        # for index in id1:
        #     # print(index)
        #     color = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
        #     # for ii in index:
        #     colors1[index] = color
        
        # colors1, colors2 = {}, {}


        # for index in id1:
        #     # print(index)
        #     color = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
        #     colors1[index] = color

        # for i, p in enumerate(v2d1):
        #     ii = id1[i]
        #     # print(ii)
        #     cv2.circle(img1, [int(p[0]), int(p[1])], 1, colors1[ii], 2)

        # unmatched = 0
        # for ii in id2:
        #     color = [0, 0, 0]
        #     this_is_umatched = 1
        #     colors2[ii] = colors1[ii] if ii in colors1 else color
        #     if ii in colors1:
        #         this_is_umatched = 0
        #     # if ii not in colors1:
        #     unmatched += this_is_umatched

        # for i, p in enumerate(v2d2):
        #     ii = id2[i]
        #     # print(p)
        #     cv2.circle(img2, [int(p[0]), int( p[1])], 1, colors2[ii], 2)

        # # print('Unmatched in Img 2: ', , '%')
        # unmatched_all.append(100 - unmatched * 100.0/len(v2d2))

        # im_h = cv2.hconcat([img1, img2])
        # print('/mnt/lustre/syli/inbetween/AnimeInbetween/corr/datasets/data_check_norm/' + dict['file_name'][0].replace('/', '_') + '.jpg', flush=True)
        # cv2.imwrite('/mnt/lustre/syli/inbetween/AnimeInbetween/corr/datasets/data_check_norm/' + dict['file_name'][0].replace('/', '_') + '.jpg', im_h)

    # print(np.mean(unmatched_all))
 

