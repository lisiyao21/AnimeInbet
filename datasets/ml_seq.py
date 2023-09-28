import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

import sys
import argparse
import cv2
from collections import Counter
import time
import json
import sknetwork
from sknetwork.embedding import Spectral

import scipy

def read_json(file_path):
    """
        input: json file path
        output: 2d vertex, connections and vertex index in original 3D domain
    """

    with open(file_path) as file:
        data = json.load(file)
        vertex2d = np.array(data['vertex location'])
        
        topology = data['connection']
        index = np.array(data['original index'])

    return vertex2d, topology, index


def matched_motion(v2d1, v2d2, match12, motion_pre=None):
    motion = np.zeros_like(v2d1)

    motion[match12 != -1] = v2d2[match12[match12 != -1]] - v2d1[match12 != -1]
    if motion_pre is not None:
        motion[match12 != -1] = motion[match12 != -1] + motion_pre[match12[match12 != -1]]
    return motion

def unmatched_motion(topo1, v2d1, motion12, match12):
    pos = np.arange(len(topo1))
    masked = (match12 == -1)

    round = 0
    former_len = 0
    while(len(pos[masked]) > 0):
        this_len = len(pos[masked])
        if former_len == this_len:
            break
        former_len = this_len
        round += 1
        for v in pos[masked]:
            unmatched = masked[topo1[v]]

            if unmatched.sum() != len(topo1[v]):
                motion12[v] = np.average(motion12[topo1[v]][np.invert(unmatched)], axis=0)
                masked[v] = False

                
    if len(pos[masked] > 0):
        # find the neast point for each unlabeled point
        index = ((v2d1[pos[masked]][:, None, :] - v2d1[pos[np.invert(masked)]]) ** 2).sum(2).argmin(1)
        motion12[pos[masked]] = motion12[pos[np.invert(masked)]][index]
        masked[pos[masked]] = False

    return motion12


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

class MixamoLineArtMotionSequence(data.Dataset):
    def __init__(self, root, gap=0, split='train', model=None, action=None, mode='train', use_vs=False, max_len=3050):
        """
            input:
                root: the root folder of the line art data
                gap: how many frames between two frames. gap should be an odd numbe.
                split: train or test
                model: indicate a specific character (default None)
                action: indicate a specific action (default None)

            output:
                image of sources (0, 1) and output (0.5)
                topo0, topo1
                v2d0, v2d1
                
                corr12, corr21

                motion0-->0.5, motion1-->0.5
                visibility0-->0.5, visibility   1-->0.5

        """
        super(MixamoLineArtMotionSequence, self).__init__()

        self.gap = gap
        if model == 'None':
            model = None
        if action == 'None':
            action = None

        assert(gap%2 != 0)

        self.is_train = True if mode == 'train' else False
        self.is_eval = True if mode == 'eval' else False
        # self.is_train = False
        self.max_len = max_len

        self.image_list = []
        self.label_list = []

        label_root = osp.join(root, split, 'labels')
        self.use_vs = False
        if use_vs:
            print('>>>>>>>> Using VS labels')
            self.use_vs = True
            label_root = osp.join(root, split, 'labels_vs')
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
                print(clip, flush=True)
                continue
            for i in range(len(image_list) - (gap+1)):
                self.image_list += [ [image_list[jj] for jj in range(i, i + gap + 2)] ]
            for i in range(len(label_list) - (gap+1)):
                self.label_list += [ [label_list[jj] for jj in range(i, i + gap + 2)] ]
        # print(clip)
        print('Len of Frame is ', len(self.image_list), flush=True)
        print('Len of Label is ', len(self.label_list), flush=True)

    def __getitem__(self, index):


        # load image/label files
        # load labels: 
        #   (a) read json (b) load image (c) make pseudo labels

        # image crop to a square (720x720) before input, 2d label same operation
        # index to index matching

        # test does not need index matching
        
        index = index % len(self.image_list)
        file_name = self.label_list[index][len(self.label_list[index])//2][:-4]

        imgt = [cv2.imread(self.image_list[index][ii]) for ii in range(0, len(self.image_list[index]))]

        labelt = []
        for ii in range(0, len(self.label_list[index])):
            v, t, id = read_json(self.label_list[index][ii])
            v[v > imgt[0].shape[0] - 1] = imgt[0].shape[0] - 1
            v[v < 0] = 0
            labelt.append({'keypoints': v.astype(int), 'topo': t, 'id': id})

        # make motion pseudo label
        motion = None
        motion01 = None

        start_frame = 0
        gap = self.gap // 2 + 1


        ######### forward direction
        for ii in reversed(range(start_frame + 1, start_frame + 2*gap + 1)):
            img1 = imgt[ii - 1]
            img2 = imgt[ii] 

            v2d1 = labelt[ii - 1]['keypoints'].astype(int)
            v2d2 = labelt[ii]['keypoints'].astype(int)

            topo1 = labelt[ii - 1]['topo']
            topo2 = labelt[ii ]['topo']

            id1 = labelt[ii - 1]['id']
            id2 = labelt[ii]['id']

            if self.use_vs:
                id1 = np.arange(len(id1))
                id2 = np.arange(len(id2))

            _, match12, matc21 = ids_to_mat(id1, id2)

            if ii <= start_frame + gap:
                motion01 = matched_motion(v2d1, v2d2, match12.astype(int), motion01)
                motion01 = unmatched_motion(topo1, v2d1, motion01, match12.astype(int))

            motion = matched_motion(v2d1, v2d2, match12.astype(int), motion)
            motion = unmatched_motion(topo1, v2d1, motion, match12.astype(int))
        motion0 = motion.copy()
 
        img2 = imgt[start_frame + gap]
        
        v2d1 = labelt[start_frame]['keypoints'].astype(int)
        source0_topo = labelt[start_frame]['topo']

        target = cv2.erode(img2, np.ones((3, 3), np.uint8), iterations=1)

        shift_plabel = v2d1 + motion01
        visible = np.ones(len(v2d1)).astype(float)
        visible[shift_plabel[:, 0] < 0] = 0
        visible[shift_plabel[:, 0] >= imgt[0].shape[0]] = 0
        visible[shift_plabel[:, 1] < 0] = 0
        visible[shift_plabel[:, 1] >= imgt[0].shape[0]] = 0

        # vertex visibility
        visible[visible == 1] = (target[:, :, 0][shift_plabel[visible == 1][:, 1], shift_plabel[visible == 1][:, 0]] < 255 ).astype(float)

        visible01 = visible.copy()
        v2d1s = shift_plabel

        # edge visibility
        for node, nbs in enumerate(source0_topo):
            for nb in nbs:
                if visible01[nb] and visible01[node] and ((v2d1s[node] - v2d1s[nb]) ** 2).sum() / (((v2d1[node] - v2d1[nb]) ** 2).sum() + 1e-7) > 25:
                    visible01[nb] = False
                    visible01[node] = False

        ######## backward direction
        motion = None
        motion21 = None

        for ii in range(start_frame + 1, start_frame + gap + gap + 1):
            img2 = imgt[ii - 1]
            img1 = imgt[ii] 

            v2d2 = labelt[ii - 1]['keypoints'].astype(int)
            v2d1 = labelt[ii]['keypoints'].astype(int)

            topo2 = labelt[ii - 1]['topo']
            topo1 = labelt[ii ]['topo']

            
            id1 = labelt[ii]['id']
            id2 = labelt[ii - 1]['id']
            if self.use_vs:
                id1 = np.arange(len(id1))
                id2 = np.arange(len(id2))
            _, match12, _ = ids_to_mat(id1, id2)

            if ii >= start_frame + gap + 1:
                motion21 = matched_motion(v2d1, v2d2, match12.astype(int), motion21)
                motion21 = unmatched_motion(topo1, v2d1, motion21, match12.astype(int))

            motion = matched_motion(v2d1, v2d2, match12.astype(int), motion)
            motion = unmatched_motion(topo1, v2d1, motion, match12.astype(int))

        motion2 = motion.copy()
        
        img1 = imgt[start_frame + 2*gap]
        img2 = imgt[start_frame + gap]
        
        v2d1 = labelt[start_frame + 2*gap]['keypoints'].astype(int)
        source2_topo = labelt[start_frame + 2*gap]['topo']

        shift_plabel = v2d1 + motion21
        visible = np.ones(len(v2d1)).astype(float)
        visible[shift_plabel[:, 0] < 0] = 0
        visible[shift_plabel[:, 0] >= imgt[0].shape[0]] = 0
        visible[shift_plabel[:, 1] < 0] = 0
        visible[shift_plabel[:, 1] >= imgt[0].shape[0]] = 0

        visible[visible == 1] = (target[:, :, 0][shift_plabel[visible == 1][:, 1], shift_plabel[visible == 1][:, 0]] < 255 ).astype(float)

        visible21 = visible.copy()

        v2d1s = shift_plabel

        for node, nbs in enumerate(source2_topo):
            for nb in nbs:
                if visible21[nb] and visible21[node] and ((v2d1s[node] - v2d1s[nb]) ** 2).sum() / (((v2d1[node] - v2d1[nb]) ** 2).sum() + 1e-7) > 25:
                    visible21[nb] = False
                    visible21[node] = False


        ###### prepare other data
        img2 = imgt[-1]
        img1 = imgt[0] 

        v2d2 = labelt[-1]['keypoints'].astype(int)
        v2d1 = labelt[0]['keypoints'].astype(int)

        topo2 = labelt[-1]['topo']
        topo1 = labelt[0]['topo']

        m, n = len(v2d1), len(v2d2)

        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() * 2 / 255.0 - 1.0 
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() * 2 / 255.0 - 1.0
        imgt = torch.from_numpy(imgt[start_frame + gap]).permute(2, 0, 1).float() * 2 / 255.0 - 1.0 

        v2d1 = torch.from_numpy(v2d1)
        v2d2 = torch.from_numpy(v2d2)

        visible01 = torch.from_numpy(visible01)
        visible21 = torch.from_numpy(visible21)
        motion0 = torch.from_numpy(motion0)
        motion2 = torch.from_numpy(motion2)

        v2d1[v2d1 > imgt[0].shape[0] - 1 ] = imgt[0].shape[0] - 1
        v2d1[v2d1 < 0] = 0
        v2d2[v2d2 > imgt[0].shape[1] - 1] = imgt[0].shape[1] - 1
        v2d2[v2d2 < 0] = 0

        
        id1 = labelt[start_frame]['id']
        id2 = labelt[-1]['id']
        if self.use_vs:
            id1 = np.arange(len(id1))
            id2 = np.arange(len(id2))

        mat_index, corr1, corr2 = ids_to_mat(id1, id2)
        mat_index = torch.from_numpy(mat_index).float()
        corr1 = torch.from_numpy(corr1).float()
        corr2 = torch.from_numpy(corr2).float()

        if self.is_train:
            v2d1 = torch.nn.functional.pad(v2d1, (0, 0, 0, self.max_len - m), mode='constant', value=0)
            v2d2 = torch.nn.functional.pad(v2d2, (0, 0, 0, self.max_len - n), mode='constant', value=0)
            corr1 = torch.nn.functional.pad(corr1, (0, self.max_len - m), mode='constant', value=0)
            corr2 = torch.nn.functional.pad(corr2, (0, self.max_len - n), mode='constant', value=0)
            motion0 = torch.nn.functional.pad(motion0, (0, 0, 0, self.max_len - m), mode='constant', value=0)
            motion2 = torch.nn.functional.pad(motion2, (0, 0, 0, self.max_len - n), mode='constant', value=0)
            visible01 = torch.nn.functional.pad(visible01, (0, self.max_len - m), mode='constant', value=0)
            visible21 = torch.nn.functional.pad(visible21, (0, self.max_len - n), mode='constant', value=0)

            mask0, mask1 = torch.zeros(self.max_len).float(), torch.zeros(self.max_len).float()
            mask0[:m] = 1
            mask1[:n] = 1
        else:
            mask0, mask1 = torch.ones(m).float(), torch.ones(n).float()
        
        for ii in range(len(topo1)):
            # if not len(topo1[ii]):
            topo1[ii].append(ii)
        for ii in range(len(topo2)):
            topo2[ii].append(ii)
        adj1 = sknetwork.data.from_adjacency_list(topo1, matrix_only=True, reindex=False).toarray()
        adj2 = sknetwork.data.from_adjacency_list(topo2, matrix_only=True, reindex=False).toarray()

        try:
            spec0, spec1 = np.abs(self.spectral.fit_transform(adj1)), np.abs(self.spectral.fit_transform(adj2))
        except:
            print('>>>>' + file_name, flush=True)
            spec0, spec1 = np.zeros((len(adj1), 64)), np.zeros((len(adj2), 64))
        # else:
        #     print('<<<<' + file_name, flush=True)

        # adj2 = adj2 + np.eye(len(adj2))

        if self.is_eval:
            return{
                'keypoints0': v2d1,
                'keypoints1': v2d2,
                'topo0': [topo1],
                'topo1': [topo2],
                # 'id0': id1,
                # 'id1': id2,
                'adj_mat0': adj1,
                'adj_mat1': adj2,
                'spec0': spec0,
                'spec1': spec1,
                'imaget': imgt,
                'image0': img1,
                'image1': img2,
                'motion0': motion0,
                'motion1': motion2,
                'visibility0': visible01,
                'visibility1': visible21,

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
        elif not self.is_train:
            return{
                'keypoints0': v2d1,
                'keypoints1': v2d2,
                # 'topo0': [topo1],
                # 'topo1': [topo2],
                # 'id0': id1,
                # 'id1': id2,
                'adj_mat0': adj1,
                'adj_mat1': adj2,
                'spec0': spec0,
                'spec1': spec1,
                'imaget': imgt,
                'image0': img1,
                'image1': img2,
                'motion0': motion0,
                'motion1': motion2,
                'visibility0': visible01,
                'visibility1': visible21,

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
                'adj_mat0': adj1,
                'adj_mat1': adj2,
                'spec0': spec0,
                'spec1': spec1,
                'imaget': imgt,
                'motion0': motion0,
                'motion1': motion2,
                'visibility0': visible01,
                'visibility1': visible21,

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
        self.label_list = v * self.label_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def fetch_dataloader(args, type='train',):
    lineart = MixamoLineArtMotionSequence(root=args.root, gap=args.gap, split=args.type, model=args.model, action=args.action, mode=args.mode if hasattr(args, 'mode') else 'train', use_vs=args.use_vs if hasattr(args, 'use_vs') else False)
    
    if args.mode == 'train':
        lineart = MixamoLineArtMotionSequence(root=args.root, gap=args.gap, split=args.type, model=args.model, action=args.action, mode=args.mode if hasattr(args, 'mode') else 'train')
    
    if args.mode == 'train':
        loader = data.DataLoader(lineart, batch_size=args.batch_size, 
            pin_memory=True, shuffle=True, num_workers=16, drop_last=True, worker_init_fn=worker_init_fn)
    else:
        loader = data.DataLoader(lineart, batch_size=args.batch_size, 
            pin_memory=True, shuffle=False, num_workers=8)
    return loader

