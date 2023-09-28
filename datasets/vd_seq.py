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
import time
import json
import sknetwork
from sknetwork.embedding import Spectral

import scipy

def read_json(file_path):
    """
        input: json file path
        output: 2d vertex 
    """

    with open(file_path) as file:
        data = json.load(file)
        vertex2d = np.array(data['vertex location'])
        
        topology = data['connection']
        index = np.array(data['original index'])

        # index, vertex2d, topology = union_pixel(vertex2d, index, topology)
        # index, vertex2d, topology = union_pixel2d(vertex2d, index, topology)

    return vertex2d, topology, index


class VideoLinSeq(data.Dataset):
    def __init__(self, root, split='train'):
        """
            input:
                root: the root folder of the line art data
                split: split folder

            output:
                image of sources (0, 1) and output (0.5)
                topo0, topo1
                v2d0, v2d1


        """
        super(VideoLinSeq, self).__init__()

        self.image_list = []
        self.label_list = []

        label_root = osp.join(root, split, 'labels')
        image_root = osp.join(root, split, 'frames')

        self.spectral = Spectral(64,  normalized=False)

        for clip in os.listdir(image_root):
            
            label_list = sorted(glob(osp.join(label_root, clip, '*.json')))

            for i in range(len(label_list) - 1):
                self.label_list += [ [label_list[jj] for jj in range(i, i + 2)] ]
                self.image_list += [ [label_list[jj].replace('labels', 'frames').replace('.json', '.png') for jj in range(i, i + 2)] ]

        # print(clip)
        print('Len of Frame is ', len(self.image_list), flush=True)
        print('Len of Label is ', len(self.label_list), flush=True)

    def __getitem__(self, index):
        # prepare images
        index = index % len(self.image_list)
        file_name0 = self.label_list[index][0][:-5].split('/')[-1]
        file_name1 = self.label_list[index][-1][:-5].split('/')[-1]
        folder0 = self.label_list[index][0][:-4].split('/')[-2]
        folder1 = self.label_list[index][-1][:-4].split('/')[-2]


        imgt = [cv2.imread(self.image_list[index][ii]) for ii in range(0, len(self.image_list[index]))]

        labelt = []
        for ii in range(0, len(self.label_list[index])):
            v, t, id = read_json(self.label_list[index][ii])
            v[v > imgt[0].shape[0] - 1] = imgt[0].shape[0] - 1
            v[v < 0] = 0
            labelt.append({'keypoints': v.astype(int), 'topo': t, 'id': id})

        # make motion pseudo label

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

        v2d1 = torch.from_numpy(v2d1)
        v2d2 = torch.from_numpy(v2d2)

        mask0, mask1 = torch.ones(m).float(), torch.ones(n).float()

        v2d1[v2d1 > imgt[0].shape[0] - 1 ] = imgt[0].shape[0] - 1
        v2d1[v2d1 < 0] = 0
        v2d2[v2d2 > imgt[0].shape[1] - 1] = imgt[0].shape[1] - 1
        v2d2[v2d2 < 0] = 0

     
        id1 = np.arange(len(v2d1))
        id2 = np.arange(len(v2d2))

       
        for ii in range(len(topo1)):
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

        return{
            'keypoints0': v2d1,
            'keypoints1': v2d2,
            'topo0': [topo1],
            'topo1': [topo2],
            'adj_mat0': adj1,
            'adj_mat1': adj2,
            'spec0': spec0,
            'spec1': spec1,
            'image0': img1,
            'image1': img2,
            'ms': m,
            'ns': n,
            'mask0': mask0,
            'mask1': mask1,
            'gen_vid': True,
            'file_name0': file_name0,
            'file_name1': file_name1,
            'folder_name0': folder0,
            'folder_name1': folder1
        }


    def __rmul__(self, v):
        self.label_list = v * self.label_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def fetch_videoloader(args, type='train',):
    lineart = VideoLinSeq(root=args.root, split=args.type, )
    
    loader = data.DataLoader(lineart, batch_size=args.batch_size, 
            pin_memory=True, shuffle=False, num_workers=8)
    return loader

