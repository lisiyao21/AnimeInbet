""" This script handling the training process. """
import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from datasets import fetch_dataloader
import random
from utils.log import Logger

from torch.optim import *
import warnings
from tqdm import tqdm
import itertools
import pdb
import numpy as np
import models
import datetime
import sys
import json
import cv2

from utils.visualize_vtx_corr import visualize
import matplotlib.cm as cm
# from models.utils import make_matching_seg_plot

warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import pdb

class VtxMat():
    def __init__(self, args):
        self.config = args
        torch.backends.cudnn.benchmark = True
        torch.multiprocessing.set_sharing_strategy('file_system')
        self._build()

    def train(self):
        
        opt = self.config
        print(opt)

        model = self.model

        if hasattr(self.config, 'init_weight'):
            checkpoint = torch.load(self.config.init_weight)
            model.load_state_dict(checkpoint['model'])

        optimizer = self.optimizer
        schedular = self.schedular
        mean_loss = []
        log = Logger(self.config, self.expdir)
        updates = 0
        
        # set seed
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        np.random.seed(opt.seed)

        # start training
        for epoch in range(1, opt.epoch+1):
            np.random.seed(opt.seed + epoch)
            train_loader = self.train_loader
            log.set_progress(epoch, len(train_loader))
            batch_loss = 0
            batch_acc = 0 
            batch_valid_acc = 0
            batch_iter = 0
            model.train()
            avg_time = 0
            avg_num = 0
            # torch.cuda.synchronize()
            
            for i, pred in enumerate(train_loader):
                # tstart = time.time()
                # print(pred['file_name'])
                data = model(pred)


                if not data['skip_train']:
                    loss = data['loss'] / opt.batch_size
                    batch_loss += loss.item()
                    batch_acc += data['accuracy'] 
                    batch_valid_acc += data['valid_accuracy'] 
                    loss.backward()
                    batch_iter += 1
                else:
                    print('Skip!')

                ## Accumulate gradient for batch training
                if ((i + 1) % opt.batch_size == 0) or (i + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_iter = 1 if batch_iter == 0 else batch_iter               
                    stats = {
                        'updates': updates,
                        'loss': batch_loss,
                        'accuracy': batch_acc / batch_iter,
                        'valid_accuracy': batch_valid_acc / batch_iter
                    }
                    log.update(stats)
                    updates += 1
                    batch_loss = 0
                    batch_acc = 0 
                    batch_valid_acc = 0
                    batch_iter = 0

            # torch.cuda.synchronize()

            # avg_num += 1
                # for name, params in model.named_parameters():
                #     print('-->name:, ', name, '-->grad mean', params.grad.mean())
            # print("All time is ", avg_time, "AVG time is ", avg_time * 1.0 /avg_num,  "number is ", avg_num, flush=True)

            # save checkpoint 
            if epoch % opt.save_per_epochs == 0 or epoch == 1:
                checkpoint = {
                    'model': model.state_dict(),
                    'config': opt,
                    'epoch': epoch
                }

                filename = os.path.join(self.ckptdir, f'epoch_{epoch}.pt')
                torch.save(checkpoint, filename)
                
            # validate
            if epoch % opt.test_freq == 0:

                if not os.path.exists(os.path.join(self.visdir, 'epoch' + str(epoch))):
                    os.mkdir(os.path.join(self.visdir, 'epoch' + str(epoch)))
                eval_output_dir = os.path.join(self.visdir, 'epoch' + str(epoch))    
                
                test_loader = self.test_loader

                with torch.no_grad():
                    # Visualize the matches.
                    mean_acc = []
                    mean_valid_acc = []
                    model.eval()
                    for i_eval, data in enumerate(tqdm(test_loader, desc='Predicting Vtx Corr...')):
                        pred = model(data)
                        # for k, v in data.items():
                        #     pred[k] = v[0]
                        #     pred = {**pred, **data}

                        mean_acc.append(pred['accuracy'])
                        mean_valid_acc.append(pred['valid_accuracy'])
                    log.log_eval({
                        'updates': opt.epoch,
                        'Accuracy': np.mean(mean_acc),
                        'Valid Accuracy': np.mean(mean_valid_acc),
                        })
                    print('Epoch [{}/{}]], Acc.: {:.4f}, Valid Acc.{:.4f}' 
                        .format(epoch, opt.epoch, np.mean(mean_acc), np.mean(mean_valid_acc)) )
                    sys.stdout.flush()
                        # make_matching_plot(
                        #     image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                        #     text, viz_path, stem, stem, True,
                        #     True, False, 'Matches')
        
            self.schedular.step()

            


    def eval(self):
        train_action = ['breakdance_1990', 'capoeira', 'chapa-giratoria', 'fist_fight', 'flying_knee', 'freehang_climb', 'running', 'shove', 'magic', 'tripping']
        test_action = ['great_sword_slash', 'hip_hop_dancing']

        train_model = ['ganfaul', 'girlscout', 'jolleen', 'kachujin', 'knight', 'maria_w_jj', 'michelle', 'peasant_girl', 'timmy', 'uriel_a_plotexia']
        test_model = ['police', 'warrok']

        log = Logger(self.config, self.expdir)
        with torch.no_grad():
            model = self.model.eval()
            config = self.config
            epoch_tested = self.config.testing.ckpt_epoch
            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
            # self.device = torch.device('cuda' if config.cuda else 'cpu')
            print("Evaluation...")
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model'])

            model.eval()

            if not os.path.exists(os.path.join(self.evaldir, 'epoch' + str(epoch_tested))):
                os.mkdir(os.path.join(self.evaldir, 'epoch' + str(epoch_tested)))
            if not os.path.exists(os.path.join(self.evaldir, 'epoch' + str(epoch_tested), 'jsons')):
                os.mkdir(os.path.join(self.evaldir, 'epoch' + str(epoch_tested), 'jsons'))
            eval_output_dir = os.path.join(self.evaldir, 'epoch' + str(epoch_tested))    
                
            test_loader = self.test_loader
            print(len(test_loader))
            mean_acc = []
            mean_valid_acc = []
            mean_invalid_acc = []

            # 144 data 
            # 10x10 is for training , 2x10 (unseen model) + 10x2 (unseen action) + 2x2 (unseen model unseen action) is for test
            # record the accuracy for each
            mean_model_acc = []
            mean_model_valid_acc = []
            mean_action_acc = []
            mean_action_valid_acc = []
            
            mean_none_acc = []
            mean_none_valid_acc = []

            mean_matched = []

            for i_eval, pred in enumerate(tqdm(test_loader, desc='Predicting Vtx Corr...')):
                data = model(pred)
                for k, v in pred.items():
                    pred[k] = v[0]
                    pred = {**pred, **data}
            
                mean_acc.append(pred['accuracy'])
                mean_valid_acc.append(pred['valid_accuracy'])
                this_pred = (pred['matches0'] != -1).float().cpu().data.numpy().astype(np.float32)
                mean_matched.append(np.mean( this_pred))

                unmarked = True
                for model_name in train_model:
                    if model_name in pred['file_name']:
                        mean_model_acc.append(pred['accuracy'])
                        mean_model_valid_acc.append(pred['valid_accuracy'])
                        unmarked = False
                        break

                for action_name in train_action:
                    if action_name in pred['file_name']:
                        mean_action_acc.append(pred['accuracy'])
                        mean_action_valid_acc.append(pred['valid_accuracy'])
                        unmarked = False
                        break
                
                if unmarked:
                    mean_none_acc.append(pred['accuracy'])
                    mean_action_valid_acc.append(pred['valid_accuracy'])

                if 'invalid_accuracy' in pred and pred['invalid_accuracy'] is not None:
                    mean_invalid_acc.append(pred['invalid_accuracy'])
                
                img_vis = visualize(pred)
                cv2.imwrite(os.path.join(eval_output_dir, pred['file_name'].replace('/', '_') + '.jpg'), img_vis)
                
            log.log_eval({
                'updates': self.config.testing.ckpt_epoch,
                'Accuracy': np.mean(mean_acc),
                'Accuracy (Matched)': np.mean(mean_valid_acc),
                'Unseen Action Accuracy': np.mean(mean_model_acc),
                'Unseen Action Accuracy (Matched)': np.mean(mean_model_valid_acc),
                'Unseen Model Accuracy': np.mean(mean_action_acc),
                'Unseen Model Accuracy (Matched)': np.mean(mean_action_valid_acc),
                'Unseen Both Accuracy': np.mean(mean_none_acc),
                'Unseen Both Valid Accuracy': np.mean(mean_none_valid_acc),
                'Matching Rate': np.mean(mean_matched)
                })
                # print ('Epoch [{}/{}]], Acc.: {:.4f}, Valid Acc.{:.4f}' 
                #     .format(epoch, opt.epoch, np.mean(mean_acc), np.mean(mean_valid_acc)) )
            sys.stdout.flush()

    def _build(self):
        config = self.config
        self.start_epoch = 0
        self._dir_setting()
        self._build_model()
        if not(hasattr(config, 'need_not_train_data') and config.need_not_train_data):
            self._build_train_loader()
        if not(hasattr(config, 'need_not_test_data') and config.need_not_train_data):      
            self._build_test_loader()
        self._build_optimizer()

    def _build_model(self):
        """ Define Model """
        config = self.config 
        if hasattr(config.model, 'name'):
            print(f'Experiment Using {config.model.name}')
            model_class = getattr(models, config.model.name)
            model = model_class(config.model)
        else:
            raise NotImplementedError("Wrong Model Selection")
        
        model = nn.DataParallel(model)
        self.model = model.cuda()

    def _build_train_loader(self):
        config = self.config
        self.train_loader = fetch_dataloader(config.data.train, type='train')

    def _build_test_loader(self):
        config = self.config
        self.test_loader = fetch_dataloader(config.data.test, type='test')

    def _build_optimizer(self):
        #model = nn.DataParallel(model).to(device)
        config = self.config.optimizer
        try:
            optim = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + config.type)

        self.optimizer = optim(itertools.chain(self.model.module.parameters(),
                                             ),
                                             **config.kwargs)
        self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)

    def _dir_setting(self):
        data = self.config.data
        self.expname = self.config.expname
        self.experiment_dir = os.path.join(".", "experiments")
        self.expdir = os.path.join(self.experiment_dir, self.expname)

        if not os.path.exists(self.expdir):
            os.mkdir(self.expdir)

        self.visdir = os.path.join(self.expdir, "vis")  # -- imgs, videos, jsons
        if not os.path.exists(self.visdir):
            os.mkdir(self.visdir)

        self.ckptdir = os.path.join(self.expdir, "ckpt")
        if not os.path.exists(self.ckptdir):
            os.mkdir(self.ckptdir)

        self.evaldir = os.path.join(self.expdir, "eval")
        if not os.path.exists(self.evaldir):
            os.mkdir(self.evaldir)

        

        # self.ckptdir = os.path.join(self.expdir, "ckpt")
        # if not os.path.exists(self.ckptdir):
        #     os.mkdir(self.ckptdir)



        




