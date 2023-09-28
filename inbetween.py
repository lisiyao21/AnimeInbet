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
from datasets import fetch_videoloader
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

from utils.visualize_inbetween3 import visualize
# from utils.visualize_inbetween import visualize
from utils.visualize_video import visvid as visgen
import matplotlib.cm as cm
# from models.utils import make_matching_seg_plot

warnings.filterwarnings('ignore')

# a, b, c, d = check_data_distribution('/mnt/lustre/lisiyao1/dance/dance2/DanceRevolution/data/aistpp_train')

import matplotlib.pyplot as plt
import pdb

class DraftRefine():
    def __init__(self, args):
        self.config = args
        torch.backends.cudnn.benchmark = True
        torch.multiprocessing.set_sharing_strategy('file_system')
        self._build()

    def train(self):
        
        opt = self.config
        print(opt)

        # store viz results
        # eval_output_dir = Path(self.expdir)
        # eval_output_dir.mkdir(exist_ok=True, parents=True)

        # print('Will write visualization images to',
        #     'directory \"{}\"'.format(eval_output_dir))

        # load training data
        
        model = self.model

        checkpoint = torch.load(self.config.corr_weights)
        dict = {k.replace('module.', ''): checkpoint['model'][k] for k in checkpoint['model']}
        model.module.corr.load_state_dict(dict)

        if hasattr(self.config, 'init_weight'):
            checkpoint = torch.load(self.config.init_weight)
            model.load_state_dict(checkpoint['model'])

        # if torch.cuda.is_available():
        #     model.cuda() # make sure it trains on GPU
        # else:
        #     print("### CUDA not available ###")
            # return
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
        # print(opt.seed)
        # start training

        for epoch in range(1, opt.epoch+1):
            np.random.seed(opt.seed + epoch)
            train_loader = self.train_loader
            log.set_progress(epoch, len(train_loader))
            batch_loss = 0
            batch_epe = 0 
            batch_acc = 0
            batch_iter = 0
            model.train()
            avg_time = 0
            avg_num = 0
            # torch.cuda.synchronize()
            
            for i, data in enumerate(train_loader):
                pred = model(data)
                if True:
                    loss = pred['loss'].mean() 
                    # print(loss.item(), opt.batch_size)
                    batch_loss += loss.item() / opt.batch_size
                    batch_acc += pred['Visibility Acc'].mean().item() / opt.batch_size
                    batch_epe += pred['EPE'].mean().item() / opt.batch_size 
                    loss.backward()
                    batch_iter += 1
                else:
                    print('Skip!')



                if ((i + 1) % opt.batch_size == 0) or (i + 1 == len(train_loader)):
                    optimizer.step()

                    optimizer.zero_grad()
                    batch_iter = 1 if batch_iter == 0 else batch_iter               
                    stats = {
                        'updates': updates,
                        'loss': batch_loss,
                        'accuracy': batch_acc,
                        'EPE': batch_epe
                    }
                    log.update(stats)
                    updates += 1
                    batch_loss = 0
                    batch_acc = 0 
                    batch_epe = 0
                    batch_iter = 0
                # tend = time.time()
                # avg_time = (tend - tstart)
                # print('Time is ', avg_time)

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
                    mean_epe = []
                    model.eval()
                    for i_eval, data in enumerate(tqdm(test_loader, desc='Refining motion and visibility...')):
                        pred = model(data)
                        # for k, v in data.items():
                        #     pred[k] = v[0]
                        #     pred = {**pred, **data}

                        mean_acc.append(pred['Visibility Acc'].mean().item())
                        mean_epe.append(pred['EPE'].mean().item())
                    log.log_eval({
                        'updates': opt.epoch,
                        'Visibility Accuracy': np.mean(mean_acc),
                        'EPE': np.mean(mean_epe),
                        })
                    print('Epoch [{}/{}]], Vis Acc.: {:.4f}, EPE: {:.4f}' 
                        .format(epoch, opt.epoch, np.mean(mean_acc), np.mean(mean_epe)) )
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

        config = self.config
        if not os.path.exists(config.imwrite_dir):
            os.mkdir(config.imwrite_dir)
            
        log = Logger(self.config, self.expdir)
        with torch.no_grad():
            model = self.model.eval()
            config = self.config
            epoch_tested = self.config.testing.ckpt_epoch
            if epoch_tested == 0 or epoch_tested == '0':
                checkpoint = torch.load(self.config.corr_weights)
                dict = {k.replace('module.', ''): checkpoint['model'][k] for k in checkpoint['model']}
                model.module.corr.load_state_dict(dict)
            else:
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

            # 144 data 10x10 is for training , 2x10 (unseen model) + 10x2 (unseen action) + 2x2 (unseen model unseen action) is for test
            # record the accuracy for 
            mean_model_acc = []
            mean_model_epe = []
            mean_action_acc = []
            mean_action_epe = []
            
            mean_none_acc = []
            mean_none_epe = []

            mean_acc = []
            mean_epe = []

            mean_cd = []
            model.eval()
            # for i_eval, data in enumerate(tqdm(test_loader, desc='Refining motion and visibility...')):
            #     pred = model(data)
            #     # for k, v in data.items():
            #     #     pred[k] = v[0]
            #     #     pred = {**pred, **data}

            #     mean_acc.append(pred['Visibility Acc'].mean().item())
            #     mean_epe.append(pred['EPE'].mean().item())
            # log.log_eval({
            #     'updates': opt.epoch,
            #     'Visibility Accuracy': np.mean(mean_acc),
            #     'EPE': np.mean(mean_epe),
            #     })

            for i_eval, data in enumerate(tqdm(test_loader, desc='Predicting Vtx Corr...')):
                # if i_eval == 34:
                #     continue
                
                pred = model(data)
                for k, v in pred.items():
                    # print(k, flush=True)
                    pred[k] = v
                    pred = {**pred, **data}
            
                mean_acc.append(pred['Visibility Acc'].mean().item())
                mean_epe.append(pred['EPE'].mean().item())

                unmarked = True
                for model_name in train_model:
                    if model_name in pred['file_name']:
                        mean_model_acc.append(pred['Visibility Acc'])
                        mean_model_epe.append(pred['EPE'])
                        unmarked = False
                        break

                for action_name in train_action:
                    if action_name in pred['file_name']:
                        mean_action_acc.append(pred['Visibility Acc'])
                        mean_action_epe.append(pred['EPE'])
                        unmarked = False
                        break
                
                if unmarked:
                    mean_none_acc.append(pred['Visibility Acc'])
                    mean_action_epe.append(pred['EPE'])

                # if 'invalid_accuracy' in pred and pred['invalid_accuracy'] is not None:
                #     mean_invalid_acc.append(pred['invalid_accuracy'])
                
                img_vis = visualize(pred)
                # mean_cd.append(cd.item())
                file_name = pred['file_name'][0].split('/')
                cv2.imwrite(os.path.join(config.imwrite_dir, (file_name[-2] + '_' + file_name[-1]) + 'png'), img_vis)

                # cv2.imwrite(os.path.join(eval_output_dir, pred['file_name'][0].replace('/', '_') + '.jpg'), img_vis)
                
            log.log_eval({
                'updates': self.config.testing.ckpt_epoch,
                # 'mean CD': np.mean(mean_cd),
                # 'Visibility Accuracy': np.mean(mean_acc),
                # 'EPE': np.mean(mean_epe),
                # 'Unseen Action Accuracy': np.mean(mean_model_acc),
                # 'Unseen Action EPE': np.mean(mean_model_epe),
                # 'Unseen Model Accuracy': np.mean(mean_action_acc),
                # 'Unseen Model EPE': np.mean(mean_action_epe),
                # 'Unseen Both Accuracy': np.mean(mean_none_acc),
                # 'Unseen Both Valid Accuracy': np.mean(mean_none_epe)
                })
                # print ('Epoch [{}/{}]], Acc.: {:.4f}, Valid Acc.{:.4f}' 
                #     .format(epoch, opt.epoch, np.mean(mean_acc), np.mean(mean_valid_acc)) )
            sys.stdout.flush()


    def gen(self):
        log = Logger(self.config, self.viddir)
        with torch.no_grad():
            model = self.model.eval()
            config = self.config
            epoch_tested = self.config.testing.ckpt_epoch
            if epoch_tested == 0 or epoch_tested == '0':
                checkpoint = torch.load(self.config.corr_weights)
                dict = {k.replace('module.', ''): checkpoint['model'][k] for k in checkpoint['model']}
                model.module.corr.load_state_dict(dict)
            else:
                ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
                # self.device = torch.device('cuda' if config.cuda else 'cpu')
                print("Evaluation...")
                checkpoint = torch.load(ckpt_path)
                model.load_state_dict(checkpoint['model'])
            model.eval()

            if not os.path.exists(os.path.join(self.viddir, 'epoch' + str(epoch_tested))):
                os.mkdir(os.path.join(self.viddir, 'epoch' + str(epoch_tested)))
            if not os.path.exists(os.path.join(self.viddir, 'epoch' + str(epoch_tested), 'frames')):
                os.mkdir(os.path.join(self.viddir, 'epoch' + str(epoch_tested), 'frames'))
            if not os.path.exists(os.path.join(self.viddir, 'epoch' + str(epoch_tested), 'videos')):
                os.mkdir(os.path.join(self.viddir, 'epoch' + str(epoch_tested), 'videos'))

            gen_frame_dir = os.path.join(self.viddir, 'epoch' + str(epoch_tested), 'frames')  
            gen_video_dir = os.path.join(self.viddir, 'epoch' + str(epoch_tested), 'videos')    
                
            vid_loader = self.vid_loader
            print(len(vid_loader))
            mean_acc = []
            mean_valid_acc = []
            mean_invalid_acc = []

            model.eval()

            for i_eval, data in enumerate(tqdm(vid_loader, desc='Gen Video...')):
                
                pred = model(data)
                for k, v in pred.items():
                    pred[k] = v
                    pred = {**pred, **data}
            

                img_vis = visgen(pred, config.inter_frames)

                if not os.path.exists(os.path.join(gen_frame_dir, pred['folder_name0'][0])):
                    os.mkdir(os.path.join(gen_frame_dir, pred['folder_name0'][0]))
                
                cv2.imwrite(os.path.join(gen_frame_dir, pred['folder_name0'][0], pred['file_name0'][0] + '_000.jpg'),img_vis[0])
                for tt in range(config.inter_frames):
                    cv2.imwrite(os.path.join(gen_frame_dir, pred['folder_name0'][0], pred['file_name0'][0] + '_' + '{:03d}'.format(tt + 1) + '.jpg'), img_vis[tt + 1])
                cv2.imwrite(os.path.join(gen_frame_dir, pred['folder_name0'][0], pred['file_name1'][0] + '_000.jpg'),img_vis[-1])
            
            for ff in os.listdir(gen_frame_dir):
                frame_dir = os.path.join(gen_frame_dir, ff)
                video_file = os.path.join(gen_video_dir, f"{ff}.mp4")
                cmd = f"ffmpeg -r {config.fps} -pattern_type glob -i '{frame_dir}/*.jpg' -vb 20M -vcodec mpeg4 -y '{video_file}'"
                
                print(cmd, flush=True)
                os.system(cmd)
                

            log.log_eval({
                'updates': self.config.testing.ckpt_epoch,
                })
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
        if hasattr(config, 'gen_video') and config.gen_video:
            self._build_video_loader()
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
    def _build_video_loader(self):
        config = self.config
        self.vid_loader = fetch_videoloader(config.video)

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
        self.expname = self.config.expname
        # self.experiment_dir = os.path.join("/mnt/cache/syli/inbetween", "experiments")

        self.experiment_dir = 'experiments'
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

        self.viddir = os.path.join(self.expdir, "video")
        if not os.path.exists(self.viddir):
            os.mkdir(self.viddir)

        

        # self.ckptdir = os.path.join(self.expdir, "ckpt")
        # if not os.path.exists(self.ckptdir):
        #     os.mkdir(self.ckptdir)



        




