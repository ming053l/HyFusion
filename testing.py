import os
import glob
import time
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import torch.nn as nn
import json
from sklearn.metrics import log_loss
import pdb
import random as rn
# from models.dcsn3d import DCSNMain as DCSN
from models.SwinFusion import SwinIR_DCSN

from utils import *
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from dataset import *
from scipy.io import savemat, loadmat
from math import acos, degrees
from tensorboardX import SummaryWriter 
from trainOps import *
from scipy.io import savemat
import models
import tqdm
import time
import argparse

# torch.backends.cudnn.benchmark=True
# Hyperparameters
batch_size = 1
device = 'cuda'
MAX_EP = 16000
BANDS = 172
SIGMA = 0.0    ## Noise free -> SIGMA = 0.0

prefix='DCSN_joint2'
DEBUG=False


def parse_args():
    parser = argparse.ArgumentParser(description='Train Convex-Optimization-Aware SR net')
    
    parser.add_argument('--SEED', type=int, default=1029)
    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument('--resume_ind', type=int, default=999)
    parser.add_argument('--snr', type=int, default=0)
    
    
    parser.add_argument('--workers', type=int, default=4)

    ## Data generator configuration
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--bands', type=int, default=172)
    parser.add_argument('--msi_bands', type=int, default=4)
    parser.add_argument('--mis_pix', type=int, default=0)
    parser.add_argument('--mixed_align_opt', type=int, default=0)
    
    # Network architecture configuration
    parser.add_argument("--network_mode", type=int, default=2, help="Training network mode: 0) Single mode, 1) LRHSI+HRMSI, 2) COCNN (LRHSI+HRMSI+CO), Default: 2")     
    parser.add_argument('--num_base_chs', type=int, default=160, help='The number of the channels of the base feature')
    parser.add_argument('--num_blocks', type=int, default=6, help='The number of the repeated blocks in backbone')
    parser.add_argument('--num_agg_feat', type=int, default=40, help='the additive feature maps in the block')
    parser.add_argument('--groups', type=int, default=4, help="light version the group value can be >1, groups=1 for full COCNN version, groups=4 is COCNN-Light for 4 HRMSI version")
    
    # Others
    parser.add_argument("--root", type=str, default="../Data", help='data root folder')   
    parser.add_argument("--val_file", type=str, default="../test.txt")   
    parser.add_argument("--prefix", type=str, default="COCNN160-light")  
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:device_id or cpu")  
    parser.add_argument("--DEBUG", type=bool, default=False)  
    parser.add_argument("--gpus", type=int, default=1)  
    
    
    args = parser.parse_args()

    return args

def testing(args):
    ## Reading files #
    
    ## Load test files from specified text with SOURCE/TARGET name (you can replace it with other path u want)
    valfn = loadTxt('../test.txt')
    rn.shuffle(valfn)
    index = np.array(range(len(valfn)))
    rn.shuffle(index)
    print(f'#Testing samples is {len(valfn)}')

    if args.network_mode==2:
        dataset = dataset_joint2
        print('Use triplet dataset')
    elif args.network_mode==1:
        dataset = dataset_joint
        print('Use pairwise (LRHSI+HRMSI) dataset')
    elif args.network_mode==0:
        dataset = dataset_h5
        print('Use CO dataset')

    val_loader = torch.utils.data.DataLoader(dataset(valfn, args, mode='val'), batch_size=args.batch_size, shuffle=False, drop_last = True, pin_memory=True, num_workers=args.workers)

    model = DCSN(args).to(device) ## cr =[1, 5, 10, 15, 20] compression ratio
    
    if args.resume_ind>0:
        args.resume_ckpt = os.path.join('checkpoint', args.prefix, 'best.pth')
        if not os.path.isfile(args.resume_ckpt):
            print(f"checkpoint is not found at {args.resume_ckpt}")
            raise 
        state_dict = torch.load(args.resume_ckpt)  
        model.load_state_dict(state_dict)
        print(f'Loading the pretrained model from {args.resume_ckpt}')
        ## finetune
        
    model.eval().cuda()

    with torch.no_grad():
        
        for X in val_loader:
            if args.network_mode==2:
                x,x2,x3,vy,vfn, maxv, minv = X
                x3 = x3.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
            elif args.network_mode==1:
                x,x2,vy,vfn, maxv, minv= X
                x2 = x2.to(device, non_blocking=True)
            elif args.network_mode==0:
                x,vy,vfn, maxv, minv = X
            x = x.to(device, non_blocking=True)
            
            if args.network_mode==2:
                val_dec = model(x, LRHSI=x2, HRMSI=x3, mode=1)
            elif args.network_mode==1:
                val_dec = model(None, LRHSI=x2, HRMSI=x, mode=1)
            elif args.network_mode==0:
                val_dec = model(x, LRHSI=None, HRMSI=None, mode=1)
            
            break
            
            
        rmses, sams, fnames, psnrs, ergas = [], [], [], [], []
        ep = 0.0
        for batch_idx, (X) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
            
            if args.network_mode==2:
                x,x2,x3,vy,vfn, maxv, minv = X
                x3 = x3.cuda()
                x2 = x2.cuda()
            elif args.network_mode==1:
                x,x2,vy,vfn, maxv, minv= X
                x2 = x2.cuda()
            elif args.network_mode==0:
                x,vy,vfn, maxv, minv = X
            x = x.cuda()
            
#             torch.cuda.synchronize()
            start_time = time.time()
            if args.network_mode==2:
                val_dec = model(x, LRHSI=x2, HRMSI=x3, mode=1)
            elif args.network_mode==1:
                val_dec = model(None, LRHSI=x2, HRMSI=x, mode=1)
            elif args.network_mode==0:
                val_dec = model(x, LRHSI=None, HRMSI=None, mode=1)
#             torch.cuda.synchronize()
            ep = ep + (time.time()-float(start_time))
            
            val_dec = val_dec.cpu().numpy()
            vy = vy.cpu().numpy()
            

            for predimg, gtimg,f, v1, v2 in zip(val_dec, vy, vfn, maxv, minv):
                predimg = (predimg/2+0.5) 
                gtimg = (gtimg/2+0.5) 

                sams.append(sam2(predimg, gtimg))
                psnrs.append(psnr(predimg, gtimg))
                ergas.append(ERGAS(predimg, gtimg))
                savemat('recimg/'+os.path.basename(f)+'.mat', {'pred':np.transpose(predimg,(1,2,0))})
#                 predimg = predimg * (v1-v2) + v2
#                 gtimg = gtimg * (v1-v2) + v2
                rmses.append(rmse(predimg, gtimg, maxv=v1, minv=v2))
                


        
        ep = ep / len(sams)
        print('val-PSNR: %.3f, val-SAM: %.3f, val-ERGAS: %.3f, val-rmse: %.3f, AVG-Time: %f ms on %d misaligned pix' %
              (np.mean(psnrs), np.mean(sams), np.mean(ergas), np.mean(rmses), ep*1000.0, args.mis_pix))


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.SEED)
    rn.seed(args.SEED)
    np.random.seed(args.SEED)

    testing(args)
