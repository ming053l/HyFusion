import torch
from torch import nn
import numpy as np, math
from torch.nn import functional as F
from torch.autograd import Variable
import functools
from module_util import *
from pthflops import count_ops
from ptflops import get_model_complexity_info

import pdb

##===============MFB + MFA ===============================================
class MultiScaleFeatFusionBlock(nn.Module):   ## MFB
    def __init__(self, nf=64, gc=32, bias=True, groups=4):
        super(MultiScaleFeatFusionBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias, groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class MultiScaleFeatAggregation(nn.Module):   ## MFA


    def __init__(self, nf, gc=32, groups=4):
        super(MultiScaleFeatAggregation, self).__init__()
        self.MFB1 = MultiScaleFeatFusionBlock(nf, gc, groups=groups)
        self.MFB2 = MultiScaleFeatFusionBlock(nf, gc, groups=groups)
        self.MFB3 = MultiScaleFeatFusionBlock(nf, gc, groups=groups)

    def forward(self, x):
        out = self.MFB1(x)
        out = self.MFB2(out)
        out = self.MFB3(out)
        return out * 0.2 + x


class DCSNDecSingle(nn.Module):
    def make_layer(block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(DCSNDecSingle, self).__init__()
        MFA_block_f = functools.partial(MultiScaleFeatAggregation, nf=nf, gc=gc, groups=1)#in_nc=172, out_nc=172, nf=172, nb=6, gc=172//4
        
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        
        
        self.MFA_trunk = make_layer(MFA_block_f, nb)
        
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv_first(x))
        x = x + self.trunk_conv(self.MFA_trunk(x))
       
        x = self.lrelu(self.HRconv(x))
        x = self.conv_last(x)
        
      
        return x    

class COGuidedDCSN(nn.Module):
    
    def make_layer(block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def __init__(self, in_nc,out_nc, nf, nb, in_msi=4, gc=32, useCO=True, groups=4):
        super(COGuidedDCSN, self).__init__()
        
        self.useCO = useCO
        # for LRHSI
        
        in_nc_group = groups
        if in_nc % groups != 0:
            in_nc_group = 1
            
        self.hsiconv1 = nn.Conv2d(in_nc, nf*2, 3, 1, 1, bias=True, groups=in_nc_group)
        block1 = functools.partial(MultiScaleFeatAggregation, nf=nf*2, gc=gc//2, groups=groups)
        self.hsifeat = make_layer(block1, 2)
        self.hsiconvlast = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True, groups=groups)
        self.up = torch.nn.Upsample(scale_factor=2)
        
        # for HRMSI
        self.msiconv1 = nn.Conv2d(in_msi, nf//2, 3, 1, 1, bias=True)
        block2 = functools.partial(MultiScaleFeatAggregation, nf=nf//2, gc=gc*2, groups=1)
        self.msifeat = make_layer(block2, 2)
        self.msiconvlast = nn.Conv2d(nf//2, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        # For CO input for reference
        
        if self.useCO:
            print('Build CO-guided blocks!')
            block4 = functools.partial(MultiScaleFeatAggregation, nf=nf, gc=gc, groups=groups)
            self.coconv1 = nn.Conv2d(out_nc, nf, 3, 1, 1, bias=True)
            self.cofeat = make_layer(block4, 2)
            
        
        # For fusion
        
        self.conv_fuse = nn.Sequential(nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True, groups=in_nc_group), 
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        self.codconv1 = nn.Conv2d(out_nc, nf, 3, 1, 1, bias=True, groups=in_nc_group)
        block3 = functools.partial(MultiScaleFeatAggregation, nf=nf, gc=gc, groups=groups)
        self.codfeat = make_layer(block3, nb)
#         self.codconvlast = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
       
        self.last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=False)
        
    def forward(self, co, lrhsi, hrmsi):
        
        lrhsi = self.lrelu(self.hsiconv1(lrhsi))
        lrhsi = self.up(lrhsi)
        lrhsi += self.hsifeat(lrhsi.clone())
        lrhsi = self.up(lrhsi)
        lrhsi = self.lrelu(lrhsi)
        lrhsi = self.hsiconvlast(lrhsi)
        
        hrmsi = self.lrelu(self.msiconv1(hrmsi))
        hrmsi += self.msifeat(hrmsi.clone())
        hrmsi = self.lrelu(hrmsi)
        hrmsi = self.msiconvlast(hrmsi)
        
        codraft = self.conv_fuse(hrmsi + lrhsi) 
        
        codraft = self.lrelu(self.codconv1(codraft) )
        codraft += self.codfeat(codraft.clone())
        codraft = self.lrelu(codraft)

        if self.useCO:
            co = self.lrelu(self.coconv1(co))
            co += self.cofeat(co.clone())
            co = self.lrelu(co)

            co = self.last(co + codraft)
        else:
            co = self.last(codraft)
        
        return co
    

##=========DCSN full module==========
class DCSN(nn.Module): 
    def __init__(self):
        super(DCSN, self).__init__()
        self.snr = 35
        self.joint = 2
        
        if self.joint==1:
            self.decoder = COGuidedDCSN(in_nc=172, out_nc=172, nf=172, nb=6, gc=172//4, in_msi=4, useCO=False, groups=1)
            print('Use the pairwise-COCNN!') # 176, 6, 48
        elif self.joint==2:
            self.decoder = COGuidedDCSN(in_nc=172, out_nc=172, nf=172, nb=6, gc=172//4, in_msi=4,groups=1)
            print('Use the COCNN!')
        else:
            self.decoder = DCSNDecSingle(in_nc=172, out_nc=172, nf=172, nb=6, gc=172//4)
            print('Use the single-input COCNN!') # 172,12,48
       
    def awgn(self, x):
        snr = 10**(self.snr/10.0)
        xpower = torch.sum(x**2)/x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(172,256,256) * npower


    def forward(self, x,data=None, LRHSI=None, HRMSI=None, mode=0): ### Mode=0, default, mode=1: encode only, mode=2: decoded only
        # if self.snr>0 and mode==0 and (self.joint==2 or self.joint==0):
        x[0] = self.awgn(x[0])
        # elif self.snr>0 and mode==0 and self.joint==1:
        #     LRHSI = self.awgn(LRHSI)

        # if self.joint==0:
            # return self.decoder(x[0])
        # elif self.joint==1: #co, lrhsi, hrmsi (No co, input lrhsi and hrmsi)
        #     return self.decoder(None, LRHSI, HRMSI)
        # else:
        return self.decoder(x[0], x[1], x[2])
    
    
##==========================================




class Discriminator(nn.Module):

    
    def __init__(self, in_nc, gc=32):
        super(Discriminator, self).__init__()
       
        self.conv1 = nn.Conv2d(in_nc, gc, 3, 2, 1, bias=True)
        self.conv2 = nn.Conv2d(gc, gc, 3, 2, 1, bias=True)
        self.conv3 = nn.Conv2d(gc, gc, 3, 2, 1, bias=True)
        self.conv4 = nn.Conv2d(gc, gc, 3, 2, 1, bias=True)
        self.conv5 = nn.Conv2d(gc, gc, 3, 2, 1, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Linear(gc, 1)
     
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.avgpool(x).view(x.shape[0], -1)
        x = self.cls(x)
        return self.sigmoid(x)
def prepare_input(resolution):
    data = torch.FloatTensor(1, 172,256,256)
    lr_hsi = torch.FloatTensor(1,172,64,64)
    hr_msi = torch.FloatTensor(1,4,256,256)
    return dict(x = [data, lr_hsi,hr_msi])
        

##==========================================

if __name__ == '__main__':
    
        # Create a network and a corresponding input
    device = 'cuda:0'
    model = DCSN()
    flops, params = get_model_complexity_info(model, input_res=(1, 224, 224), 
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)

