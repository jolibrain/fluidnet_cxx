#!/usr/bin/env python3
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import glob
from shutil import copyfile

import lib
import lib.fluid as fluid
from config import defaultConf # Dictionnary with configuration and model params.

#********************************** Define Config ******************************

#TODO: allow to overwrite params from the command line by parsing.

conf = defaultConf.copy()
conf['modelDirname'] = conf['modelDir'] + conf['modelFilename']
resume = False

#*********************************** Select the GPU ****************************

print('Active CUDA Device: GPU', torch.cuda.current_device())

te = lib.FluidNetDataset(conf, 'te', save_dt=4, resume=resume) # Test instance of custom Dataset

conf, mconf = te.createConfDict()

test_loader = torch.utils.data.DataLoader(te, batch_size=1, \
        num_workers=0, shuffle=False, pin_memory=True)

print('==> loading checkpoint')
mpath = glob.os.path.join(conf['modelDir'],conf['modelFilename'] + '_lastEpoch_best.pth')
assert glob.os.path.isfile(mpath), mpath  + ' does not exits!'
state = torch.load(mpath)

print('Data loading: done')

#********************************** Create the model ***************************

print('')
print('----- Model ------')

# Create model and print layers and params
net = lib.FluidNet(mconf, dropout=False)
if torch.cuda.is_available():
    net = net.cuda()
#lib.summary(net, (3,1,128,128))

net.load_state_dict(state['state_dict'])


#********************************* Run the model and print ****************************
from itertools import count

batch_print = [100, 1000, 4000, 5000, 8000, 10000, 15000, 20000]

def val():
    net.eval()
    total_val_loss = 0
    for batch_idx, (data, target) in zip(count(step=1), test_loader):
        with torch.no_grad():
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            if batch_idx in batch_print:
                output = net(data)
                target_p = target[:,0].unsqueeze(1)
                loss_size = torch.sum( (output - target_p)**2 )
                err = (output - target_p)**2

                max_val = torch.max(target_p).cpu().data.numpy()
                min_val = torch.min(target_p).cpu().data.numpy()

                max_err = torch.max(err).cpu().data.numpy()

                mask = data[:,3].eq(2).unsqueeze(1)
                target.masked_fill_(mask, 100)
                output.masked_fill_(mask, 100)
                err.masked_fill_(mask, 100)
                y0 =torch.squeeze(target_p.cpu()).data.numpy()
                y1 =torch.squeeze(output.cpu()).data.numpy()
                y2 =torch.squeeze(err.cpu()).data.numpy()

                my_cmap = cm.jet
                my_cmap.set_over('gray')
                my_cmap.set_under('gray')

                fig,axes = plt.subplots(nrows = 1, ncols =3 )
                fig.suptitle('FluidNet output')
                im = axes[0].imshow(y0,cmap=my_cmap, origin='lower', interpolation='none', \
                        clim=[min_val,max_val])
                axes[0].set_title('P_target')
                axes[0].axis('off')
                im = axes[1].imshow(y1,cmap=my_cmap, origin='lower', interpolation='none', \
                        clim=[min_val,max_val])
                axes[1].set_title('P_predicted')
                axes[1].axis('off')
                imErr = axes[2].imshow(y2,cmap=my_cmap, origin='lower', interpolation='none', \
                        clim=[0,max_err])
                axes[2].set_title('error')
                axes[2].axis('off')
                cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.02])
                fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
                plt.show(block=True)

                total_val_loss += loss_size.item()

    total_val_loss /= len(test_loader.dataset)
    print('Validation set: Avg loss: {:.6f}\n'.format(total_val_loss))
    return total_val_loss

#********************************* Run epochs ****************************************

state = {'bestPerf': float('Inf')}
n_epochs = conf['maxEpochs']

print('')
print('==> Beginning simulation')
# Train on training set and test on validation set
val_loss = val()

