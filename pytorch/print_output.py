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
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np
import glob
from shutil import copyfile
import importlib.util

import lib
import lib.fluid as fluid
from config import defaultConf # Dictionnary with configuration and model params.

# Use: python3 print_output.py <folder_with_model> <modelName>
# e.g: python3 print_output.py data/model_test convModel
#
# Utility to print data, output or target fields, as well as errors.
# It loads the state and runs one epoch with a batch size = 1. It can be used while
# training, to have some visual help.

#**************************** Load command line arguments *********************
assert (len(sys.argv) == 3), 'Usage: python3 print_output.py <modelDir> <modelName>'
assert (glob.os.path.exists(sys.argv[1])), 'Directory ' + str(sys.argv[1]) + ' does not exists'

#********************************** Define Config ******************************

#TODO: allow to overwrite params from the command line by parsing.

conf = defaultConf.copy()
conf['modelDir'] = sys.argv[1]
print(sys.argv[1])
conf['modelDirname'] = conf['modelDir'] + '/' + conf['modelFilename']
resume = False

#*********************************** Select the GPU ****************************
print('Active CUDA Device: GPU', torch.cuda.current_device())

path = conf['modelDir']
path_list = path.split(glob.os.sep)
saved_model_name = glob.os.path.join(*path_list[:-1], path_list[-2] + '_saved.py')
temp_model = glob.os.path.join('lib', path_list[-2] + '_saved_print.py')
copyfile(saved_model_name, temp_model)

assert glob.os.path.isfile(temp_model), temp_model  + ' does not exits!'
spec = importlib.util.spec_from_file_location('model_saved', temp_model)
model_saved = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_saved)

try:
    te = lib.FluidNetDataset(conf, 'te', save_dt=4, resume=resume) # Test instance of custom Dataset

    conf, mconf = te.createConfDict()

    print('==> overwriting conf and file_mconf')
    cpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_conf.pth')
    mcpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_mconf.pth')
    assert glob.os.path.isfile(mcpath), cpath  + ' does not exits!'
    assert glob.os.path.isfile(mcpath), mcpath  + ' does not exits!'
    conf = torch.load(cpath)
    mconf = torch.load(mcpath)

    test_loader = torch.utils.data.DataLoader(te, batch_size=1, \
            num_workers=0, shuffle=False, pin_memory=True)

    print('==> loading checkpoint')
    mpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_lastEpoch_best.pth')
    assert glob.os.path.isfile(mpath), mpath  + ' does not exits!'
    state = torch.load(mpath)

    print('Data loading: done')

    #********************************** Create the model ***************************

    print('')
    print('----- Model ------')

    # Create model and print layers and params

    net = model_saved.FluidNet(mconf, dropout=False)
    if torch.cuda.is_available():
        net = net.cuda()
    #lib.summary(net, (3,1,128,128))

    net.load_state_dict(state['state_dict'])

    #********************************* Plot functions *************************************

    def plotField(output, target, mconf, flags):
        div = 0
        exit = True
        p_out = output[0]
        p_tar = target[:,0].unsqueeze(1)
        U_norm_out = torch.zeros_like(p_out)
        U_norm_tar = torch.zeros_like(p_tar)

        torch.norm(output[1], p=2, dim=1, keepdim=True, out=U_norm_out)
        torch.norm(target[:,1:3], p=2, dim=1, keepdim=True, out=U_norm_tar)

        div = output[2]

        err_p = (p_out - p_tar)**2
        err_U = (U_norm_out - U_norm_tar)**2
        err_div = (div)**2

        max_val_p = np.maximum(torch.max(p_tar).cpu().data.numpy(), \
                             torch.max(p_out).cpu().data.numpy() )
        min_val_p = np.minimum(torch.min(p_tar).cpu().data.numpy(), \
                             torch.min(p_out).cpu().data.numpy())
        max_val_U = np.maximum(torch.max(U_norm_out).cpu().data.numpy(), \
                             torch.max(U_norm_tar).cpu().data.numpy() )
        min_val_U = np.minimum(torch.min(U_norm_out).cpu().data.numpy(), \
                             torch.min(U_norm_tar).cpu().data.numpy())
        max_err_p = torch.max(err_p).cpu().data.numpy()
        max_err_U = torch.max(err_U).cpu().data.numpy()

        max_div = torch.max(div).cpu().data.numpy()
        min_div = torch.min(div).cpu().data.numpy()

        mask = flags.eq(2)
        p_tar.masked_fill_(mask, 100)
        p_out.masked_fill_(mask, 100)
        U_norm_tar.masked_fill_(mask, 100)
        U_norm_out.masked_fill_(mask, 100)
        div.masked_fill_(mask, 100)

        err_p.masked_fill_(mask, 100)
        err_U.masked_fill_(mask, 100)
        err_div.masked_fill_(mask, 100)

        p_tar_np =torch.squeeze(p_tar.cpu()).data.numpy()
        p_out_np =torch.squeeze(p_out.cpu()).data.numpy()
        U_tar_np =torch.squeeze(U_norm_tar.cpu()).data.numpy()
        U_out_np =torch.squeeze(U_norm_out.cpu()).data.numpy()
        div_np =torch.squeeze(div).cpu().data.numpy()
        err_p_np =torch.squeeze(err_p.cpu()).data.numpy()
        err_U_np =torch.squeeze(err_U.cpu()).data.numpy()
        err_div_np =torch.squeeze(err_div.cpu()).data.numpy()

        my_cmap = cm.jet
        my_cmap.set_over('gray')
        my_cmap.set_under('gray')

        title_list = []
        numLoss = 0
        if mconf['pL2Lambda'] > 0:
            numLoss +=1

        if mconf['divL2Lambda'] > 0:
            numLoss +=1

        if mconf['pL1Lambda'] > 0:
            numLoss +=1

        if mconf['divL1Lambda'] > 0:
            numLoss +=1

        if mconf['pL2Lambda'] > 0:
            title_list.append(str(mconf['pL2Lambda']) + ' * L2(p)')

        if mconf['divL2Lambda'] > 0:
            title_list.append(str(mconf['divL2Lambda']) + ' * L2(div)')

        if mconf['pL1Lambda'] > 0:
            title_list.append(str(mconf['pL1Lambda']) + ' * L1(p)')

        if mconf['divL1Lambda'] > 0:
            title_list.append(str(mconf['divL1Lambda']) + ' * L1(div)')

        title = ''
        for string in range(0, numLoss - 1):
            title += title_list[string] + ' + '
        title += title_list[numLoss-1]

        nrow = 3
        ncol = 3

        fig = plt.figure(figsize=(ncol+1, nrow+1))
        gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1],
                         wspace=0.1, hspace=0.1, top=0.9, bottom=0.01, left=0.1, right=0.9)
        fig.suptitle('FluidNet output for loss = ' + title )

        #plt.subplot2grid((3,3), (0,0))
        ax = plt.subplot(gs[0,0])
        ax.set_title('P_target')
        ax.axis('off')
        ax.imshow(p_tar_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_val_p,max_val_p])

        #plt.subplot2grid((3,3), (1,0))
        ax = plt.subplot(gs[1,0])
        ax.set_title('P_predicted')
        ax.axis('off')
        ax.imshow(p_out_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_val_p,max_val_p])

        #plt.subplot2grid((3,3), (2,0))
        ax = plt.subplot(gs[2,0])
        ax.set_title('error P')
        ax.axis('off')
        ax.imshow(err_p_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[0,max_err_p])

        #plt.subplot2grid((3,3), (0,1))
        ax = plt.subplot(gs[0,1])
        ax.set_title('|U|_target')
        ax.axis('off')
        ax.imshow(U_tar_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_val_U,max_val_U])

        #plt.subplot2grid((3,3), (1,1))
        ax = plt.subplot(gs[1,1])
        ax.set_title('|U|_predicted')
        ax.axis('off')
        ax.imshow(U_out_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_val_U,max_val_U])

        #plt.subplot2grid((3,3), (2,1))
        ax = plt.subplot(gs[2,1])
        ax.set_title('error |U|')
        ax.axis('off')
        ax.imshow(err_U_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[0,max_err_U])

        #plt.subplot2grid((3,3), (0,2))
        ax = plt.subplot(gs[0,2])
        ax.set_title('div at output')
        ax.axis('off')
        ax.imshow(div_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_div,max_div])

        #plt.subplot2grid((3,3), (1,2))
        ax = plt.subplot(gs[1,2])
        ax.set_title('div error')
        ax.axis('off')
        ax.imshow(err_div_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[0,max_div**2])

        #cbar_ax_p = fig.add_axes([0.08, 0.45, 0.01, 0.33])
        #fig.colorbar(imP, cax=cbar_ax_p, orientation='vertical')
        #cbar_ax_U = fig.add_axes([0.375, 0.45, 0.01, 0.33])
        #fig.colorbar(imU, cax=cbar_ax_U, orientation='vertical')
        plt.show(block=True)


    #********************************* Run the model and print ****************************
    from itertools import count

    batch_print = [100, 1000, 4000, 5000, 8000, 10000, 15000, 20000]
    def val():
        net.eval()
        loss = nn.MSELoss()
        total_val_loss = 0
        for batch_idx, (data, target) in zip(count(step=1), test_loader):
            with torch.no_grad():
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                if batch_idx in batch_print:
                    out_p, out_U = net(data)
                    target_p = target[:,0].unsqueeze(1)
                    out_div = fluid.velocityDivergence(\
                            out_U.contiguous(), \
                            data[:,3].unsqueeze(1).contiguous())
                    target_div = torch.zeros_like(out_div)

                    loss_size = 0
                    # Measure loss and save it
                    plotField([out_p, out_U, out_div], target, mconf, data[:,3].unsqueeze(1).contiguous())


    #********************************* Run one epoch ***************************************

    print('Plotting results at epoch ' + str(state['epoch']) )
    # Train on training set and test on validation set
    val()

finally:
    # Delete model_saved.py
    print()
    print('Deleting' + temp_model)
    glob.os.remove(temp_model)
