#!/usr/bin/env python3
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
from math import inf
import glob
import shutil
from util_print import summary


#********************************** Define Config ******************************

from config import conf # Dictionnary with configuration and model params.
#TODO: allow to overwrite params from the command line by parsing.

conf['modelDirname'] = conf['modelDir'] + conf['modelFilename']

#*********************************** Select the GPU ****************************

print('Active CUDA Device: GPU', torch.cuda.current_device())

from dataset_load import FluidNetDataset
import torch.utils.data

tr = FluidNetDataset(conf, 'tr', save_dt=4) # Training instance of custom Dataset
te = FluidNetDataset(conf, 'te', save_dt=4) # Test instance of custom Dataset

train_loader = torch.utils.data.DataLoader(tr, batch_size=conf['batch_size'], \
        num_workers=conf['num_workers'], shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(te, batch_size=conf['batch_size'], \
        num_workers=conf['num_workers'], shuffle=False, pin_memory=True)

resume = conf['resume_training']
if resume:
    print('==> loading checkpoint')
    mpath = glob.os.path.join('model','convModel_lastEpoch_best.pth')
    state = torch.load(mpath)

print('Data loading: done')
#********************************** Create the model ***************************

print('')
print('----- Model ------')

class ScaleNet(nn.Module):
    def __init__(self):
        super(ScaleNet, self).__init__()

    def forward(self, x, scale, invertScale=True):
        if invertScale:
            bsz = x.size(0)
            # Rehaspe form (b x 2/3 x d x h x w) to (b x -1)
            y = x.view(bsz, -1)
            # Calculate std using Bessel's correction (correction with n/n-1)
            std = torch.std(y, dim=1, keepdim=True) # output is size (b x 1)
            scale = torch.clamp(std, \
                conf['newModel']['normalizeInputThreshold'] , inf)
            scale = scale.view(bsz, 1, 1, 1, 1)
            x = torch.div(x, scale)

        else:
            x = torch.mul(x, scale)

        return x, scale

from flags_to_occupancy import FlagsToOccupancy

class FluidNet(nn.Module):
    # For now, only 2D model. Add 2D/3D option. Only known from data!
    # Also, build model with MSE of pressure as loss func, therefore input is velocity
    # and output is pressure, to be compared to target pressure.
    def __init__(self, conf):
        super(FluidNet, self).__init__()

        self.scale = ScaleNet()
        # Input channels = 3 (Ux, Uy, flags)
        # We add padding to make sure that Win = Wout and Hin = Hout with ker_size=3
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=1)
        self.modDown1 = torch.nn.AvgPool2d(kernel_size=2)
        self.modDown2 = torch.nn.AvgPool2d(kernel_size=4)

        self.upscale1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.upscale2 = torch.nn.Upsample(scale_factor=4, mode='nearest')

        # Output channels = 1 (pressure)
        self.convOut = torch.nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):

        UDiv = torch.cat((x.select(1,0).unsqueeze(1), x.select(1,1).unsqueeze(1)), 1)
        flags = x.select(1,2).unsqueeze(1)
        # Apply setWallBcs to zero out obstacle velocities on the boundary

        # Apply scale to input velocity
        s = 0
        UDiv, s = self.scale(UDiv, s, invertScale=True)
        # Flags to occupancy
        flags = FlagsToOccupancy(flags)
        x[:,0,:,:,:] = UDiv.select(1,0)
        x[:,1,:,:,:] = UDiv.select(1,1)
        x[:,2,:,:,:] = flags.squeeze(1)

        # Squeeze unary dimension as we are in 2D
        x = torch.squeeze(x,2)
        x = F.relu(self.conv1(x))

        # We divide the network in 3 banks, applying average pooling
        x0 = F.relu(self.conv2(x))

        x1 = self.modDown1(x)

        x1 = F.relu(self.conv2(x1))
        x2 = self.modDown2(x)
        x2 = F.relu(self.conv2(x2))

        # Process every bank in parallel
        x0 = F.relu(self.conv2(x0))
        x1 = F.relu(self.conv2(x1))
        x2 = F.relu(self.conv2(x2))

        # Upsample banks 1 and 2 to bank 0 size and accumulate inputs
        x1 = self.upscale1(x1)
        x2 = self.upscale2(x2)
        x = x0 + x1 + x2

        # Apply lasts 2 convolutions
        x = F.relu(self.conv3(x))
        x = F.relu(self.convOut(x))

        # Add back the unary dimension
        x = torch.unsqueeze(x, 2)

        x, s = self.scale(x, s, invertScale=False)
        return x

# Create model and print layers and params
net = FluidNet(conf)
if torch.cuda.is_available():
    net = net.cuda()
summary(net, (3,1,128,128))

if resume:
    net.load_state_dict(state['state_dict'])

#********************************** Define the optimizer ***********************
print('==> defining optimizer')

optimizer = torch.optim.Adam(net.parameters(), lr=0.0025)
if (resume):
    optimizer.load_state_dict(state['optimizer'])

for param_group in optimizer.param_groups:
    print('lr of optimizer')
    print(param_group['lr'])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',factor = 0.4, patience = 10, verbose = True, threshold = 3e-4, threshold_mode = 'rel')

#********************************* Training ************************************
def train(epoch):
    #set model to train
    net.train()

    #initialise loss scores
    total_train_loss = 0

    #loop through data, sorted into batches
    for batch_idx, (data, target) in enumerate (train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        # Run the model forward
        output = net(data)

        # Measure loss and save it
        loss_size = torch.sum( (output - target)**2 )
        err = (target - output)**2

        # Print statistics
        total_train_loss += loss_size.data[0]

        # Step the optimizer
        loss_size.backward()
        optimizer.step()

        # Print every 20th batch of an epoch
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader)))

    # Divide loss by dataset length
    total_train_loss /= len(train_loader.dataset)

    # Print for the whole dataset
    print('\nTrain set: Avg loss: {:.6f}'.format(total_train_loss))

    # Return loss scores
    return total_train_loss

#********************************* Validation **********************************

# val is the same as train, except we don't perform backprop (with model.eval())
def val():
    net.eval()
    total_val_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = net(data)
        loss_size = torch.sum( (output - target)**2 )
        total_val_loss += loss_size.data[0]

    total_val_loss /= len(test_loader.dataset)
    print('Validation set: Avg loss: {:.6f}\n'.format(total_val_loss))
    return total_val_loss

#********************************* Run epochs **********************************

def save_checkpoint(state, is_best, save_path, filename):
    filename = glob.os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = glob.os.path.join(save_path, 'convModel_lastEpoch_best.pth')
        shutil.copyfile(filename, bestname)

# Create some arrays for recording results
train_loss_plot = np.empty((0,2))
val_loss_plot = np.empty((0,2))

# Save loss as numpy arrays to disk
save_dir = conf['save_plots_dir']
file_train = glob.os.path.join(save_dir, 'train_loss')
file_val = glob.os.path.join(save_dir, 'val_loss')

# raw_input returns the empty string for "enter"
yes = {'yes','y', 'ye', ''}
no = {'no','n'}

if resume:
    start_epoch = state['epoch']
else:
    start_epoch = 1
    # Delete plot file
    if (glob.os.path.isfile(file_train + '.npy') and glob.os.path.isfile(file_val + '.npy')):
        print('Are you sure you want to delete plot files? [y/n]')
        choice = input().lower()
        if choice in yes:
            glob.os.remove(file_train + '.npy')
            glob.os.remove(file_val + '.npy')
        elif choice in no:
            print('Exiting program')
            sys.exit()
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'")

state = {'bestPerf': float('Inf')}
mPath = 'model'
model_save_path = glob.os.path.join(mPath, 'convModel')
n_epochs = conf['maxEpochs']

print('')
print('==> Beginning simulation')
for epoch in range(start_epoch, n_epochs+1):
    # Train on training set and test on validation set
    train_loss = train(epoch)
    val_loss = val()

    #Step scheduler, will reduce LR if loss has plateaued
    scheduler.step(val_loss)

    # Store training loss function
    train_loss_plot = np.append(train_loss_plot, [[epoch, train_loss]], axis=0)
    val_loss_plot = np.append(val_loss_plot, [[epoch, val_loss]], axis=0)

    # Check if this is the best model so far and if so save to disk
    is_best = False
    if val_loss < state['bestPerf']:
        is_best = True
    save_checkpoint({ \
        'epoch': epoch + 1, \
        'state_dict': net.state_dict(), \
        'optimizer' : optimizer.state_dict(), \
        'bestPerf': val_loss, \
        }, is_best, mPath, 'convModel_lastEpoch.pth')

    # Save loss to disk -- TODO: Check if there is a more efficient way, instead
    # of loading the whole file...
    if epoch % conf['freq_to_file'] == 0:
        plot_train_file = file_train + '.npy'
        plot_val_file = file_val + '.npy'
        train_old = np.empty((0,2))
        val_old = np.empty((0,2))
        if (glob.os.path.isfile(plot_train_file) and glob.os.path.isfile(plot_val_file)):
            train_old = np.load(plot_train_file)
            val_old = np.load(plot_val_file)
        train_loss = np.append(train_old, train_loss_plot, axis=0)
        val_loss = np.append(val_old, val_loss_plot, axis=0)
        np.save(file_val, val_loss)
        np.save(file_train, train_loss)
        # Reset arrays
        train_loss_plot = np.empty((0,2))
        val_loss_plot = np.empty((0,2))

