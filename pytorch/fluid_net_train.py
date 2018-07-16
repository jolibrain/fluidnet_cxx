import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data

import matplotlib.pyplot as plt
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
resume = conf['resumeTraining']
if (conf['preprocOnly']):
    print('Running preprocessing only')
    resume = False

#*********************************** Select the GPU ****************************

print('Active CUDA Device: GPU', torch.cuda.current_device())

tr = lib.FluidNetDataset(conf, 'tr', save_dt=4, resume=resume) # Training instance of custom Dataset
te = lib.FluidNetDataset(conf, 'te', save_dt=4, resume=resume) # Test instance of custom Dataset

if (conf['preprocOnly']):
    sys.exit()

# We create two conf files, general params and model params.
conf, mconf = tr.createConfDict()

train_loader = torch.utils.data.DataLoader(tr, batch_size=conf['batchSize'], \
        num_workers=conf['numWorkers'], shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(te, batch_size=conf['batchSize'], \
        num_workers=conf['numWorkers'], shuffle=False, pin_memory=True)

if resume:
    print('==> loading checkpoint')
    mpath = glob.os.path.join(conf['modelDir'],conf['modelFilename'] + '_lastEpoch_best.pth')
    assert glob.os.path.isfile(mpath), mpath  + ' does not exits!'
    state = torch.load(mpath)

print('Data loading: done')

#********************************** Create the model ***************************
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)

print('')
print('----- Model ------')

# Create model and print layers and params
net = lib.FluidNet(mconf)

if torch.cuda.is_available():
    net = net.cuda()

# Initialize network weights with Kaiming normal method (a.k.a MSRA)
net.apply(init_weights)
#lib.summary(net, (3,1,128,128))

if resume:
    net.load_state_dict(state['state_dict'])

#********************************** Define the optimizer ***********************

print('==> defining optimizer')

optimizer = torch.optim.Adam(net.parameters(), lr=0.0025)
if (resume):
    optimizer.load_state_dict(state['optimizer'])
    optimizer.param_groups[0]['lr'] = 0.0025


for param_group in optimizer.param_groups:
    print('lr of optimizer')
    print(param_group['lr'])

#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',factor = 0.4, patience = 10, verbose = True, threshold = 3e-4, threshold_mode = 'rel')

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
        optimizer.zero_grad()

        # Run the model forward
        output = net(data)

        # Measure loss and save it
        target_p = target[:,0].unsqueeze(1)
        loss_size = torch.sum( (output - target_p)**2 )

        # Print statistics
        total_train_loss += loss_size.data.item()

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
        with torch.no_grad():
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            #data, target = Variable(data), Variable(target)
            output = net(data)
            target_p = target[:,0].unsqueeze(1)
            loss_size = torch.sum( (output - target_p)**2 )
            total_val_loss += loss_size.item()

    total_val_loss /= len(test_loader.dataset)
    print('Validation set: Avg loss: {:.6f}\n'.format(total_val_loss))
    return total_val_loss

#********************************* Prepare saving files *******************************

def save_checkpoint(state, is_best, save_path, filename):
    filename = glob.os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = glob.os.path.join(save_path, 'convModel_lastEpoch_best.pth')
        copyfile(filename, bestname)

# Create some arrays for recording results
train_loss_plot = np.empty((0,2))
val_loss_plot = np.empty((0,2))

# Save loss to disk
m_path = conf['modelDir']
model_save_path = glob.os.path.join(m_path, 'convModel')

# Save loss as numpy arrays to disk
p_path = conf['plotDir']
file_train = glob.os.path.join(p_path, 'train_loss')
file_val = glob.os.path.join(p_path, 'val_loss')

# raw_input returns the empty string for "enter"
yes = {'yes','y', 'ye', ''}
no = {'no','n'}

if resume:
    start_epoch = state['epoch']
else:
    start_epoch = 1
    if ((not glob.os.path.exists(p_path)) and (not glob.os.path.exists(m_path))):
        glob.os.makedirs(p_path)
        glob.os.makedirs(m_path)

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

#********************************* Run epochs ****************************************

state = {'bestPerf': float('Inf')}
n_epochs = conf['maxEpochs']

print('')
print('==> Beginning simulation')
for epoch in range(start_epoch, n_epochs+1):
    # Train on training set and test on validation set
    train_loss = train(epoch)
    val_loss = val()

    #Step scheduler, will reduce LR if loss has plateaued
    #scheduler.step(val_loss)

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
        }, is_best, m_path, 'convModel_lastEpoch.pth')

    # Save loss to disk -- TODO: Check if there is a more efficient way, instead
    # of loading the whole file...
    if epoch % conf['freqToFile'] == 0:
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

