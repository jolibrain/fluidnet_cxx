import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import matplotlib.pyplot as plt
import numpy as np
import glob
from shutil import copyfile
import importlib.util

import lib
import lib.fluid as fluid
from config import defaultConf # Dictionnary with configuration and model params.

#********************************** Define Config ******************************

#TODO: allow to overwrite params from the command line by parsing.

conf = defaultConf.copy()
conf['modelDirname'] = conf['modelDir'] + conf['modelFilename']
resume = conf['resumeTraining']
num_workers = conf['numWorkers']
batch_size = conf['batchSize']
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


if resume:
    print('==> loading checkpoint')
    mpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_lastEpoch_best.pth')
    assert glob.os.path.isfile(mpath), mpath  + ' does not exits!'
    state = torch.load(mpath)

    print('==> overwriting conf and file_mconf')
    cpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_conf.pth')
    mcpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_mconf.pth')
    assert glob.os.path.isfile(mpath), cpath  + ' does not exits!'
    assert glob.os.path.isfile(mpath), mcpath  + ' does not exits!'
    conf = torch.load(cpath)
    mconf = torch.load(mcpath)

    print('==> copying and loading corresponding model module')
    path = conf['modelDir'] + '/'
    path_list = path.split(glob.os.sep)
    saved_model_name = glob.os.path.join(*path_list[:-1], path_list[-2] + '_saved.py')
    print(saved_model_name)
    temp_model = glob.os.path.join('lib', path_list[-2] + '_saved_resume.py')
    print(temp_model)
    copyfile(saved_model_name, temp_model)

    assert glob.os.path.isfile(temp_model), temp_model  + ' does not exits!'
    spec = importlib.util.spec_from_file_location('model_saved', temp_model)
    model_saved = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_saved)

print('Data loading: done')

try:
    # Create train and validation loaders
    print('Number of workers: ' + str(num_workers) )

    train_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size, \
            num_workers=num_workers, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(te, batch_size=batch_size, \
            num_workers=num_workers, shuffle=False, pin_memory=True)

    #********************************** Create the model ***************************
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_uniform_(m.weight)

    print('')
    print('----- Model ------')

    # Create model and print layers and params
    if not resume:
        net = lib.FluidNet(mconf)
    else:
        net = model_saved.FluidNet(mconf)

    if torch.cuda.is_available():
        net = net.cuda()

    # Initialize network weights with Kaiming normal method (a.k.a MSRA)
    net.apply(init_weights)
    lib.summary(net, (5,1,128,128))

    if resume:
        net.load_state_dict(state['state_dict'])

    #********************************** Define the optimizer ***********************

    print('==> defining optimizer')

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0025)

    for param_group in optimizer.param_groups:
        print('lr of optimizer')
        print(param_group['lr'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',factor = 0.6, patience = 10, verbose = True, threshold = 3e-4, threshold_mode = 'rel')

    #********************************* Training ************************************

    def train(epoch):
        #set model to train
        net.train()

        #initialise loss scores
        total_train_loss = 0
        p_l2_total_loss = 0
        div_l2_total_loss = 0
        p_l1_total_loss = 0
        div_l1_total_loss = 0

        n_batches = 0 # Number processed

        # Loss types
        _pL2Loss = nn.MSELoss()
        _divL2Loss = nn.MSELoss()
        _pL1Loss = nn.L1Loss()
        _divL1Loss = nn.L1Loss()

        # Loss lambdas (multiply the corresponding loss)
        pL2Lambda = mconf['pL2Lambda']
        divL2Lambda = mconf['divL2Lambda']
        pL1Lambda = mconf['pL1Lambda']
        divL1Lambda = mconf['divL1Lambda']

        #loop through data, sorted into batches
        for batch_idx, (data, target) in enumerate (train_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Set gradients to zero, clearing previous batches.
            optimizer.zero_grad()

            # data indexes     |           |
            #       (dim 1)    |    2D     |    3D
            # ----------------------------------------
            #   DATA:
            #       pDiv       |    0      |    0
            #       UDiv       |    1:3    |    1:4
            #       flags      |    3      |    4
            #       densityDiv |    4      |    5
            #   TARGET:
            #       p          |    0      |    0
            #       U          |    1:3    |    1:4
            #       density    |    3      |    4

            # Run the model forward
            out_p, out_U = net(data)

            # Calculate targets
            target_p = target[:,0].unsqueeze(1)
            out_div = fluid.velocityDivergence(out_U.contiguous(), data[:,3].unsqueeze(1).contiguous())
            target_div = torch.zeros_like(out_div)

            # Measure loss and save it
            pL2Loss = pL2Lambda *_pL2Loss(out_p, target_p)
            divL2Loss = divL2Lambda *_divL2Loss(out_div, target_div)
            pL1Loss =  pL1Lambda *_pL1Loss(out_p, target_p)
            divL1Loss = divL1Lambda *_divL1Loss(out_div, target_div)

            loss_size =  pL2Loss + divL2Loss + pL1Loss + divL1Loss

            # Print statistics
            p_l2_total_loss += pL2Loss.data.item()
            div_l2_total_loss += divL2Loss.data.item()
            p_l1_total_loss += pL1Loss.data.item()
            div_l1_total_loss += divL1Loss.data.item()
            total_train_loss += loss_size.data.item()

            # Step the optimizer
            loss_size.backward()
            optimizer.step()

            n_batches += 1

            # Print every 20th batch of an epoch
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] \t'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)))

        # Divide loss by dataset length
        p_l2_total_loss /= n_batches
        div_l2_total_loss /= n_batches
        p_l1_total_loss /= n_batches
        div_l1_total_loss /= n_batches
        total_train_loss /= n_batches

        # Print for the whole dataset
        print('\nTrain set: Avg total loss: {:.6f} (L2(p): {:.6f}; L2(div): {:.6f}; L1(p): {:.6f}; L1(div): {:.6f})'.format(\
                        total_train_loss, p_l2_total_loss, div_l2_total_loss, p_l1_total_loss, div_l1_total_loss))

        # Return loss scores
        return total_train_loss, p_l2_total_loss, div_l2_total_loss, \
                p_l1_total_loss, div_l1_total_loss

    #********************************* Validation **********************************

    # val is the same as train, except we don't perform backprop (with model.eval())
    def val():
        net.eval()
        total_val_loss = 0
        p_l2_total_loss = 0
        div_l2_total_loss = 0
        p_l1_total_loss = 0
        div_l1_total_loss = 0

        n_batches = 0

        # Loss types
        _pL2Loss = nn.MSELoss()
        _divL2Loss = nn.MSELoss()
        _pL1Loss = nn.L1Loss()
        _divL1Loss = nn.L1Loss()

        # Loss lambdas (multiply the corresponding loss)
        pL2Lambda = mconf['pL2Lambda']
        divL2Lambda = mconf['divL2Lambda']
        pL1Lambda = mconf['pL1Lambda']
        divL1Lambda = mconf['divL1Lambda']

        for batch_idx, (data, target) in enumerate(test_loader):
            print('here')
            with torch.no_grad():
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                #data, target = Variable(data), Variable(target)
                out_p, out_U = net(data)

                # Calculate targets
                target_p = target[:,0].unsqueeze(1)
                out_div = fluid.velocityDivergence(out_U.contiguous(), data[:,3].unsqueeze(1).contiguous())
                target_div = torch.zeros_like(out_div)

                # Measure loss and save it
                pL2Loss = pL2Lambda *_pL2Loss(out_p, target_p)
                divL2Loss = divL2Lambda *_divL2Loss(out_div, target_div)
                pL1Loss =  pL1Lambda *_pL1Loss(out_p, target_p)
                divL1Loss = divL1Lambda *_divL1Loss(out_div, target_div)

                loss_size =  pL2Loss + divL2Loss + pL1Loss + divL1Loss

                p_l2_total_loss += pL2Loss.data.item()
                div_l2_total_loss += divL2Loss.data.item()
                p_l1_total_loss += pL1Loss.data.item()
                div_l1_total_loss += divL1Loss.data.item()
                total_val_loss += loss_size.item()
                n_batches += 1

        p_l2_total_loss /= n_batches
        div_l2_total_loss /= n_batches
        p_l1_total_loss /= n_batches
        div_l1_total_loss /= n_batches
        total_val_loss /= n_batches

        print('\nValidation set: Avg total loss: {:.6f} (L2(p): {:.6f}; L2(div): {:.6f}; L1(p): {:.6f}; L1(div): {:.6f} )'.format(\
                total_val_loss, p_l2_total_loss, div_l2_total_loss, p_l1_total_loss, div_l1_total_loss))

        # Return loss scores
        return total_val_loss, p_l2_total_loss, div_l2_total_loss, \
                p_l1_total_loss, div_l1_total_loss

    #********************************* Prepare saving files *******************************

    def save_checkpoint(state, is_best, save_path, filename):
        filename = glob.os.path.join(save_path, filename)
        torch.save(state, filename)
        if is_best:
            bestname = glob.os.path.join(save_path, 'convModel_lastEpoch_best.pth')
            copyfile(filename, bestname)

    # Create some arrays for recording results
    train_loss_plot = np.empty((0,6))
    val_loss_plot = np.empty((0,6))

    # Save loss to disk
    m_path = conf['modelDir']
    model_save_path = glob.os.path.join(m_path, conf['modelFilename'])

    # Save loss as numpy arrays to disk
    p_path = conf['modelDir']
    file_train = glob.os.path.join(p_path, 'train_loss')
    file_val = glob.os.path.join(p_path, 'val_loss')

    # Save mconf and conf to disk
    file_conf = glob.os.path.join(m_path, conf['modelFilename'] + '_conf.pth')
    file_mconf = glob.os.path.join(m_path, conf['modelFilename'] + '_mconf.pth')


    # raw_input returns the empty string for "enter"
    yes = {'yes','y', 'ye', ''}
    no = {'no','n'}

    if resume:
        start_epoch = state['epoch']
    else:
        start_epoch = 1
        if ((not glob.os.path.exists(p_path)) and (not glob.os.path.exists(m_path))):
            if (p_path == m_path):
                glob.os.makedirs(p_path)
            else:
                glob.os.makedirs(m_path)
                glob.os.makedirs(p_path)

        # Here we are a bit barbaric, and we copy the whole model.py into the saved model
        # folder, so that we don't lose the network architecture.
        # We do that only if not resuming training.
        path, last = glob.os.path.split(m_path)
        saved_model_name = glob.os.path.join(path, last, last + '_saved.py')
        copyfile('lib/model.py', saved_model_name)

        # Delete plot file if starting from scratch
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

    # Save dicts
    torch.save(conf, file_conf)
    torch.save(mconf, file_mconf)

    #********************************* Run epochs ****************************************

    n_epochs = conf['maxEpochs']
    if not resume:
        state = {}
        state['bestPerf'] = float('Inf')

    print('')
    print('==> Beginning simulation')
    for epoch in range(start_epoch, n_epochs+1):
        # Train on training set and test on validation set
        train_loss, p_l2_tr, div_l2_tr, p_l1_tr, div_l1_tr = train(epoch)
        val_loss, p_l2_val, div_l2_val, p_l1_val, div_l1_val = val()

        #Step scheduler, will reduce LR if loss has plateaued
        scheduler.step(val_loss)

        # Store training loss function
        train_loss_plot = np.append(train_loss_plot, [[epoch, train_loss, p_l2_tr, div_l2_tr, p_l1_tr, div_l1_tr]], axis=0)
        val_loss_plot = np.append(val_loss_plot, [[epoch, val_loss, p_l2_val, div_l2_val, p_l1_val, div_l1_val]], axis=0)

        # Check if this is the best model so far and if so save to disk
        is_best = False
        state['epoch'] = epoch +1
        state['state_dict'] = net.state_dict()
        state['optimizer'] = optimizer.state_dict()

        if val_loss < state['bestPerf']:
            is_best = True
            state['bestPerf'] = val_loss
        save_checkpoint(state, is_best, m_path, 'convModel_lastEpoch.pth')

        # Save loss to disk -- TODO: Check if there is a more efficient way, instead
        # of loading the whole file...
        if epoch % conf['freqToFile'] == 0:
            plot_train_file = file_train + '.npy'
            plot_val_file = file_val + '.npy'
            train_old = np.empty((0,6))
            val_old = np.empty((0,6))
            if (glob.os.path.isfile(plot_train_file) and glob.os.path.isfile(plot_val_file)):
                train_old = np.load(plot_train_file)
                val_old = np.load(plot_val_file)
            train_loss = np.append(train_old, train_loss_plot, axis=0)
            val_loss = np.append(val_old, val_loss_plot, axis=0)
            np.save(file_val, val_loss)
            np.save(file_train, train_loss)
            # Reset arrays
            train_loss_plot = np.empty((0,6))
            val_loss_plot = np.empty((0,6))

finally:
    if resume:
        # Delete model_saved_resume.py
        print()
        print('Deleting ' + temp_model)
        glob.os.remove(temp_model)
