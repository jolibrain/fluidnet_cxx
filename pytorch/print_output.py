#!/usr/bin/env python3
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data

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

    #********************************* Run the model and print ****************************
    from itertools import count

    batch_print = [100, 1000, 4000, 5000, 8000, 10000, 15000, 20000]
    def val():
        net.eval()

        #initialise loss scores
        total_loss = 0
        p_l2_total_loss = 0
        div_l2_total_loss = 0
        p_l1_total_loss = 0
        div_l1_total_loss = 0
        div_lt_total_loss = 0

        n_batches = 0 # Number processed

        # Loss types
        _pL2Loss = nn.MSELoss()
        _divL2Loss = nn.MSELoss()
        _divLTLoss = nn.MSELoss()
        _pL1Loss = nn.L1Loss()
        _divL1Loss = nn.L1Loss()

        # Loss lambdas (multiply the corresponding loss)
        pL2Lambda = mconf['pL2Lambda']
        divL2Lambda = mconf['divL2Lambda']
        pL1Lambda = mconf['pL1Lambda']
        divL1Lambda = mconf['divL1Lambda']
        divLTLambda = mconf['divLongTermLambda']

        for batch_idx, (data, target) in zip(count(step=1), test_loader):
            with torch.no_grad():
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                if batch_idx in batch_print:
                    out_p, out_U = net(data)
                    flags = data[:,3].unsqueeze(1).contiguous()
                    target_p = target[:,0].unsqueeze(1)
                    out_div = fluid.velocityDivergence(\
                            out_U.contiguous(), \
                            flags)
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
                    #if (divLTLambda > 0):
                    #    div_lt_total_loss += divLTLoss.data.item()
                    total_loss += loss_size.data.item()


                    # Measure loss and save it
                    lib.plotField(out=[out_p, out_U, out_div],
                                  tar=target,
                                  flags=flags,
                                  loss=[total_loss, p_l2_total_loss,
                                        div_l2_total_loss, div_lt_total_loss,
                                        p_l1_total_loss, div_l1_total_loss],
                                  mconf=mconf,
                                  save=False,
                                  y_slice=8)

    #********************************* Run one epoch ***************************************

    print('Plotting results at epoch ' + str(state['epoch']) )
    # Train on training set and test on validation set
    val()

finally:
    # Delete model_saved.py
    print()
    print('Deleting' + temp_model)
    glob.os.remove(temp_model)
