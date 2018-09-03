import glob
import argparse
import yaml

import torch
import torch.nn as nn
import torch.autograd
import time

import matplotlib
if 'DISPLAY' not in glob.os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

import numpy as np
import numpy.ma as ma

import pyevtk.hl as vtk

from shutil import copyfile
import importlib.util

import lib
import lib.fluid as fluid

# Usage python3 plume.py
# Use python3 plume.py -h for more details

#**************************** Load command line arguments *********************

parser = argparse.ArgumentParser(description='Buoyant plume simulation. \n'
        'Read plumeConfig.yaml for more information', \
        formatter_class= lib.SmartFormatter)
parser.add_argument('--simConf',
        default='plumeConfig.yaml',
        help='R|Simulation yaml config file.\n'
        'Overwrites parameters from trainingConf file.\n'
        'Default: plumeConfig.yaml')
parser.add_argument('--trainingConf',
        default='config.yaml',
        help='R|Training yaml config file.\n'
        'Default: config.yaml')
parser.add_argument('--modelDir',
        help='R|Neural network model location.\n'
        'Default: written in simConf file.')
parser.add_argument('--modelFilename',
        help='R|Model name.\n'
        'Default: written in simConf file.')
parser.add_argument('--outputFolder',
        help='R|Folder for sim output.\n'
        'Default: written in simConf file.')
parser.add_argument('--restartSim', action='store_true', default=False,
        help='R|Restarts simulation from checkpoint.\n'
        'Default: written in simConf file.')

arguments = parser.parse_args()

# Loading a YAML object returns a dict
with open(arguments.simConf, 'r') as f:
    simConf = yaml.load(f)
with open(arguments.trainingConf, 'r') as f:
    conf = yaml.load(f)

if not arguments.restartSim:
    restart_sim = simConf['restartSim']
else:
    restart_sim = arguments.restartSim

folder = arguments.outputFolder or simConf['outputFolder']
if (not glob.os.path.exists(folder)):
    glob.os.makedirs(folder)

restart_config_file = glob.os.path.join('/', folder, 'plumeConfig.yaml')
restart_state_file = glob.os.path.join('/', folder, 'restart.pth')
if restart_sim:
    # Check if configPlume.yaml exists in folder
    assert glob.os.path.isfile(restart_config_file), 'YAML config file does not exists for restarting.'
    with open(restart_config_file) as f:
        simConfig = yaml.load(f)

conf['modelDir'] = arguments.modelDir or simConf['modelDir']
assert (glob.os.path.exists(conf['modelDir'])), 'Directory ' + str(conf['modelDir']) + ' does not exists'
conf['modelFilename'] = arguments.modelFilename or simConf['modelFilename']
conf['modelDirname'] = conf['modelDir'] + '/' + conf['modelFilename']
resume = False # For training, at inference set always to false


print('Active CUDA Device: GPU', torch.cuda.current_device())
print()

m = 'jacobi'
conf['modelDir'] = '/data1/aalgua/data_training/model_divL2'
file_loss = 'FluidNet_Jacobi.npy'

path = conf['modelDir']
path_list = path.split(glob.os.sep)
saved_model_name = glob.os.path.join('/', *path_list, path_list[-1] + '_saved.py')
temp_model = glob.os.path.join('lib', path_list[-1] + '_saved_simulate.py')
try:
    copyfile(saved_model_name, temp_model)
    
    assert glob.os.path.isfile(temp_model), temp_model  + ' does not exits!'
    spec = importlib.util.spec_from_file_location('model_saved', temp_model)
    model_saved = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_saved)
    
    te = lib.FluidNetDataset(conf, 'te', save_dt=4, resume=resume) # Test instance of custom Dataset
    
    conf, mconf = te.createConfDict()
    
    cpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_conf.pth')
    mcpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_mconf.pth')
    assert glob.os.path.isfile(mcpath), cpath  + ' does not exits!'
    assert glob.os.path.isfile(mcpath), mcpath  + ' does not exits!'
    conf.update(torch.load(cpath))
    mconf.update(torch.load(mcpath))
    
    print('==> overwriting mconf with user-defined simulation parameters')
    # Overwrite mconf values with user-defined simulation values.
    mconf.update(simConf)
    
    print('==> loading model')
    mpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_lastEpoch_best.pth')
    assert glob.os.path.isfile(mpath), mpath  + ' does not exits!'
    state = torch.load(mpath)
    
    print('Data loading: done')
    batch_size = 10
    num_workers = 3
    test_loader = torch.utils.data.DataLoader(te, batch_size=batch_size, \
                           num_workers=num_workers, shuffle=False, pin_memory=True)

    #********************************** Create the model ***************************
    with torch.no_grad():
    
        cuda = torch.device('cuda')
    
        net = model_saved.FluidNet(mconf, dropout=False)
        if torch.cuda.is_available():
            net = net.cuda()
    
        net.load_state_dict(state['state_dict'])
    
        #*********************** Simulation parameters **************************

        mconf['jacobiIter'] = 28
        mconf['pTol'] = 0
        mconf['dt'] = 0.8
        method = m
        max_iter = 64
        divLoss = nn.MSELoss()
        losses_plot = np.empty((64,2))
        losses_plot[:,0] = np.linspace(0, 63, num=64)
        n_batches = 0

        for n in range(0, 319):
            data, _ = te.__getitem__(64*n)
        #for batch_idx, (data, _) in enumerate(test_loader):
            #if batch_idx % 20 == 0:
            #    print('[{}/{} ({:.0f}%)] \t'.format(
            #    batch_idx * len(data), len(test_loader.dataset),
            #                100. * batch_idx / len(test_loader)))
            data = data.cuda().unsqueeze(0)
            print(data.size())
            batch_dict = {}
            batch_dict['U'] = data[:,1:3].contiguous()
            batch_dict['flags'] = data[:,3].unsqueeze(1).contiguous()
            batch_dict['p'] = torch.zeros_like(batch_dict['flags'])
            it = 0
            while (it < max_iter):
                lib.simulate(conf, mconf, batch_dict, net, method)
                div = fluid.velocityDivergence(batch_dict['U'], batch_dict['flags'])
                target = torch.zeros_like(div)
                loss = divLoss(div, target)
                losses_plot[it,1] += loss.data.item()
                it += 1
            n_batches += 1
        losses_plot[:,1] /= n_batches

        np.save(file_loss, losses_plot)
finally:
    # Properly deleting model_saved.py, even when ctrl+C
    print()
    print('Deleting' + temp_model)
    glob.os.remove(temp_model)

