import glob
import sys
import argparse
import yaml

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
from utils.flops_benchmark import add_flops_counting_methods

# Parse arguments
parser = argparse.ArgumentParser(description='Training script.', \
        formatter_class= lib.SmartFormatter)
parser.add_argument('--trainingConf',
        default='config.yaml',
        help='R|Training yaml config file.\n'
        '  Default: config.yaml')
parser.add_argument('--modelDir',
        help='R|Output folder location for trained model.\n'
        'When resuming, reads from this location.\n'
        '  Default: written in trainingConf file.')
parser.add_argument('--modelFilename',
        help='R|Model name.\n'
        '  Default: written in trainingConf file.')
parser.add_argument('--dataDir',
        help='R|Dataset location.\n'
        '  Default: written in trainingConf file.')
parser.add_argument('--resume', action="store_true", default=False,
        help='R|Resumes training from checkpoint in modelDir.\n'
        '  Default: written in trainingConf file.')
parser.add_argument('--bsz', type=int,
        help='R|Batch size for training.\n'
        '  Default: written in trainingConf file.')
parser.add_argument('--maxEpochs', type=int,
        help='R|Maximum number training epochs.\n'
        '  Default: written in trainingConf file.')
parser.add_argument('--noShuffle', action="store_true", default=False,
        help='R|Remove dataset shuffle when training.\n'
        '  Default: written in trainingConf file.')
parser.add_argument('--lr', type=float,
        help='R|Learning rate.\n'
        '  Default: written in trainingConf file.')
parser.add_argument('--numWorkers', type=int,
        help='R|Number of parallel workers for dataset loading.\n'
        '  Default: written in trainingConf file.')
parser.add_argument('--outMode', choices=('save', 'show', 'none'),
        help='R|Training debug options. Prints or shows validation dataset.\n'
        ' save = saves plots to disk \n'
        ' show = shows plots in window during training \n'
        ' none = do nothing \n'
        '  Default: written in trainingConf file.')


# ************************** Check arguments *********************************

print('Parsing and checking arguments')

arguments = parser.parse_args()
with open(arguments.trainingConf, 'r') as f:
    conf = yaml.load(f)

conf['dataDir'] = arguments.dataDir or conf['dataDir']
conf['modelDir'] = arguments.modelDir or conf['modelDir']
conf['modelFilename'] = arguments.modelFilename or conf['modelFilename']
conf['modelDirname'] = conf['modelDir'] + '/' + conf['modelFilename']

# If options not defined in cmd line, go to config.yaml to find value.
if not arguments.resume:
    resume = conf['resumeTraining']
else:
    resume = arguments.resume

# If options not defined in cmd line, go to config.yaml to find value.
if arguments.outMode is None:
    output_mode = conf['printTraining']
    assert output_mode == 'save' or output_mode == 'show' or output_mode == 'none',\
            'In config.yaml printTraining options are save, show or none.'
else:
    output_mode = arguments.outMode

# If options not defined in cmd line, go to config.yaml to find value.
if not arguments.noShuffle:
    shuffle_training = conf['shuffleTraining']
else:
    shuffle_training = not arguments.noShuffle

conf['shuffleTraining'] = not arguments.noShuffle


print('Active CUDA Device: GPU', torch.cuda.current_device())

tr = lib.FluidNetDataset(conf, 'tr', save_dt=4, resume=resume)

conf, mconf = tr.createConfDict()

num_workers = arguments.numWorkers or conf['numWorkers']
batch_size = arguments.bsz or conf['batchSize']

print('Data loading: done')

# Create train and validation loaders
print('Number of workers: ' + str(num_workers) )
batch_size = 10
train_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size, \
        num_workers=0, shuffle=shuffle_training)


# FluidNet model 5x1xresxres input


flops_to_plot = np.empty((0,4))

for n in range(4, 8):
    res = 2**n
    print("Resolution = " + str(res))
    data = torch.rand(1,5,1,res,res)
    mconf['model'] = "FluidNet"
    fn = lib.FluidNet(mconf)
    
    fn = add_flops_counting_methods(fn)
    if torch.cuda.is_available():
        fn = fn.cuda().train()
    
    fn.start_flops_count()
    batch = data.cuda()
    _ = fn(batch)
    
    flops_fn = fn.compute_average_flops_cost() / 2 /1e9
     
    mconf['model'] = "ScaleNet"
    sn = lib.FluidNet(mconf)
    
    sn = add_flops_counting_methods(sn)
    if torch.cuda.is_available():
        sn = sn.cuda().train()
    
    sn.start_flops_count()
    data = torch.rand(1,5,1,res,res)
    batch = data.cuda()
    _ = sn(batch)
   
    jacobi_iter = 28
    flops_34 = jacobi_iter*(2*res**3) / 1e9
    
    flops_sn = sn.compute_average_flops_cost() / 2 / 1e9
    flops_to_plot = np.append(flops_to_plot, [[res, flops_fn, \
        flops_sn, flops_34]], axis=0)
    
    # Find Jacobi iteration count for same flops
    # Jacobi algorithm performs in ~O(res^2)
    
    nIter_fn = flops_fn*1e9 / (2*res**3)
    nIter_sn = flops_sn*1e9 / (2*res**3)
    print('Jacobi Iterations for same GFlops as FN: ' + str(nIter_fn))
    print('Jacobi Iterations for same GFlops as SN: ' + str(nIter_sn))
    
    flops_34 = jacobi_iter*(2*res**3)/ 1e9
    print('For ' + str(jacobi_iter) + ' Iter, GFlops: ' + str(flops_34))

print(flops_to_plot)

plt.grid(True)
plt.semilogx(flops_to_plot[:,0], flops_to_plot[:,1], 'k',basex=2, label='FluidNet')
plt.semilogx(flops_to_plot[:,0], flops_to_plot[:,2], 'b', basex=2, label='MultiScale')
plt.semilogx(flops_to_plot[:,0], flops_to_plot[:,3], 'r',  basex=2, label='Jacobi (28 iterations)')
plt.legend()
plt.xlabel('resolution')
plt.ylabel('GFlops')
plt.tight_layout()
plt.savefig('figures/Flops_FluidN_deconv_ScaleN_jacobi_28_smallScale.pdf', type='pdf')


