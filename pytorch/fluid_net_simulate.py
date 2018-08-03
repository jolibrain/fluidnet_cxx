import sys
import argparse

import torch
import torch.autograd
import math
import time

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np
import glob
from shutil import copyfile
import importlib.util

import lib
import lib.fluid as fluid
from config import defaultConf

# Use: python3 print_output.py <folder_with_model> <modelName>
# e.g: python3 print_output.py data/model_test convModel
#
# Utility to print data, output or target fields, as well as errors.
# It loads the state and runs one epoch with a batch size = 1. It can be used while
# training, to have some visual help.

#**************************** Load command line arguments *********************
assert (len(sys.argv) == 3), 'Usage: python3 print_output.py <modelDir> <modelName>'
assert (glob.os.path.exists(sys.argv[1])), 'Directory ' + str(sys.argv[1]) + ' does not exists'

def createPlumeBCs(batch_dict, density_val, u_scale, rad):

    cuda = torch.device('cuda')
    # batch_dict at input: {p, UDiv, flags, density}
    assert len(batch_dict) == 4, "Batch must contain 4 tensors (p, UDiv, flags, density)"
    UDiv = batch_dict['U']
    density = batch_dict['density']
    UBC = UDiv.clone().fill_(0)
    UBCInvMask = UDiv.clone().fill_(1)

    # Single density value
    densityBC = density.clone().fill_(0)
    densityBCInvMask = density.clone().fill_(1)

    assert UBC.dim() == 5, 'UBC must have 5 dimensions'
    assert UBC.size(0) == 1, 'Only single batches allowed (inference)'

    xdim = UBC.size(4)
    ydim = UBC.size(3)
    zdim = UBC.size(2)
    is3D = (UBC.size(1) == 3)
    if not is3D:
        assert zdim == 1, 'For 2D, zdim must be 1'
    centerX = xdim // 2
    centerZ = max( zdim // 2, 1.0)
    plumeRad = math.floor(xdim*rad)

    y = 1
    if (not is3D):
        vec = torch.arange(0,2, device=cuda)
    else:
        vec = torch.arange(0,3, device=cuda)
        vec[2] = 0

    vec.mul_(u_scale)

    index_x = torch.arange(0, xdim, device=cuda).view(xdim).expand_as(density[0][0])
    index_y = torch.arange(0, ydim, device=cuda).view(ydim, 1).expand_as(density[0][0])
    if (is3D):
        index_z = torch.arange(0, zdim, device=cuda).view(zdim, 1, 1).expand_as(density[0][0])

    if (not is3D):
        index_ten = torch.stack((index_x, index_y), dim=0)
    else:
        index_ten = torch.stack((index_x, index_y, index_z), dim=0)

    #TODO 3d implementation
    indx_circle = index_ten[:,:,0:4]
    indx_circle[0] -= centerX
    maskInside = (indx_circle[0].pow(2) <= plumeRad*plumeRad)

    # Inside the plume. Set the BCs.

    #It is clearer to just multiply by mask (casted into Float)
    maskInside_f = maskInside.float().clone()
    UBC[:,:,:,0:4] = maskInside_f * vec.view(1,2,1,1,1).expand_as(UBC[:,:,:,0:4])
    UBCInvMask[:,:,:,0:4].masked_fill_(maskInside, 0)

    densityBC[:,:,:,0:4].masked_fill_(maskInside, density_val)
    densityBCInvMask[:,:,:,0:4].masked_fill_(maskInside, 0)

    # Outside the plume. Set the velocity to zero and leave density alone.

    maskOutside = (maskInside == 0)
    UBC[:,:,:,0:4].masked_fill_(maskOutside, 0)
    UBCInvMask[:,:,:,0:4].masked_fill_(maskOutside, 0)

    # Insert the new tensors in the batch_dict.
    batch_dict['UBC'] = UBC
    batch_dict['UBCInvMask'] = UBCInvMask
    batch_dict['densityBC'] = densityBC
    batch_dict['densityBCInvMask'] = densityBCInvMask

    # batch_dict at output = {p, UDiv, flags, density, UBC,
    #                         UBCInvMask, densityBC, densityBCInvMask}

def setConstVals(batch_dict, p, U, flags, density):
    # apply external BCs.
    # batch_dict at output = {p, UDiv, flags, density, UBC,
    #                         UBCInvMask, densityBC, densityBCInvMask}

    # Zero out the U values on the BCs.
    U.mul_(batch_dict['UBCInvMask']);
    # Add back the values we want to specify.
    U.add_(batch_dict['UBC']);
    batch_dict['U'] = U.clone();

    density.mul_(batch_dict['densityBCInvMask']);
    density.add_(batch_dict['densityBC']);
    batch_dict['density'] = density.clone();

def simulate(batch_dict, res, net, sim_method):
    with torch.autograd.no_grad():
        net.eval()

        dt = 0.1;
        maccormackStrength = 0.6;
        sampleOutsideFluid = False;

        buoyancyScale = 0 * (res / 128);
        gravityScale = 0 * (res / 128);

        # Get p, U, flags and density from batch.
        p = batch_dict['p'];
        U = batch_dict['U'];
        flags = batch_dict['flags'];
        density = batch_dict['density'];

        # First advect all scalar fields.
        density = fluid.advectScalar(dt, density, U, flags, method="maccormackFluidNet", \
                    boundary_width=1, sample_outside_fluid=sampleOutsideFluid, \
                    maccormack_strength=maccormackStrength);

        # Self-advect velocity
        U = fluid.advectVelocity(dt, U, flags, method="maccormackFluidNet", \
                boundary_width=1, maccormack_strength=maccormackStrength);

        # Set the manual BCs.
        #setConstVals(batch_dict, p, U, flags, density);

        #gravity = torch.FloatTensor(3).fill_(0);
        #gravity[1] = 1;

        # Add external forces: buoyancy.
        #gravity.mul_(-(fluid.getDx(flags) / 4) * buoyancyScale);

        #fluid.addBuoyancy(U, flags, density, gravity, dt);
        # Add external forces: gravity.
        #gravity.mul_((-fluid.getDx(flags) / 4) * gravityScale);
        #fluid.addGravity(U, flags, gravity, dt);

        # Set the constant domain values.
        #if (sim_method != 'convnet'):
        fluid.setWallBcs(U, flags);
        setConstVals(batch_dict, p, U, flags, density);

        if (sim_method == 'convnet'):
            # fprop the model to perform the pressure projection and velocity calculation.
            # Set wall BCs is performed inside the model, before and after the projection.
            # No need to call it again.
            data = torch.cat((p, U, flags, density), 1)
            out_p, out_U = net(data)
            p = out_p.clone()
            U = out_U.clone()

        else:
            div = fluid.velocityDivergence(U, flags);

            is3D = (U.size(2) > 1);
            pTol = 0;
            maxIter = 34;

            _p, residual = fluid.solveLinearSystemJacobi(flags, div, is3D, p_tol=pTol, \
                    max_iter=maxIter);

            p = _p
            fluid.velocityUpdate(p, U, flags);

        setConstVals(batch_dict, p, U, flags, density);


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
temp_model = glob.os.path.join('lib', path_list[-2] + '_saved_simulate.py')
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

    res = 512

    p =       torch.zeros((1,1,1,res,res), dtype=torch.float).cuda()
    U =       torch.zeros((1,2,1,res,res), dtype=torch.float).cuda()
    flags =   torch.zeros((1,1,1,res,res), dtype=torch.float).cuda()
    density = torch.zeros((1,1,1,res,res), dtype=torch.float).cuda()

    fluid.emptyDomain(flags)
    batch_dict = {}
    batch_dict['p'] = p
    batch_dict['U'] = U
    batch_dict['flags'] = flags
    batch_dict['density'] = density

    density_val = 1
    rad = 0.2
    plume_scale = 1.0 * res/128

    createPlumeBCs(batch_dict, density_val, plume_scale, rad)
    max_iter = 10000
    outIter = 40
    it = 0

    my_map = cm.jet

    fig = plt.figure(figsize=(10,6))
    gs = gridspec.GridSpec(1,2)
    fig.show()
    fig.canvas.draw()
    ax_div = fig.add_subplot(gs[0], frameon=False, aspect=1)
    ax_rho = fig.add_subplot(gs[1], frameon=False, aspect=1)

    while (it < max_iter):
        simulate(batch_dict, res, net, 'convnet')
        if (it% 20 == 0):
            print("It = " + str(it))
            #plotField(batch_dict, 500, 'Hello.png')
            tensor_div = fluid.velocityDivergence(batch_dict['U'], batch_dict['flags'])
            tensor_rho = batch_dict['density']
            img_div = torch.squeeze(tensor_div).cpu().data.numpy()
            img_rho = torch.squeeze(tensor_rho).cpu().data.numpy()

            ax_div.imshow(img_div, cmap=my_map, origin='lower', interpolation='none')
            ax_rho.imshow(img_rho, cmap=my_map, origin='lower', interpolation='none')
            fig.canvas.draw()

        it += 1

finally:
    # Delete model_saved.py
    print()
    print('Deleting' + temp_model)
    glob.os.remove(temp_model)

