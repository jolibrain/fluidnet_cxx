import sys
import argparse

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data

import matplotlib
import matplotlib.colors as colors
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
    batch_dict['U'] = U;

    density.mul_(batch_dict['densityBCInvMask']);
    density.add_(batch_dict['densityBC']);
    batch_dict['density'] = density;

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
        setConstVals(batch_dict, p, U, flags, density);

        # Set the constant domain values.
        if (sim_method != 'convnet'):
            fluid.setWallBcs(U, flags);
        setConstVals(batch_dict, p, U, flags, density);

        if (sim_method == 'convnet'):
            # fprop the model to perform the pressure projection and velocity calculation.
            # Set wall BCs is performed inside the model, before and after the projection.
            # No need to call it again.
            data = torch.cat((p, U, flags), 1)
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

def plotField_den(tensor, res, name):

    tensor_to_p = torch.norm(tensor,2,dim=1,keepdim=True).squeeze(0).squeeze(0).squeeze(0)
    img_tensor = tensor_to_p.cpu().data.numpy()

    my_map = cm.jet
    fig = plt.figure()
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False, aspect=1)
    ax.imshow(img_tensor, cmap=my_map, interpolation='none')

    plt.show(block=False)
    plt.pause(2)
    plt.close()

#********************************** Define Config ******************************

#TODO: allow to overwrite params from the command line by parsing.

conf = defaultConf.copy()
conf['modelDir'] = sys.argv[1]
print(sys.argv[1])
conf['modelDirname'] = conf['modelDir'] + '/' + conf['modelFilename']
resume = False
data_dir = conf['dataDir']

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
    conf['dataDir'] = data_dir

    test_loader = torch.utils.data.DataLoader(te, batch_size=1, \
            num_workers=0, shuffle=False, pin_memory=True)

    print('==> loading checkpoint')
    mpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_lastEpoch_best.pth')
    assert glob.os.path.isfile(mpath), mpath  + ' does not exits!'
    state = torch.load(mpath)

    print('Data loading: done')

  #********************************* Plot functions *************************************

    def plotField(out, tar, flags, loss, mconf, filename):
        div = 0
        exit = True
        target = tar.clone()
        output_p = out[0].clone()
        output_U = out[1].clone()
        p_out = output_p
        p_tar = target[:,0].unsqueeze(1)
        U_norm_out = torch.zeros_like(p_out)
        U_norm_tar = torch.zeros_like(p_tar)

        Ux_out = output_U[:,0].unsqueeze(1)
        Ux_tar = target[:,1].unsqueeze(1)
        #torch.norm(output[1], p=2, dim=1, keepdim=True, out=U_norm_out)
        Uy_out = output_U[:,1].unsqueeze(1)
        Uy_tar = target[:,2].unsqueeze(1)
        torch.norm(output_U, p=2, dim=1, keepdim=True, out=U_norm_out)
        torch.norm(target[:,1:3], p=2, dim=1, keepdim=True, out=U_norm_tar)

        div = out[2].clone()

        err_p = (p_out - p_tar)**2
        err_Ux = (Ux_out - Ux_tar)**2
        err_Uy = (Uy_out - Uy_tar)**2
        err_div = (div)**2

        max_val_p = np.maximum(torch.max(p_tar).cpu().data.numpy(), \
                             torch.max(p_out).cpu().data.numpy() )
        min_val_p = np.minimum(torch.min(p_tar).cpu().data.numpy(), \
                             torch.min(p_out).cpu().data.numpy())
        max_val_Ux = np.maximum(torch.max(Ux_out).cpu().data.numpy(), \
                             torch.max(Ux_tar).cpu().data.numpy() )
        min_val_Ux = np.minimum(torch.min(Ux_out).cpu().data.numpy(), \
                             torch.min(Ux_tar).cpu().data.numpy())
        max_val_Uy = np.maximum(torch.max(Uy_out).cpu().data.numpy(), \
                             torch.max(Uy_tar).cpu().data.numpy() )
        min_val_Uy = np.minimum(torch.min(Uy_out).cpu().data.numpy(), \
                             torch.min(Uy_tar).cpu().data.numpy())
        max_val_Unorm = np.maximum(torch.max(U_norm_out).cpu().data.numpy(), \
                             torch.max(U_norm_tar).cpu().data.numpy() )
        min_val_Unorm = np.minimum(torch.min(U_norm_out).cpu().data.numpy(), \
                             torch.min(U_norm_tar).cpu().data.numpy() )
        max_err_p = torch.max(err_p).cpu().data.numpy()
        max_err_Ux = torch.max(err_Ux).cpu().data.numpy()
        max_err_Uy = torch.max(err_Uy).cpu().data.numpy()

        max_div = torch.max(div).cpu().data.numpy()
        min_div = torch.min(div).cpu().data.numpy()

        mask = flags.eq(2)
        p_tar.masked_fill_(mask, 100)
        p_out.masked_fill_(mask, 100)
        Ux_tar.masked_fill_(mask, 0)
        Ux_out.masked_fill_(mask, 0)
        Uy_tar.masked_fill_(mask, 0)
        Uy_out.masked_fill_(mask, 0)
        U_norm_tar.masked_fill_(mask, 100)
        U_norm_out.masked_fill_(mask, 100)
        div.masked_fill_(mask, 100)

        err_p.masked_fill_(mask, 100)
        err_Ux.masked_fill_(mask, 100)
        err_Uy.masked_fill_(mask, 100)
        err_div.masked_fill_(mask, 100)

        p_tar_np =torch.squeeze(p_tar.cpu()).data.numpy()
        p_out_np =torch.squeeze(p_out.cpu()).data.numpy()
        Ux_tar_np =torch.squeeze(Ux_tar.cpu()).data.numpy()
        Ux_out_np =torch.squeeze(Ux_out.cpu()).data.numpy()
        Uy_tar_np =torch.squeeze(Uy_tar.cpu()).data.numpy()
        Uy_out_np =torch.squeeze(Uy_out.cpu()).data.numpy()
        U_norm_tar_np =torch.squeeze(U_norm_tar.cpu()).data.numpy()
        U_norm_out_np =torch.squeeze(U_norm_tar.cpu()).data.numpy()
        div_np =torch.squeeze(div).cpu().data.numpy()
        err_p_np =torch.squeeze(err_p.cpu()).data.numpy()
        err_Ux_np =torch.squeeze(err_Ux.cpu()).data.numpy()
        err_Uy_np =torch.squeeze(err_Uy.cpu()).data.numpy()
        err_div_np =torch.squeeze(err_div.cpu()).data.numpy()

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

        my_cmap = cm.jet
        my_cmap.set_over('white')
        my_cmap.set_under('white')

        nrow = 3
        ncol = 3
        matplotlib.rc('text')
        fig = plt.figure(figsize=(nrow+1, ncol+1))
        gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1, 1, 1],
                         wspace=0.1, hspace=0.1, top=0.9, bottom=0.01, left=0.1, right=0.9)
        fig.suptitle('FluidNet output for loss = ' + title )

        ax = plt.subplot(gs[0,0])
        ax.set_title('P_target')
        ax.axis('off')
        ax.imshow(p_tar_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_val_p,max_val_p])

        ax = plt.subplot(gs[1,0])
        ax.set_title('P_predicted')
        ax.axis('off')
        ax.imshow(p_out_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_val_p,max_val_p])

        ax = plt.subplot(gs[2,0])
        ax.set_title('error P')
        ax.axis('off')
        ax.imshow(err_p_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[0,max_err_p])

        X, Y = np.meshgrid(np.arange(0, 128, 8), np.arange(0, 128, 8))

        ax = plt.subplot(gs[0,1])
        ax.set_title('|U|_target')
        ax.axis('off')
        X, Y = np.linspace(0, 127, num=128), np.linspace(0, 127, num=128)
        ax.imshow(U_norm_tar_np, cmap=my_cmap, origin='lower',
                interpolation='none', clim=[min_val_Unorm,max_val_Unorm])
        ax.quiver(X[::2], Y[::2],
                Ux_tar_np[::2, ::2], Uy_tar_np[::2, ::2], units='xy'
                , pivot='tail', width=0.3, headwidth=3, scale=0.95, color='black')

        ax = plt.subplot(gs[1,1])
        ax.set_title('|U|_predicted')
        ax.axis('off')
        ax.imshow(U_norm_out_np, cmap=my_cmap, origin='lower',
                interpolation='none', clim=[min_val_Unorm,max_val_Unorm])
        ax.quiver(X[::2], Y[::2],
                Ux_out_np[::2, ::2], Uy_out_np[::2, ::2], units='xy'
                , pivot='tail', width=0.3, headwidth=3, scale=0.95, color='black')

        ax = plt.subplot(gs[2,1])
        ax.set_title('error Ux')
        ax.axis('off')
        ax.imshow(err_Ux_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[0,max_err_Ux])

        ax = plt.subplot(gs[0,2])
        ax.set_title('div at output')
        ax.axis('off')
        ax.imshow(div_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_div,max_div])

        ax = plt.subplot(gs[1,2])
        ax.set_title('div error')
        ax.axis('off')
        ax.imshow(err_div_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[0,max_div**2])

        ax = plt.subplot(gs[2,2])
        ax.set_title('Losses')
        ax.axis('off')
        for y, loss, val in zip(np.linspace(0.2,0.8,5),
                                 ['Total' ,'L2(p)', 'L2(div)', 'L1(p)', 'L1(div)'],
                                 loss):
            ax.text(0.2, y,  ('Loss {:} : ').format(loss) \
                    + ('{:.6f}').format(val), fontsize=12)

        #fig.colorbar(imP, cax=cbar_ax_p, orientation='vertical')
        #cbar_ax_U = fig.add_axes([0.375, 0.45, 0.01, 0.33])
        #fig.colorbar(imU, cax=cbar_ax_U, orientation='vertical')
        #fig.set_size_inches((11, 11), forward=False)
        #fig.savefig(filename)
        plt.show(block=True)


        #plt.figure()
        #plt.title('Ux_target')
        #X, Y = np.linspace(0, 127, num=128), np.linspace(0, 127, num=128)
        #print(X)
        #I = plt.imshow(U_norm_tar_np, cmap=my_cmap2, origin='lower',
        #        interpolation='none', clim=[min_val_Unorm,max_val_Unorm])
        #Q = plt.quiver(X[::2], Y[::2],
        #        Ux_tar_np[::2, ::2], Uy_tar_np[::2, ::2], units='xy'
        #        , pivot='tail', width=0.2, headwidth=2.5, scale=0.95, color='black')
        #plt.show(block=True)


    #********************************** Create the model ***************************


    print('')
    print('----- Model ------')

    # Create model and print layers and params

    net = model_saved.FluidNet(mconf, dropout=False)
    if torch.cuda.is_available():
        net = net.cuda()
    #lib.summary(net, (3,1,128,128))

    net.load_state_dict(state['state_dict'])

    #res = 512

    #p =       torch.zeros((1,1,1,res,res), dtype=torch.float).cuda()
    #U =       torch.zeros((1,2,1,res,res), dtype=torch.float).cuda()
    #flags =   torch.zeros((1,1,1,res,res), dtype=torch.float).cuda()
    #density = torch.zeros((1,1,1,res,res), dtype=torch.float).cuda()

    #fluid.emptyDomain(flags)
    #batch_dict = {}
    #batch_dict['p'] = p
    #batch_dict['U'] = U
    #batch_dict['flags'] = flags
    #batch_dict['density'] = density

    #density_val = 1
    #rad = 0.2
    #plume_scale = 1.0 * res/128

    #createPlumeBCs(batch_dict, density_val, plume_scale, rad)

    from itertools import count

    #batch_print = [100, 1000, 4000, 5000, 8000, 10000, 15000, 20000]
    def val(data, target, it):
        net.eval()
        loss = nn.MSELoss()
        total_val_loss = 0
        p_l2_total_loss = 0
        div_l2_total_loss = 0
        p_l1_total_loss = 0
        div_l1_total_loss = 0

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

        dt = 0.1
        maccormackStrength = 0.6
        with torch.no_grad():
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            U = data[:,1:3].clone()
            flags = data[:,3].unsqueeze(1)
            U = fluid.advectVelocity(dt, U, flags, \
                    method="maccormackFluidNet", \
                    boundary_width=1, maccormack_strength=maccormackStrength);
            data[:,1:3] = U
            out_p, out_U = net(data)
            target_p = target[:,0].unsqueeze(1)
            out_div = fluid.velocityDivergence(\
                    out_U.contiguous(), \
                    data[:,3].unsqueeze(1).contiguous())
            target_div = torch.zeros_like(out_div)

            loss_size = 0
            # Measure loss and save it
            pL2Loss = pL2Lambda *_pL2Loss(out_p, target_p)
            divL2Loss = divL2Lambda *_divL2Loss(out_div, target_div)
            pL1Loss =  pL1Lambda *_pL1Loss(out_p, target_p)
            divL1Loss = divL1Lambda *_divL1Loss(out_div, target_div)

            loss_size =  pL2Loss + divL2Loss + pL1Loss + divL1Loss

            # Just 1 batch
            p_l2_total_loss += pL2Loss.data.item()
            div_l2_total_loss += divL2Loss.data.item()
            p_l1_total_loss += pL1Loss.data.item()
            div_l1_total_loss += divL1Loss.data.item()
            total_val_loss += loss_size.item()

            flags = data[:,3].unsqueeze(1).contiguous()
            out_list = [out_p, out_U, out_div]
            loss = [total_val_loss, p_l2_total_loss, div_l2_total_loss, \
                    p_l1_total_loss, div_l1_total_loss]
            filename = 'figures/fig_' + str(it) + '.png'
            #if (it % 8 == 0):
            #    plotField(out_list, target, flags, loss, mconf, filename)
            data[:,1:3] = out_U.clone()

        return data, target, loss

    print('Dataset in ' + str(conf['dataDir']))
    print('Plotting results at epoch ' + str(state['epoch']) )
    it = 0
    it_8dt = 0
    max_iter = 8000
    data, _ = te.__getitem__(0)
    data.unsqueeze_(0)
    data_loss_plot = np.empty((0,2))
    fig = plt.gcf()
    fig.show()
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Divergence')
    fig.canvas.draw()

    while (it < max_iter):
        print('It = ' + str(it))

        if (it % 8 == 0):
            _ , target = te.__getitem__(it_8dt)
            target.unsqueeze_(0)
            it_8dt += 1

        data_temp, target_temp, loss = val(data, target, it)
        data_loss_plot = np.append(data_loss_plot, [[it, loss[0]]],
                axis = 0)
        data = data_temp.clone()
        target = target_temp.clone()
        if (it % 128 == 0):
            plt.plot(data_loss_plot[:,0], data_loss_plot[:,1], 'r')
            fig.canvas.draw()
            data_loss_plot = [data_loss_plot[-1,:]]
        it +=1

    plt.plot(data_loss_plot[:,0], data_loss_plot[:,1])
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Divergence')
    plt.show(block=True)

finally:
    # Delete model_saved.py
    print()
    print('Deleting' + temp_model)
    glob.os.remove(temp_model)

