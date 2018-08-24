import sys
import os
import argparse

import torch
import torch.autograd
import math
import time

import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

import numpy as np
import numpy.ma as ma

import pyevtk.hl as vtk

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

def createRayleighTaylorBCs(batch_dict, mconf, rho1, rho2):

    cuda = torch.device('cuda')
    # batch_dict at input: {p, UDiv, flags, density}
    assert len(batch_dict) == 4, "Batch must contain 4 tensors (p, UDiv, flags, density)"
    UDiv = batch_dict['U']
    flags = batch_dict['flags']

    resX = UDiv.size(4)
    resY = UDiv.size(3)

    # Here, we just impose initial conditions.
    # Upper layer rho2, vel = 0
    # Lower layer rho1, vel = 0

    X = torch.arange(1, resX-1, device=cuda).view(resX-2).expand((1,resY-2,resX-2))
    Y = torch.arange(1, resY-1, device=cuda).view(resY-2, 1).expand((1,resY-2,resX-2))
    coord = torch.cat((X,Y), dim=0).unsqueeze(0).unsqueeze(2)

    # Atwood number
    #A = ((1+rho2) - (1+rho1)) / ((1+rho2) + (1+rho1))
    #print('Atwood number : ' + str(A))
    #density = ((1-A) * torch.tanh(100*(coord[:,1]/resY - (0.85 - \
    #                0.05*torch.cos(math.pi*(coord[:,0]/resX)))))).unsqueeze(1)
    A = (rho2 - rho1) / (rho2 + rho1)
    print('Atwood number : ' + str(A))
    density[:,:,:,1:(resY-1),1:(resX-1)] = 0.5*(rho2+rho1 + (rho2-rho1)*torch.tanh(100*(coord[:,1]/resY - (0.5 - \
                    0.05*torch.cos(math.pi*(coord[:,0]/resX)))))).unsqueeze(1)

    #TopWall = (Y > (flags.size(3) - 2)).__and__(X > 0).__and__(X < flags.size(4) - 1)
    #flags.masked_fill_(TopWall, fluid.CellType.TypeEmpty)
    print('density')
    print(density)
    print('flags')
    print(flags)
    #upper_mask = ((Y/resY) >= (0.5 + 0.01 * \
    #    torch.cos(math.pi*(X/resX))))
    #lower_mask = upper_mask.eq(0)
    #density.masked_fill_(upper_mask, rho2)
    #density.masked_fill_(lower_mask, rho1)
    batch_dict['density'] = density
    batch_dict['flags'] = flags
    #Initialize pressure to hydrostatic:
    # p = 0 on top
    # p = P0 - rho*g*y
    #gravity = torch.FloatTensor(3).fill_(0).cuda()
    #buoyancyScale = mconf['buoyancyScale']
    #gravity[0] = mconf['gravityVec']['x']
    #gravity[1] = mconf['gravityVec']['y']
    #gravity[2] = mconf['gravityVec']['z']
    #gravity.mul_(-buoyancyScale)

    #pressure = coord[:,1].unsqueeze(1)*gravity[1]
    #fluid.velocityUpdate(0.1, pressure, UDiv, flags)
    #batch_dict['U'] = UDiv
    #print(pressure)
    #print(UDiv[:,1])
    # batch_dict at output = {p, UDiv, flags, density}

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
    with torch.no_grad():

        print('')
        print('----- Model ------')

        # Create model and print layers and params
        cuda = torch.device('cuda')

        resX = 100#6000
        resY = 200#800

        p =       torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()
        U =       torch.zeros((1,2,1,resY,resX), dtype=torch.float).cuda()
        flags =   torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()
        density = torch.ones((1,1,1,resY,resX), dtype=torch.float).cuda()

        fluid.emptyDomain(flags)
        batch_dict = {}
        batch_dict['p'] = p
        batch_dict['U'] = U
        batch_dict['flags'] = flags
        batch_dict['temperature'] = density

        restart = False
        real_time = True
        folder = 'data2/rayleigh_taylor_atwood/'
        filename_restart = folder + 'restart.pth'
        method = 'jacobi'
        it = 0
        if restart:
            restart_dict = torch.load(filename_restart)
            batch_dict = restart_dict['batch_dict']
            it = restart_dict['it']
            print('Restarting at it = ' + str(it))

        mconf['maccormackStrength'] = 0.2
        mconf['buoyancyTemperatureScale'] = 0.01#0.2#0.1#9.81/resY
        mconf['kt'] = 0.01#0.2#0.1#9.81/resY
        mconf['buoyancyScale'] = 0.0#0.2#0.1#9.81/resY
        mconf['gravityScale'] = 0.0#0.1#2.0/resY
        mconf['viscosity'] = 0.0
        mconf['dt'] = 1.0
        mconf['jacobiIter'] = 50

        mconf['gravityVec'] = {'x': 0, 'y': -1, 'z': 0}
        max_iter = 10000
        outIter = 10

        net = model_saved.FluidNet(mconf, dropout=False)
        if torch.cuda.is_available():
            net = net.cuda()
        net.load_state_dict(state['state_dict'])

        my_map = cm.jet
        my_map.set_bad('gray')

        skip = 20
        scale = 0.6
        scale_units = 'xy'
        angles = 'xy'
        headwidth = 0.8#2.5
        headlength = 5#2

        torch.set_printoptions(precision=3, edgeitems = 5)

        minY = 0
        maxY = resY
        maxY_win = resY
        minX = 0
        maxX = resX
        maxX_win = resX
        X, Y = np.linspace(0, resX-1, num=resX),\
                np.linspace(0, resY-1, num=resY)

        createRayleighTaylorBCs(batch_dict, mconf, rho1=1.1, rho2=0.9)
        tensor_vel = batch_dict['U'].clone()
        u1 = (torch.zeros_like(torch.squeeze(tensor_vel[:,0]))).cpu().data.numpy()
        v1 = (torch.zeros_like(torch.squeeze(tensor_vel[:,0]))).cpu().data.numpy()

        #rho_init = torch.squeeze(batch_dict['density'].clone()).cpu().data.numpy()
        #plt.imshow(rho_init)
        #plt.show(block=True)
        if real_time:
            fig = plt.figure(figsize=(20,10))
            gs = gridspec.GridSpec(1,5,
                 wspace=0.5, hspace=0.2)

            fig.show()
            fig.canvas.draw()
            ax_rho = fig.add_subplot(gs[0], frameon=False, aspect=1)
            cax_rho = make_axes_locatable(ax_rho).append_axes("right", size="5%", pad="2%")
            ax_velx = fig.add_subplot(gs[1], frameon=False, aspect=1)
            cax_velx = make_axes_locatable(ax_velx).append_axes("right", size="5%", pad="2%")
            ax_vely = fig.add_subplot(gs[2], frameon=False, aspect=1)
            cax_vely = make_axes_locatable(ax_vely).append_axes("right", size="5%", pad="2%")
            ax_p = fig.add_subplot(gs[3], frameon=False, aspect=1)
            cax_p = make_axes_locatable(ax_p).append_axes("right", size="5%", pad="2%")
            ax_div = fig.add_subplot(gs[4], frameon=False, aspect=1)
            cax_div = make_axes_locatable(ax_div).append_axes("right", size="5%", pad="2%")
            qx = ax_rho.quiver(X[:maxX_win:skip], Y[:maxY_win:skip],
                u1[minY:maxY:skip,minX:maxX:skip],
                v1[minY:maxY:skip,minX:maxX:skip],
                scale_units = 'height',
                scale=scale,
                #headwidth=headwidth, headlength=headlength,
                color='black')
        #ax_vely = fig.add_subplot(gs[1], frameon=False, aspect=1)

        while (it < max_iter):
            lib.simulate(conf, mconf, batch_dict, net, method)
            if (it% outIter == 0):
                print("It = " + str(it))
                #print('pressure')
                #print(batch_dict['p'])
                #print('vel-y')
                #print(batch_dict['U'][:,1])
                #print()
                #plotField(batch_dict, 500, 'Hello.png')
                #tensor_div = torch.zeros_like(flags)
                tensor_div = fluid.velocityDivergence(batch_dict['U'].clone(),
                        batch_dict['flags'].clone())
                pressure = batch_dict['p'].clone()
                tensor_vel = fluid.getCentered(batch_dict['U'].clone())
                density = batch_dict['density'].clone()
                div = torch.squeeze(tensor_div).cpu().data.numpy()
                np_mask = torch.squeeze(flags.eq(2)).cpu().data.numpy().astype(float)
                rho = torch.squeeze(density).cpu().data.numpy()
                p = torch.squeeze(pressure).cpu().data.numpy()
                img_norm_vel = torch.squeeze(torch.norm(tensor_vel,
                    dim=1, keepdim=True)).cpu().data.numpy()
                img_velx = torch.squeeze(tensor_vel[:,0]).cpu().data.numpy()
                img_vely = torch.squeeze(tensor_vel[:,1]).cpu().data.numpy()
                img_vel_norm = torch.squeeze( \
                        torch.norm(tensor_vel, dim=1, keepdim=True)).cpu().data.numpy()

                img_velx_masked = ma.array(img_velx, mask=np_mask)
                img_vely_masked = ma.array(img_vely, mask=np_mask)
                img_vel_norm_masked = ma.array(img_vel_norm, mask=np_mask)
                ma.set_fill_value(img_velx_masked, np.nan)
                ma.set_fill_value(img_vely_masked, np.nan)
                ma.set_fill_value(img_vel_norm_masked, np.nan)
                img_velx_masked = img_velx_masked.filled()
                img_vely_masked = img_vely_masked.filled()
                img_vel_norm_masked = img_vel_norm_masked.filled()

                #img_zeros_x = np.zeros_like(img_velx)
                #img_zeros_y = np.zeros_like(img_vely)
                #ax_div.imshow(img_div, cmap=my_map, origin='lower',
                #        interpolation='none')
                if real_time:
                    cax_rho.clear()
                    cax_velx.clear()
                    cax_vely.clear()
                    cax_p.clear()
                    cax_div.clear()
                    im0 = ax_rho.imshow(rho[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    fig.colorbar(im0, cax=cax_rho, format='%.0e')
                    qx.set_UVC(img_velx[minY:maxY:skip,minX:maxX:skip],
                           img_vely[minY:maxY:skip,minX:maxX:skip])

                    im1 = ax_velx.imshow(img_velx[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    fig.colorbar(im1, cax=cax_velx, format='%.0e')
                    im2 = ax_vely.imshow(img_vely[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    fig.colorbar(im2, cax=cax_vely, format='%.0e')
                    im3 = ax_p.imshow(p[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    fig.colorbar(im3, cax=cax_p, format='%.0e')
                    im4 = ax_div.imshow(div[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    fig.colorbar(im4, cax=cax_div, format='%.0e')

                    fig.canvas.draw()
                    #scale_units=scale_units,
                    #angles=angles,
                    #headwidth=headwidth, headlength=headlength,
                    #scale=scale,
                    #color='black')

                else:
                    px, py = 1580, 950
                    dpi = 100
                    figx = px / dpi
                    figy = py / dpi

                    nx = maxX_win
                    ny = maxY_win
                    nz = 1
                    ncells = nx*ny*nz

                    ratio = nx/ny
                    lx, ly = ratio, 1.0
                    dx, dy = lx/nx, ly/ny

                    # Coordinates
                    x = np.arange(0, lx + 0.1*dx, dx, dtype='float32')
                    y = np.arange(0, ly + 0.1*dy, dy, dtype='float32')
                    z = np.zeros(1, dtype='float32')

                    # Variables
                    div = fluid.velocityDivergence(\
                        batch_dict['U'].clone(), \
                        batch_dict['flags'].clone())[0,0]
                    vel = fluid.getCentered(batch_dict['U'].clone())
                    density = batch_dict['density'][0,0].clone()
                    pressure = batch_dict['p'][0,0].clone()
                    velX = vel[0,0].clone()
                    velY = vel[0,1].clone()
                    flags = batch_dict['flags'][0,0].clone()

                    # Change shape form (D,H,W) to (W,H,D)
                    div.transpose_(0,2).contiguous()
                    density.transpose_(0,2).contiguous()
                    pressure.transpose_(0,2).contiguous()
                    velX.transpose_(0,2).contiguous()
                    velY.transpose_(0,2).contiguous()
                    flags.transpose_(0,2).contiguous()

                    div_np = div.cpu().data.numpy()
                    density_np = density.cpu().data.numpy()
                    pressure_np = pressure.cpu().data.numpy()
                    velX_np = velX.cpu().data.numpy()
                    velY_np = velY.cpu().data.numpy()
                    np_mask = flags.eq(2).cpu().data.numpy().astype(float)
                    pressure_masked = ma.array(pressure_np, mask=np_mask)
                    velx_masked = ma.array(velX_np, mask=np_mask)
                    vely_masked = ma.array(velY_np, mask=np_mask)
                    ma.set_fill_value(pressure_masked, np.nan)
                    ma.set_fill_value(velx_masked, np.nan)
                    ma.set_fill_value(vely_masked, np.nan)
                    pressure_masked = pressure_masked.filled()
                    velx_masked = velx_masked.filled()
                    vely_masked = vely_masked.filled()

                    divergence = np.ascontiguousarray(div[minX:maxX,minY:maxY])
                    rho = np.ascontiguousarray(density_np[minX:maxX,minY:maxY])
                    p = np.ascontiguousarray(pressure_masked[minX:maxX,minY:maxY])
                    velx = np.ascontiguousarray(velx_masked[minX:maxX,minY:maxY])
                    vely = np.ascontiguousarray(vely_masked[minX:maxX,minY:maxY])
                    filename = './' + folder + 'output_{0:05}'.format(it)
                    vtk.gridToVTK(filename, x, y, z, cellData = {
                        'density': rho,
                        'divergence': divergence,
                        'pressure' : p,
                        'ux' : velx,
                        'uy' : vely
                        })

                    restart_dict = {'batch_dict': batch_dict, 'it': it}
                    torch.save(restart_dict, filename_restart)
                    #fig = plt.figure(figsize=(figx, figy), dpi=dpi)
                    #gs = gridspec.GridSpec(1,1)
                    #t = mconf['dt'] * it
                    #fig.suptitle(('t = {:7.2f} s').format(t),
                    #        fontsize=14,
                    #        weight='bold')
                    #ax_vel = fig.add_subplot(gs[0], frameon=False, aspect=1)
                    #ax_vel.imshow(img_vel_norm_masked[minY:maxY,minX:maxX],
                    #    cmap=my_map,
                    #    origin='lower',
                    #    interpolation='none')
                    #qx = ax_vel.quiver(X[:maxX_win:skip], Y[:maxY_win:skip],
                    #    img_velx[minY:maxY:skip,minX:maxX:skip],
                    #    img_vely[minY:maxY:skip,minX:maxX:skip],
                    #    #headwidth=headwidth, headlength=headlength,
                    #    color='black')
                    #filename = 'data2/simulation/vel_it_{0:05}.png'.format(it)
                    #fig.savefig(filename)
                    #plt.close()
                #ax_vely.imshow(img_vely_masked[minY:maxY,:maxX],
                #        cmap=my_map,
                #        origin='lower',
                #        interpolation='none')
                #ax_vely.quiver(X[minX:maxX,:maxY], Y[minX:maxX,:maxY],
                #    img_zeros_x[minX:maxX,:maxY],
                #    img_zeros_y[minX:maxX,:maxY],
                #    scale_units=scale_units,
                #    angles=angles,
                #    headwidth=headwidth, headlength=headlength, scale=scale,
                #    color='black')

            it += 1

finally:
    # Delete model_saved.py
    print()
    print('Deleting' + temp_model)
    glob.os.remove(temp_model)

