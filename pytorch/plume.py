import glob
import argparse
import yaml

import torch
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

simConf['modelDir'] = arguments.modelDir or simConf['modelDir']
assert (glob.os.path.exists(simConf['modelDir'])), 'Directory ' + str(simConf['modelDir']) + ' does not exists'
simConf['modelFilename'] = arguments.modelFilename or simConf['modelFilename']
simConf['modelDirname'] = simConf['modelDir'] + '/' + simConf['modelFilename']
resume = False # For training, at inference set always to false


print('Active CUDA Device: GPU', torch.cuda.current_device())
print()
path = simConf['modelDir']
path_list = path.split(glob.os.sep)
saved_model_name = glob.os.path.join('/', *path_list, path_list[-1] + '_saved.py')
temp_model = glob.os.path.join('lib', path_list[-1] + '_saved_simulate.py')
copyfile(saved_model_name, temp_model)

assert glob.os.path.isfile(temp_model), temp_model  + ' does not exits!'
spec = importlib.util.spec_from_file_location('model_saved', temp_model)
model_saved = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_saved)

try:
    mconf = {}

    mcpath = glob.os.path.join(simConf['modelDir'], simConf['modelFilename'] + '_mconf.pth')
    assert glob.os.path.isfile(mcpath), mcpath  + ' does not exits!'
    mconf.update(torch.load(mcpath))

    print('==> overwriting mconf with user-defined simulation parameters')
    # Overwrite mconf values with user-defined simulation values.
    mconf.update(simConf)

    print('==> loading model')
    mpath = glob.os.path.join(simConf['modelDir'], simConf['modelFilename'] + '_lastEpoch_best.pth')
    assert glob.os.path.isfile(mpath), mpath  + ' does not exits!'
    state = torch.load(mpath)

    print('Data loading: done')

    #********************************** Create the model ***************************
    with torch.no_grad():

        cuda = torch.device('cuda')

        net = model_saved.FluidNet(mconf, dropout=False)
        if torch.cuda.is_available():
            net = net.cuda()

        net.load_state_dict(state['state_dict'])

        #*********************** Simulation parameters **************************

        resX = simConf['resX']
        resY = simConf['resY']

        p =       torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()
        U =       torch.zeros((1,2,1,resY,resX), dtype=torch.float).cuda()
        flags =   torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()
        density = torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()

        fluid.emptyDomain(flags)
        batch_dict = {}
        batch_dict['p'] = p
        batch_dict['U'] = U
        batch_dict['flags'] = flags
        batch_dict['density'] = density

        real_time = simConf['realTimePlot']
        save_vtk = simConf['saveVTK']
        method = simConf['simMethod']
        it = 0

        max_iter = simConf['maxIter']
        outIter = simConf['statIter']

        rho1 = simConf['injectionDensity']
        rad = simConf['sourceRadius']
        plume_scale = simConf['injectionVelocity']

        #**************************** Initial conditions ***************************

        fluid.createPlumeBCs(batch_dict, rho1, plume_scale, rad)
        #XXX: Create Box2D and Cylinders from YAML config file
        # Uncomment to create Cylinder or Box2D obstacles
        #fluid.createCylinder(batch_dict, centerX=0.5*resX,
        #                                 centerY=0.5*resY,
        #                                 radius=50)
        #fluid.createBox2D(batch_dict, x0=0.5*resX, x1=0.5*resX,
        #                              y0=0.7*resY, y1=0.7*resY)

        # If restarting, overwrite all fields with checkpoint.
        if restart_sim:
            # Check if restart file exists in folder
            assert glob.os.path.isfile(restart_state_file), 'Restart file does not exists.'
            restart_dict = torch.load(restart_state_file)
            batch_dict = restart_dict['batch_dict']
            it = restart_dict['it']
            print('Restarting from checkpoint at it = ' + str(it))

        # Create YAML file in output folder
        with open(restart_config_file, 'w') as outfile:
                yaml.dump(simConf, outfile)

        # Print options for debug
        torch.set_printoptions(precision=1, edgeitems = 5)

        # Parameters for matplotlib draw
        my_map = cm.jet
        my_map.set_bad('gray')

        skip = 10
        scale = 20
        scale_units = 'xy'
        angles = 'xy'
        headwidth = 0.8#2.5
        headlength = 5#2

        minY = 0
        maxY = resY
        maxY_win = resY
        minX = 0
        maxX = resX
        maxX_win = resX
        X, Y = np.linspace(0, resX-1, num=resX),\
                np.linspace(0, resY-1, num=resY)

        tensor_vel = batch_dict['U'].clone()
        u1 = (torch.zeros_like(torch.squeeze(tensor_vel[:,0]))).cpu().data.numpy()
        v1 = (torch.zeros_like(torch.squeeze(tensor_vel[:,0]))).cpu().data.numpy()

        # Initialize figure
        if real_time:
            fig = plt.figure(figsize=(20,20))
            gs = gridspec.GridSpec(2,3,
                 wspace=0.5, hspace=0.2)
            fig.show()
            ax_rho = fig.add_subplot(gs[0,0], frameon=False, aspect=1)
            cax_rho = make_axes_locatable(ax_rho).append_axes("right", size="5%", pad="2%")
            ax_velx = fig.add_subplot(gs[0,1], frameon=False, aspect=1)
            cax_velx = make_axes_locatable(ax_velx).append_axes("right", size="5%", pad="2%")
            ax_vely = fig.add_subplot(gs[0,2], frameon=False, aspect=1)
            cax_vely = make_axes_locatable(ax_vely).append_axes("right", size="5%", pad="2%")
            ax_p = fig.add_subplot(gs[1,0], frameon=False, aspect=1)
            cax_p = make_axes_locatable(ax_p).append_axes("right", size="5%", pad="2%")
            ax_div = fig.add_subplot(gs[1,1], frameon=False, aspect=1)
            cax_div = make_axes_locatable(ax_div).append_axes("right", size="5%", pad="2%")
            qx = ax_rho.quiver(X[:maxX_win:skip], Y[:maxY_win:skip],
                u1[minY:maxY:skip,minX:maxX:skip],
                v1[minY:maxY:skip,minX:maxX:skip],
                scale_units = 'height',
                scale=scale,
                #headwidth=headwidth, headlength=headlength,
                color='black')

        # Main loop
        while (it < max_iter):
            #if it < 50:
            #    method = 'jacobi'
            #else:
            method = mconf['simMethod']
            lib.simulate(mconf, batch_dict, net, method)
            if (it% outIter == 0):
                print("It = " + str(it))
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

                if real_time:
                    cax_rho.clear()
                    cax_velx.clear()
                    cax_vely.clear()
                    cax_p.clear()
                    cax_div.clear()
                    fig.suptitle("it = " + str(it), fontsize=16)
                    im0 = ax_rho.imshow(rho[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    ax_rho.set_title('Density')
                    fig.colorbar(im0, cax=cax_rho, format='%.0e')
                    qx.set_UVC(img_velx[minY:maxY:skip,minX:maxX:skip],
                           img_vely[minY:maxY:skip,minX:maxX:skip])

                    im1 = ax_velx.imshow(img_velx[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    ax_velx.set_title('x-velocity')
                    fig.colorbar(im1, cax=cax_velx, format='%.0e')
                    im2 = ax_vely.imshow(img_vely[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    ax_vely.set_title('y-velocity')
                    fig.colorbar(im2, cax=cax_vely, format='%.0e')
                    im3 = ax_p.imshow(p[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    ax_p.set_title('pressure')
                    fig.colorbar(im3, cax=cax_p, format='%.0e')
                    im4 = ax_div.imshow(div[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    ax_div.set_title('divergence')
                    fig.colorbar(im4, cax=cax_div, format='%.0e')

                    fig.canvas.draw()
                    filename = folder + '/output_{0:05}.png'.format(it)
                    fig.savefig(filename)

                if save_vtk:
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
                    density = batch_dict['density'].clone()
                    pressure = batch_dict['p'].clone()
                    b = 1
                    w = pressure.size(4)
                    h = pressure.size(3)
                    d = pressure.size(2)

                    rho = density.narrow(4, 1, w-2).narrow(3, 1, h-2)
                    rho = rho.clone().expand(b, 2, d, h-2, w-2)
                    rho_m = rho.clone().expand(b, 2, d, h-2, w-2)
                    rho_m[:,0] = density.narrow(4, 0, w-2).narrow(3, 1, h-2).squeeze(1)
                    rho_m[:,1] = density.narrow(4, 1, w-2).narrow(3, 0, h-2).squeeze(1)
                    gradRho_center = torch.zeros_like(vel)[:,0:2].contiguous()
                    gradRho_faces = rho - rho_m
                    gradRho_center[:,0:2,0,1:(h-1),1:(w-1)]= fluid.getCentered(gradRho_faces)[:,0:2,0]

                    Pijk = pressure.narrow(4, 1, w-2).narrow(3, 1, h-2)
                    Pijk = Pijk.clone().expand(b, 2, d, h-2, w-2)
                    Pijk_m = Pijk.clone().expand(b, 2, d, h-2, w-2)
                    Pijk_m[:,0] = pressure.narrow(4, 0, w-2).narrow(3, 1, h-2).squeeze(1)
                    Pijk_m[:,1] = pressure.narrow(4, 1, w-2).narrow(3, 0, h-2).squeeze(1)
                    gradP_center = torch.zeros_like(vel)[:,0:2].contiguous()
                    gradP_faces = Pijk - Pijk_m
                    gradP_center[:,0:2,0,1:(h-1),1:(w-1)]= fluid.getCentered(gradP_faces)[:,0:2,0]

                    pressure = pressure[0,0]
                    density = density[0,0]

                    velX = vel[0,0].clone()
                    velY = vel[0,1].clone()
                    gradRhoX = gradRho_center[0,0].clone()
                    gradRhoY = gradRho_center[0,1].clone()
                    gradPX = gradP_center[0,0].clone()
                    gradPY = gradP_center[0,1].clone()
                    flags = batch_dict['flags'][0,0].clone()

                    # Change shape form (D,H,W) to (W,H,D)
                    div.transpose_(0,2).contiguous()
                    density.transpose_(0,2).contiguous()
                    pressure.transpose_(0,2).contiguous()
                    velX.transpose_(0,2).contiguous()
                    velY.transpose_(0,2).contiguous()
                    gradRhoX.transpose_(0,2).contiguous()
                    gradRhoY.transpose_(0,2).contiguous()
                    gradPX.transpose_(0,2).contiguous()
                    gradPY.transpose_(0,2).contiguous()
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

                    divergence = np.ascontiguousarray(div_np[minX:maxX,minY:maxY])
                    rho = np.ascontiguousarray(density_np[minX:maxX,minY:maxY])
                    p = np.ascontiguousarray(pressure_masked[minX:maxX,minY:maxY])
                    velx = np.ascontiguousarray(velx_masked[minX:maxX,minY:maxY])
                    vely = np.ascontiguousarray(vely_masked[minX:maxX,minY:maxY])
                    gradRhox = np.ascontiguousarray(gradRhoX.cpu().data.numpy()[minX:maxX,minY:maxY])
                    gradRhoy = np.ascontiguousarray(gradRhoY.cpu().data.numpy()[minX:maxX,minY:maxY])
                    gradPx = np.ascontiguousarray(gradPX.cpu().data.numpy()[minX:maxX,minY:maxY])
                    gradPy = np.ascontiguousarray(gradPY.cpu().data.numpy()[minX:maxX,minY:maxY])
                    filename = folder + '/output_{0:05}'.format(it)
                    vtk.gridToVTK(filename, x, y, z, cellData = {
                        'density': rho,
                        'divergence': divergence,
                        'pressure' : p,
                        'ux' : velx,
                        'uy' : vely,
                        'gradPx' : gradPx,
                        'gradPy' : gradPy,
                        'gradRhox' : gradRhox,
                        'gradRhoy' : gradRhoy
                        })

                restart_dict = {'batch_dict': batch_dict, 'it': it}
                torch.save(restart_dict, restart_state_file)

            # Update iterations
            it += 1

finally:
    # Properly deleting model_saved.py, even when ctrl+C
    print()
    print('Deleting' + temp_model)
    glob.os.remove(temp_model)

