import glob
import sys
import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import random
import glob
from shutil import copyfile
import importlib.util

import lib
import lib.fluid as fluid

# Parse arguments
parser = argparse.ArgumentParser(description='Training script.', \
        formatter_class= lib.SmartFormatter)
parser.add_argument('--trainingConf',
        default='trainConfig.yaml',
        help='R|Training yaml config file.\n'
        '  Default: trainConfig.yaml')
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

# Preprocessing dataset message (will exit after preproc)
if (conf['preprocOriginalFluidNetDataOnly']):
    print('Running preprocessing only')
    resume = False

print('Active CUDA Device: GPU', torch.cuda.current_device())
cuda0 = torch.device('cuda:0')

# Define training and test datasets
tr = lib.FluidNetDataset(conf, 'tr', save_dt=4, resume=resume)
te = lib.FluidNetDataset(conf, 'te', save_dt=4, resume=resume)

if (conf['preprocOriginalFluidNetDataOnly']):
    sys.exit()

# We create two conf dicts, general params and model params.
conf, mconf = tr.createConfDict()

# Separate some variables from conf dict. When resuming training, this ones will
# overwrite saved conf (where model is saved).
# User can modify them in YAML config file or in command line.
num_workers = arguments.numWorkers or conf['numWorkers']
batch_size = arguments.bsz or conf['batchSize']
max_epochs = arguments.maxEpochs or conf['maxEpochs']
print_training = output_mode == 'show' or output_mode == 'save'
save_or_show = output_mode == 'save'
lr = arguments.lr or mconf['lr']

#******************************** Restarting training ***************************

if resume:
    print()
    print('            RESTARTING TRAINING            ')
    print()
    print('==> loading checkpoint')
    mpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_lastEpoch_best.pth')
    assert glob.os.path.isfile(mpath), mpath  + ' does not exits!'
    state = torch.load(mpath)

    print('==> overwriting conf and file_mconf')
    cpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_conf.pth')
    mcpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_mconf.pth')
    assert glob.os.path.isfile(mpath), cpath  + ' does not exits!'
    assert glob.os.path.isfile(mpath), mcpath  + ' does not exits!'
    conf.update(torch.load(cpath))
    mconf.update(torch.load(mcpath))

    print('==> copying and loading corresponding model module')
    path = conf['modelDir']
    path_list = path.split(glob.os.sep)
    saved_model_name = glob.os.path.join('/', *path_list, path_list[-1] + '_saved.py')
    temp_model = glob.os.path.join('lib', path_list[-1] + '_saved_resume.py')
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
            num_workers=num_workers, shuffle=shuffle_training, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(te, batch_size=batch_size, \
            num_workers=num_workers, shuffle=False, pin_memory=True)

    #********************************** Create the model ***************************
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_uniform_(m.weight)

    print('')
    print('------------------- Model ----------------------------')

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

    #********************** Define the optimizer ***********************

    print('==> defining optimizer')

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    if resume:
        optimizer.load_state_dict(state['optimizer'])

    for param_group in optimizer.param_groups:
        print('lr of optimizer')
        print(param_group['lr'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',factor = 0.6, patience = 10, verbose = True, threshold = 3e-4, threshold_mode = 'rel')

    #************************ Training and Validation*******************
    # Test set Scenes idx to plot during validation
    list_to_plot = [64, 2560, 5120, 11392]

    def run_epoch(epoch, loader, training=True):
        if training:
            #set model to train
            net.train()
        else:
            #otherwise, set it to eval.
            net.eval()

        #initialise loss scores
        total_loss = 0
        p_l2_total_loss = 0
        div_l2_total_loss = 0
        p_l1_total_loss = 0
        div_l1_total_loss = 0
        div_lt_total_loss = 0

        n_batches = 0 # Number of processed batches

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

        #loop through data, sorted into batches
        for batch_idx, (data, target) in enumerate (loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            if training:
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

            is3D = data.size(1) == 6
            assert (is3D and data.size(1) == 6) or (not is3D and data.size(1) == 5), "Data must have \
                    5 input chan for 2D, 6 input chan for 3D"

            # Run the model forward
            flags = data[:,3].unsqueeze(1).contiguous()
            out_p, out_U = net(data)

            # Calculate targets
            target_p = target[:,0].unsqueeze(1)
            out_div = fluid.velocityDivergence(out_U.contiguous(), flags)
            target_div = torch.zeros_like(out_div)

            # Measure loss and save it
            pL2Loss = pL2Lambda *_pL2Loss(out_p, target_p)
            divL2Loss = divL2Lambda *_divL2Loss(out_div, target_div)
            pL1Loss =  pL1Lambda *_pL1Loss(out_p, target_p)
            divL1Loss = divL1Lambda *_divL1Loss(out_div, target_div)

            loss_size =  pL2Loss + divL2Loss + pL1Loss + divL1Loss

            # We calculate the divergence of a future frame.
            if (divLTLambda > 0):
                # Check if additional buoyancy or gravity is added to future frames.
                # Adding Buoyancy means adding a source term in the momentum equation, of
                # the type f = delta_rho*g and rho = rho_0 + delta_rho (constant term + fluctuation)
                # with rho' << rho_0
                # Adding gravity: source of the type f = rho_0*g
                # Source term is a vector (direction and magnitude).

                oldBuoyancyScale = mconf['buoyancyScale']
                # rand(1) is an uniform dist on the interval [0,1)
                if torch.rand(1)[0] < mconf['trainBuoyancyProb']:
                    # Add buoyancy to this batch (only in the long term frames)
                    var = torch.tensor([1.], device=cuda0)
                    mconf['buoyancyScale'] = torch.normal(mconf['trainBuoyancyScale'], var)

                oldGravityScale = mconf['gravityScale']
                # rand(1) is an uniform dist on the interval [0,1)
                if torch.rand(1)[0] < mconf['trainGravityProb']:
                    # Add gravity to this batch (only in the long term frames)
                    var = torch.tensor([1.], device=cuda0)
                    mconf['gravityScale'] = torch.normal(mconf['trainGravityScale'], var)

                oldGravity = mconf['gravityVec']
                if mconf['buoyancyScale'] > 0 or mconf['gravityScale'] > 0:
                    # Set to 0 gravity vector (direction of gravity)
                    mconf['gravityVec']['x'] = 0
                    mconf['gravityVec']['y'] = 0
                    mconf['gravityVec']['z'] = 0

                    # Chose randomly one of three cardinal directions and set random + or - dir
                    card_dir = 0
                    if is3D:
                        card_dir = random.randint(0,2)
                    else:
                        card_dir = random.randint(0,1)

                    updown = random.randint(0,1) * 2 - 1
                    if card_dir == 0:
                        mconf['gravityVec']['x'] = updown
                    elif card_dir == 1:
                        mconf['gravityVec']['y'] = updown
                    elif card_dir == 2:
                        mconf['gravityVec']['z'] = updown

                base_dt = mconf['dt']

                if mconf['timeScaleSigma'] > 0:
                    # FluidNet: randn() returns normal distribution with mean 0 and var 1.
                    # The mean of abs(randn) ~= 0.7972, hence the 0.2028 value below.
                    scale_dt = 0.2028 + torch.abs(torch.randn(1))[0] * \
                            mconf['timeScaleSigma']
                    mconf['dt'] = base_dt * scale_dt

                num_future_steps = mconf['longTermDivNumSteps'][0]
                # rand(1) is an uniform dist on the interval [0,1)
                # longTermDivProbability is the prob that longTermDivNumSteps[0] is taken.
                # otherwise, longTermDivNumSteps[1] is taken with prob 1 - longTermDivProbability
                if torch.rand(1)[0] > mconf['longTermDivProbability']:
                    num_future_steps = mconf['longTermDivNumSteps'][1]

                batch_dict = {}
                batch_dict['p'] = out_p
                batch_dict['U'] = out_U
                batch_dict['flags'] = flags

                # Set the simulation forward n steps (using model, no grad calculation),
                # but on the last do not perform a pressure projection.
                # We then input last state to model with grad calculation and add to global loss.
                with torch.no_grad():
                    for i in range(0, num_future_steps):
                        output_div = (i == num_future_steps)
                        lib.simulate(mconf, batch_dict, net, \
                                'convnet', output_div=output_div)

                data_lt = torch.zeros_like(data)
                data_lt[:,0] = batch_dict['p'].squeeze(1)
                data_lt[:,1:3] = batch_dict['U']
                data_lt[:,3] = batch_dict['flags'].squeeze(1)
                data_lt = data_lt.contiguous()

                mconf['dt'] = base_dt

                out_p_LT, out_U_LT = net(data_lt)
                out_div_LT = fluid.velocityDivergence(out_U_LT.contiguous(), flags)
                target_div_LT = torch.zeros_like(out_div)
                divLTLoss = divLTLambda *_divLTLoss(out_div_LT, target_div_LT)

                loss_size += divLTLoss

            # Print statistics
            p_l2_total_loss += pL2Loss.data.item()
            div_l2_total_loss += divL2Loss.data.item()
            p_l1_total_loss += pL1Loss.data.item()
            div_l1_total_loss += divL1Loss.data.item()
            if (divLTLambda > 0):
                div_lt_total_loss += divLTLoss.data.item()
            total_loss += loss_size.data.item()

            shuffled = True
            if shuffle_training and not training:
                shuffled = False
            if not shuffle_training and training:
                shuffled = False

            # Print fields for debug
            if print_training and (not shuffled) and (batch_idx*len(data) in list_to_plot) \
                and ((epoch-1) % 5 == 0):
                print_list = [batch_idx*len(data), epoch]
                filename_p = 'output_p_{0:05d}_ep_{1:03d}.png'.format(*print_list)
                filename_vel = 'output_v_{0:05d}_ep_{1:03d}.png'.format(*print_list)
                filename_div = 'output_div_{0:05d}_ep_{1:03d}.png'.format(*print_list)
                file_plot_p = glob.os.path.join(m_path, filename_p)
                file_plot_vel = glob.os.path.join(m_path, filename_vel)
                file_plot_div = glob.os.path.join(m_path, filename_div)
                with torch.no_grad():
                    lib.plotField(out=[out_p[0].unsqueeze(0),
                                       out_U[0].unsqueeze(0),
                                       out_div[0].unsqueeze(0)],
                                  tar=target[0].unsqueeze(0),
                                  flags=flags[0].unsqueeze(0),
                                  loss=[total_loss, p_l2_total_loss,
                                        div_l2_total_loss, div_lt_total_loss,
                                        p_l1_total_loss, div_l1_total_loss],
                                  mconf=mconf,
                                  epoch=epoch,
                                  filename=file_plot_p,
                                  save=save_or_show,
                                  plotPres=True,
                                  plotVel=False,
                                  plotDiv=False,
                                  title=False,
                                  x_slice=104)
                    lib.plotField(out=[out_p[0].unsqueeze(0),
                                       out_U[0].unsqueeze(0),
                                       out_div[0].unsqueeze(0)],
                                  tar=target[0].unsqueeze(0),
                                  flags=flags[0].unsqueeze(0),
                                  loss=[total_loss, p_l2_total_loss,
                                        div_l2_total_loss, div_lt_total_loss,
                                        p_l1_total_loss, div_l1_total_loss],
                                  mconf=mconf,
                                  epoch=epoch,
                                  filename=file_plot_vel,
                                  save=save_or_show,
                                  plotPres=False,
                                  plotVel=True,
                                  plotDiv=False,
                                  title=False,
                                  x_slice=104)
                    lib.plotField(out=[out_p[0].unsqueeze(0),
                                       out_U[0].unsqueeze(0),
                                       out_div[0].unsqueeze(0)],
                                  tar=target[0].unsqueeze(0),
                                  flags=flags[0].unsqueeze(0),
                                  loss=[total_loss, p_l2_total_loss,
                                        div_l2_total_loss, div_lt_total_loss,
                                        p_l1_total_loss, div_l1_total_loss],
                                  mconf=mconf,
                                  epoch=epoch,
                                  filename=file_plot_div,
                                  save=save_or_show,
                                  plotPres=False,
                                  plotVel=False,
                                  plotDiv=True,
                                  title=False,
                                  x_slice=104)

            if training:
                # Run the backpropagation for all the losses.
                loss_size.backward()

                # Step the optimizer
                optimizer.step()

            n_batches += 1

            if training:
                # Print every 20th batch of an epoch
                if batch_idx % 20 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)] \t'.format(
                        epoch, batch_idx * len(data), len(loader.dataset),
                        100. * batch_idx / len(loader)))

        # Divide loss by dataset length
        p_l2_total_loss /= n_batches
        div_l2_total_loss /= n_batches
        p_l1_total_loss /= n_batches
        div_l1_total_loss /= n_batches
        div_lt_total_loss /= n_batches
        total_loss /= n_batches

        # Print for the whole dataset
        if training:
            sstring = 'Train'
        else:
            sstring = 'Validation'
        print('\n{} set: Avg total loss: {:.6f} (L2(p): {:.6f}; L2(div): {:.6f}; \
                L1(p): {:.6f}; L1(div): {:.6f}; LTDiv: {:.6f})'.format(\
                        sstring,
                        total_loss, p_l2_total_loss, div_l2_total_loss, \
                        p_l1_total_loss, div_l1_total_loss, div_lt_total_loss))

        # Return loss scores
        return total_loss, p_l2_total_loss, div_l2_total_loss, \
                p_l1_total_loss, div_l1_total_loss, div_lt_total_loss


    #********************************* Prepare saving files *******************************

    def save_checkpoint(state, is_best, save_path, filename):
        filename = glob.os.path.join(save_path, filename)
        torch.save(state, filename)
        if is_best:
            bestname = glob.os.path.join(save_path, 'convModel_lastEpoch_best.pth')
            copyfile(filename, bestname)

    # Create some arrays for recording results
    train_loss_plot = np.empty((0,7))
    val_loss_plot = np.empty((0,7))

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
        # We do that only if resuming training.
        path, last = glob.os.path.split(m_path)
        saved_model_name = glob.os.path.join(path, last, last + '_saved.py')
        copyfile('lib/model.py', saved_model_name)

        # Delete plot file if starting from scratch
        if (glob.os.path.isfile(file_train + '.npy') and glob.os.path.isfile(file_val + '.npy')):
            print('Are you sure you want to delete existing files and start training from scratch. [y/n]')
            choice = input().lower()
            if choice in yes:
                glob.os.remove(file_train + '.npy')
                glob.os.remove(file_val + '.npy')
            elif choice in no:
                print('Exiting program')
                sys.exit()
            else:
                sys.stdout.write("Please respond with 'yes' or 'no'")

    # Save config dicts
    torch.save(conf, file_conf)
    torch.save(mconf, file_mconf)

    #********************************* Run epochs ****************************************

    n_epochs = max_epochs
    if not resume:
        state = {}
        state['bestPerf'] = float('Inf')

    print('')
    print('==> Beginning simulation')
    for epoch in range(start_epoch, n_epochs+1):
        # Train on training set and test on validation set
        train_loss, p_l2_tr, div_l2_tr, p_l1_tr, div_l1_tr, div_lt_tr = \
                run_epoch(epoch, train_loader, training=True)
        with torch.no_grad():
            val_loss, p_l2_val, div_l2_val, p_l1_val, div_l1_val, div_lt_val = \
                    run_epoch(epoch, test_loader, training=False)

        #Step scheduler, will reduce LR if loss has plateaued
        scheduler.step(train_loss)

        # Store training loss function
        train_loss_plot = np.append(train_loss_plot, [[epoch, train_loss, p_l2_tr,
            div_l2_tr, p_l1_tr, div_l1_tr, div_lt_tr]], axis=0)
        val_loss_plot = np.append(val_loss_plot, [[epoch, val_loss, p_l2_val,
            div_l2_val, p_l1_val, div_l1_val, div_lt_val]], axis=0)

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
            train_old = np.empty((0,7))
            val_old = np.empty((0,7))
            if (glob.os.path.isfile(plot_train_file) and glob.os.path.isfile(plot_val_file)):
                train_old = np.load(plot_train_file)
                val_old = np.load(plot_val_file)
            train_loss = np.append(train_old, train_loss_plot, axis=0)
            val_loss = np.append(val_old, val_loss_plot, axis=0)
            np.save(file_val, val_loss)
            np.save(file_train, train_loss)
            # Reset arrays
            train_loss_plot = np.empty((0,7))
            val_loss_plot = np.empty((0,7))

finally:
    if resume:
        # Delete model_saved_resume.py
        print()
        print('Deleting ' + temp_model)
        glob.os.remove(temp_model)
