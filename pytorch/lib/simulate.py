import torch
import lib.fluid as fluid

def setConstVals(batch_dict, p, U, flags, density):
    # apply external BCs.
    # batch_dict at output = {p, UDiv, flags, density, UBC,
    #                         UBCInvMask, densityBC, densityBCInvMask}

    #if 'cylinder' in batch_dict:
    #    # Zero out the U values on the BCs.
    #    U.mul_(batch_dict['InvInletMask'])
    #    # Add back the values we want to specify.
    #    U.add_(batch_dict['UInlet'])
    #    batch_dict['U'] = U.clone()

    if ('UBCInvMask' in batch_dict) and ('UBC' in batch_dict):
        # Zero out the U values on the BCs.
        U.mul_(batch_dict['UBCInvMask'])
        # Add back the values we want to specify.
        U.add_(batch_dict['UBC'])
        batch_dict['U'] = U.clone()

    if ('densityBCInvMask' in batch_dict) and ('densityBC' in batch_dict):
        density.mul_(batch_dict['densityBCInvMask'])
        density.add_(batch_dict['densityBC'])
        batch_dict['density'] = density.clone()

def simulate(conf, mconf, batch_dict, net, sim_method, output_div=False):
    r"""Top level simulation loop.

    Arguments:
        conf (dict): Configuration dictionnary.
        mconf (dict): Model configuration dictionnary.
        batch_dict (dict): Dictionnary of torch Tensors.
            Keys must be 'U', 'flags', 'p', 'density'.
            Simulations are done INPLACE.
        net (nn.Module): convNet model.
        sim_method (string): Options are 'convnet' and 'jacobi'
        output_div (bool, optional): returns just before solving for pressure.
            i.e. leave the state as UDiv and pDiv (before substracting divergence)

    """
    cuda = torch.device('cuda')
    assert sim_method == 'convnet' or sim_method == 'jacobi', 'Simulation method \
                not supported. Choose either convnet or jacobi.'

    dt = float(mconf['dt'])
    maccormackStrength = mconf['maccormackStrength']
    sampleOutsideFluid = mconf['sampleOutsideFluid']

    buoyancyScale = mconf['buoyancyScale']
    gravityScale = mconf['gravityScale']

    viscosity = mconf['viscosity']
    assert viscosity >= 0, 'Viscosity must be positive'

    # Get p, U, flags and density from batch.
    p = batch_dict['p']
    U = batch_dict['U']

    flags = batch_dict['flags']
    stick = False
    if 'flags_stick' in batch_dict:
        stick = True
        flags_stick = batch_dict['flags_stick']

    # If viscous model, add viscosity
    if (viscosity > 0):
        orig = U.clone()
        fluid.addViscosity(dt, orig, flags, viscosity)

    if 'density' in batch_dict:
        density = batch_dict['density']

        # First advect all scalar fields.
        density = fluid.advectScalar(dt, density, U, flags, \
                method="maccormackFluidNet", \
                boundary_width=1, sample_outside_fluid=sampleOutsideFluid, \
                maccormack_strength=maccormackStrength)
        if mconf['correctScalar']:
            div = fluid.velocityDivergence(U, flags)
            fluid.correctScalar(dt, density, div, flags)
    else:
        density = torch.zeros_like(flags)

    if viscosity == 0:
        # Self-advect velocity if inviscid
        U = fluid.advectVelocity(dt=dt, orig=U, U=U, flags=flags, method="maccormackFluidNet", \
            boundary_width=1, maccormack_strength=maccormackStrength)
    else:
        # Advect viscous velocity field orig by the non-divergent
        # velocity field U.
        U = fluid.advectVelocity(dt=dt, orig=orig, U=U, flags=flags, method="maccormackFluidNet", \
            boundary_width=1, maccormack_strength=maccormackStrength)

    # Set the manual BCs.
    setConstVals(batch_dict, p, U, flags, density)

    if 'density' in batch_dict:
        if buoyancyScale > 0:
            # Add external forces: buoyancy.
            gravity = torch.FloatTensor(3).fill_(0).cuda()
            gravity[0] = mconf['gravityVec']['x']
            gravity[1] = mconf['gravityVec']['y']
            gravity[2] = mconf['gravityVec']['z']
            gravity.mul_(-buoyancyScale)
            rho_star = mconf['operatingDensity']
            U = fluid.addBuoyancy(U, flags, density, gravity, rho_star, dt)
        if gravityScale > 0:
            gravity = torch.FloatTensor(3).fill_(0).cuda()
            gravity[0] = mconf['gravityVec']['x']
            gravity[1] = mconf['gravityVec']['y']
            gravity[2] = mconf['gravityVec']['z']
            # Add external forces: gravity.
            gravity.mul_(-gravityScale)
            U = fluid.addGravity(U, flags, gravity, dt)

    if (output_div):
        return

    if sim_method != 'convnet':
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
        U = fluid.setWallBcs(U, flags)
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:
                U[:,1,:,:,1] = U_temp[:,1,:,:,U.size(4)-1]
            if mconf['periodic-y']:
                U[:,0,:,1] = U_temp[:,0,:,U.size(3)-1]
    elif stick:
        fluid.setWallBcsStick(U, flags, flags_stick)

    # Set the constant domain values.
    setConstVals(batch_dict, p, U, flags, density)


    if (sim_method == 'convnet'):
        # fprop the model to perform the pressure projection and velocity calculation.
        # Set wall BCs is performed inside the model, before and after the projection.
        # No need to call it again.
        net.eval()
        data = torch.cat((p, U, flags, density), 1)
        p, U = net(data)

    elif (sim_method == 'jacobi'):
        div = fluid.velocityDivergence(U, flags)

        is3D = (U.size(2) > 1)
        pTol = mconf['pTol']
        maxIter = mconf['jacobiIter']

        p, residual = fluid.solveLinearSystemJacobi( \
                flags=flags, div=div, is_3d=is3D, p_tol=pTol, \
                max_iter=maxIter)
        fluid.velocityUpdate(pressure=p, U=U, flags=flags)

    if sim_method != 'convnet':
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
        U = fluid.setWallBcs(U, flags)
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:
                U[:,1,:,:,1] = U_temp[:,1,:,:,U.size(4)-1]
            if mconf['periodic-y']:
                U[:,0,:,1] = U_temp[:,0,:,U.size(3)-1]
    elif stick:
        fluid.setWallBcsStick(U, flags, flags_stick)

    setConstVals(batch_dict, p, U, flags, density)
    batch_dict['U'] = U
    batch_dict['density'] = density
    batch_dict['p'] = p
