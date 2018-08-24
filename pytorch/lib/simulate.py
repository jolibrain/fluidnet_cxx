import torch
import math
import lib.fluid as fluid

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

    dt = mconf['dt']
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
    else:
        density = torch.zeros_like(flags)

    if viscosity == 0:
        # Self-advect velocity if inviscid
        U = fluid.advectVelocity(dt=dt, orig=U, U=U, flags=flags, method="maccormackFluidNet", \
            boundary_width=1, maccormack_strength=maccormackStrength)
    else:
        # Self-advect velocity if inviscid
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
            U = fluid.addBuoyancy(U, flags, density, gravity, dt)
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
        U = fluid.setWallBcs(U, flags)
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
        p, U = net(data, float(dt))

    elif (sim_method == 'jacobi'):
        div = fluid.velocityDivergence(U, flags)

        is3D = (U.size(2) > 1)
        pTol = 0
        maxIter = mconf['jacobiIter']

        p, residual = fluid.solveLinearSystemJacobi(dt, flags, div, is_3d=is3D, p_tol=pTol, \
                max_iter=maxIter)

        fluid.velocityUpdate(dt, p, U, flags)

    if sim_method != 'convnet':
        U = fluid.setWallBcs(U, flags)
    elif stick:
        fluid.setWallBcsStick(U, flags, flags_stick)

    setConstVals(batch_dict, p, U, flags, density)
    batch_dict['U'] = U
    batch_dict['density'] = density
    batch_dict['p'] = p
