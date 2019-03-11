import torch
import math

def createPlumeBCs(batch_dict, density_val, u_scale, rad):
    r"""Creates masks to enforce an inlet at the domain bottom wall.
    Modifies batch_dict inplace.
    Arguments:
        batch_dict (dict): Input tensors (p, UDiv, flags, density)
        density_val (float): Inlet density.
        u_scale (float); Inlet velocity.
        rad (float): radius of inlet circle (centered around midpoint of wall)
    """

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
    UBC[:,:,:,0:4] = maskInside_f * vec.view(1,2,1,1,1).expand_as(UBC[:,:,:,0:4]).float()
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

def createRayleighTaylorBCs(batch_dict, mconf, rho1, rho2):
    r"""Creates masks to enforce a Rayleigh-Taylor instability initial conditions.
    Top fluid has a density rho1 and lower one rho2. rho1 > rho2 to trigger instability.
    Modifies batch_dict inplace.
    Arguments:
        batch_dict (dict): Input tensors (p, UDiv, flags, density)
        mconf (dict): configuration dict (to set thickness and amplitude of interface).
        rho1 (float): Top fluid density.
        rho2 (float): Lower fluid density.
    """

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

    X = torch.arange(0, resX, device=cuda).view(resX).expand((1,resY,resX))
    Y = torch.arange(0, resY, device=cuda).view(resY, 1).expand((1,resY,resX))
    coord = torch.cat((X,Y), dim=0).unsqueeze(0).unsqueeze(2)

    # Atwood number
    #A = ((1+rho2) - (1+rho1)) / ((1+rho2) + (1+rho1))
    #print('Atwood number : ' + str(A))
    #density = ((1-A) * torch.tanh(100*(coord[:,1]/resY - (0.85 - \
    #                0.05*torch.cos(math.pi*(coord[:,0]/resX)))))).unsqueeze(1)
    thick = mconf['perturbThickness']
    ampl = mconf['perturbAmplitude']
    h = mconf['height']
    density = 0.5*(rho2+rho1 + (rho2-rho1)*torch.tanh(thick*(coord[:,1]/resY - \
            (h + ampl*torch.cos(2*math.pi*(coord[:,0]/resX)))))).unsqueeze(1)

    batch_dict['density'] = density
    batch_dict['flags'] = flags

    # batch_dict at output = {p, UDiv, flags, density}

