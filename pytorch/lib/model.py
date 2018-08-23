import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import fluid, MultiScaleNet
from math import inf

class _ScaleNet(nn.Module):
    def __init__(self, mconf):
        super(_ScaleNet, self).__init__()
        self.mconf = mconf

    def forward(self, x):
        bsz = x.size(0)
        # Rehaspe form (b x chan x d x h x w) to (b x -1)
        y = x.view(bsz, -1)
        # Calculate std using Bessel's correction (correction with n/n-1)
        std = torch.std(y, dim=1, keepdim=True) # output is size (b x 1)
        scale = torch.clamp(std, \
            self.mconf['normalizeInputThreshold'] , inf)
        scale = scale.view(bsz, 1, 1, 1, 1)

        return scale

class _HiddenConvBlock(nn.Module):
    def __init__(self, dropout=True):
        super(_HiddenConvBlock, self).__init__()
        layers = [
            nn.Conv2d(16, 16, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding = 1),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FluidNet(nn.Module):
    # For now, only 2D model. Add 2D/3D option. Only known from data!
    # Also, build model with MSE of pressure as loss func, therefore input is velocity
    # and output is pressure, to be compared to target pressure.
    def __init__(self, mconf, dropout=True):
        super(FluidNet, self).__init__()

        self.dropout = dropout
        self.mconf = mconf
        self.inDims = mconf['inputDim']
        self.is3D = mconf['is3D']

        self.scale = _ScaleNet(self.mconf)
        # Input channels = 3 (inDims, flags)
        # We add padding to make sure that Win = Wout and Hin = Hout with ker_size=3
        self.conv1 = torch.nn.Conv2d(self.inDims, 16, kernel_size=3, padding=1)

        self.modDown1 = torch.nn.AvgPool2d(kernel_size=2)
        self.modDown2 = torch.nn.AvgPool2d(kernel_size=4)

        self.convBank = _HiddenConvBlock(dropout)

        #self.upscale1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        #self.upscale2 = torch.nn.Upsample(scale_factor=4, mode='nearest')

        self.deconv1 = torch.nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.deconv2 = torch.nn.ConvTranspose2d(16, 16, kernel_size=4, stride=4)

        self.conv2 = torch.nn.Conv2d(16*3, 16, kernel_size=1)

        # Output channels = 1 (pressure)
        self.convOut = torch.nn.Conv2d(16, 1, kernel_size=1)

        # MultiScaleNet
        self.multiScale = MultiScaleNet(self.inDims)

    def forward(self, input_, dt=1.0):

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

        # For now, we work ONLY in 2d

        assert self.is3D == False, 'Input can only be 2D'

        assert self.mconf['inputChannels']['pDiv'] or \
                self.mconf['inputChannels']['UDiv'] or \
                self.mconf['inputChannels']['div'], 'Choose at least one field (U, div or p).'

        pDiv = None
        UDiv = None
        div = None

        # Flags are always loaded
        if self.is3D:
            flags = input_[:,4].unsqueeze(1)
        else:
            flags = input_[:,3].unsqueeze(1).contiguous()

        if (self.mconf['inputChannels']['pDiv'] or (self.mconf['normalizeInput'] \
            and self.mconf['normalizeInputChan'] == 'pDiv')):
            pDiv = input_[:,0].unsqueeze(1).contiguous()

        if (self.mconf['inputChannels']['UDiv'] or self.mconf['inputChannels']['div'] \
            or (self.mconf['normalizeInput'] \
            and self.mconf['normalizeInputChan'] == 'UDiv')):
            if self.is3D:
                UDiv = input_[:,1:4].contiguous()
            else:
                UDiv = input_[:,1:3].contiguous()

            # Apply setWallBcs to zero out obstacles velocities on the boundary
            UDiv = fluid.setWallBcs(UDiv, flags)

            if self.mconf['inputChannels']['div']:
                div = fluid.velocityDivergence(UDiv, flags)

        # Apply scale to input
        if self.mconf['normalizeInput']:
            if self.mconf['normalizeInputChan'] == 'UDiv':
                s = self.scale(UDiv)
            elif self.mconf['normalizeInputChan'] == 'pDiv':
                s = self.scale(pDiv)
            elif self.mconf['normalizeInputChan'] == 'div':
                s = self.scale(div)
            else:
                raise Exception('Incorrect normalize input channel.')

            if pDiv is not None:
                pDiv = torch.div(pDiv, s)
            if UDiv is not None:
                UDiv = torch.div(UDiv, s)
            if div is not None:
                div = torch.div(div, s)

        x = torch.FloatTensor(input_.size(0), \
                              self.inDims,    \
                              input_.size(2), \
                              input_.size(3), \
                              input_.size(4)).type_as(input_)

        chan = 0
        if self.mconf['inputChannels']['pDiv']:
            x[:, chan] = pDiv[:,0]
            chan += 1
        elif self.mconf['inputChannels']['UDiv']:
            if self.is3D:
                x[:,chan:(chan+3)] = UDiv
                chan += 3
            else:
                x[:,chan:(chan+2)] = UDiv
                chan += 2
        elif self.mconf['inputChannels']['div']:
            x[:, chan] = div[:,0]
            chan += 1

        # FlagsToOccupancy creates a [0,1] grid out of the manta flags
        x[:,chan,:,:,:] = fluid.flagsToOccupancy(flags).squeeze(1)

        if not self.is3D:
            # Squeeze unary dimension as we are in 2D
            x = torch.squeeze(x,2)

        if self.mconf['model'] == 'ScaleNet':
            p = self.multiScale(x)

        else:
            # Inital layers
            x = F.relu(self.conv1(x))

            # We divide the network in 3 banks, applying average pooling
            x1 = self.modDown1(x)
            x2 = self.modDown2(x)

            # Process every bank in parallel
            x0 = self.convBank(x)
            x1 = self.convBank(x1)
            x2 = self.convBank(x2)

            # Upsample banks 1 and 2 to bank 0 size and accumulate inputs
            #x1 = self.upscale1(x1)
            #x2 = self.upscale2(x2)
            x1 = self.deconv1(x1)
            x2 = self.deconv2(x2)

            x = torch.cat((x0, x1, x2), dim=1)
            #x = x0 + x1 + x2

            # Apply last 2 convolutions
            x = F.relu(self.conv2(x))

            # Output pressure (1 chan)
            p = self.convOut(x)


        # Add back the unary dimension
        if not self.is3D:
            p = torch.unsqueeze(p, 2)

        # Correct U = UDiv - grad(p)
        # flags is the one with Manta's values, not occupancy in [0,1]
        fluid.velocityUpdate(dt, p, UDiv, flags)

        # We now UNDO the scale factor we applied on the input.
        if self.mconf['normalizeInput']:
            p = torch.mul(p,s)  # Applies p' = *= scale
            UDiv = torch.mul(UDiv,s)

        # Set BCs after velocity update.
        UDiv = fluid.setWallBcs(UDiv, flags)
        return p, UDiv


