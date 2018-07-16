import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fluid
from math import inf

class _ScaleNet(nn.Module):
    def __init__(self, mconf):
        super(_ScaleNet, self).__init__()
        self.mconf = mconf

    def forward(self, x, scale, invertScale=True):
        if invertScale:
            bsz = x.size(0)
            # Rehaspe form (b x 2/3 x d x h x w) to (b x -1)
            y = x.view(bsz, -1)
            # Calculate std using Bessel's correction (correction with n/n-1)
            std = torch.std(y, dim=1, keepdim=True) # output is size (b x 1)
            scale = torch.clamp(std, \
                self.mconf['normalizeInputThreshold'] , inf)
            scale = scale.view(bsz, 1, 1, 1, 1)
            x = torch.div(x, scale)

        else:
            x = torch.mul(x, scale)

        return x, scale

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
        # Input channels = 3 (Ux, Uy, flags)
        # We add padding to make sure that Win = Wout and Hin = Hout with ker_size=3
        self.conv1 = torch.nn.Conv2d(self.inDims, 16, kernel_size=3, padding=1)

        self.modDown1 = torch.nn.AvgPool2d(kernel_size=2)
        self.modDown2 = torch.nn.AvgPool2d(kernel_size=4)

        self.convBank = _HiddenConvBlock(dropout)

        self.upscale1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.upscale2 = torch.nn.Upsample(scale_factor=4, mode='nearest')

        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=1)

        # Output channels = 1 (pressure)
        self.convOut = torch.nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, input_):

        # data indexes     |           |
        #       (dim 1)    |    2D     |    3D
        # ----------------------------------------
        #   DATA:
        #       pDiv       |    0      |    0
        #       UDiv       |    1:3    |    1:4
        #       flags      |    3      |    4
        #       densityDiv |    4      |    5
        #   TARGET:
        #       p          |    5      |    6
        #       U          |    6:8    |    7:10
        #       density    |    8      |    10

        # For now, we work ONLY in 2d

        assert self.is3D == False, 'Input can only be 2D'

        assert self.mconf['inputChannels']['pDiv'] or \
                self.mconf['inputChannels']['UDiv'] or \
                self.mconf['inputChannels']['div'], 'Choose at least one field (U, div or p).'

        if (self.mconf['inputChannels']['pDiv'] or (self.mconf['normalizeInput'] \
            and self.mconf['normalizeInputChan'] == 'pDiv')):
            pDiv = input_[:,0].unsqueeze(1)
        if (self.mconf['inputChannels']['UDiv'] or self.mconf['inputChannels']['div'] \
            or (self.mconf['normalizeInput'] \
            and self.mconf['normalizeInputChan'] == 'UDiv')):
            if self.is3D:
                UDiv = input_[:,1:4]
            else:
                UDiv = input_[:,1:3]
        # Flags are always loaded
        if self.is3D:
            flags = input_[:,4]
        else:
            flags = input_[:,3]

        # Apply setWallBcs to zero out obstacle velocities on the boundary

        # Apply scale to input velocity
        s = 0
        UDiv, s = self.scale(UDiv, s, invertScale=True)
        # Flags to occupancy
        flags = fluid.flagsToOccupancy(flags)
        x = torch.FloatTensor(input_.size(0), \
                              self.inDims,    \
                              input_.size(2), \
                              input_.size(3), \
                              input_.size(4)).type_as(input_)
        x[:,0,:,:,:] = UDiv.select(1,0)
        x[:,1,:,:,:] = UDiv.select(1,1)
        x[:,2,:,:,:] = flags

        # Squeeze unary dimension as we are in 2D
        x = torch.squeeze(x,2)
        x = F.relu(self.conv1(x))

        # We divide the network in 3 banks, applying average pooling
        x1 = self.modDown1(x)
        x2 = self.modDown2(x)

        # Process every bank in parallel
        x0 = self.convBank(x)
        x1 = self.convBank(x1)
        x2 = self.convBank(x2)

        # Upsample banks 1 and 2 to bank 0 size and accumulate inputs
        x1 = self.upscale1(x1)
        x2 = self.upscale2(x2)
        x = x0 + x1 + x2

        # Apply lasts 2 convolutions
        x = F.relu(self.conv2(x))
        x = self.convOut(x)

        # Add back the unary dimension
        x = torch.unsqueeze(x, 2)
        x, s = self.scale(x, s, invertScale=False)
        return x


class FluidNetOld(nn.Module):
    # For now, only 2D model. Add 2D/3D option. Only known from data!
    # Also, build model with MSE of pressure as loss func, therefore input is velocity
    # and output is pressure, to be compared to target pressure.
    def __init__(self, conf, dropout=True):
        super(FluidNetOld, self).__init__()

        self.dropout = dropout
        self.conf = conf

        self.scale = _ScaleNet(self.conf)
        # Input channels = 3 (Ux, Uy, flags)
        # We add padding to make sure that Win = Wout and Hin = Hout with ker_size=3
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=1)
        self.modDown1 = torch.nn.AvgPool2d(kernel_size=2)
        self.modDown2 = torch.nn.AvgPool2d(kernel_size=4)

        self.upscale1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.upscale2 = torch.nn.Upsample(scale_factor=4, mode='nearest')

        # Output channels = 1 (pressure)
        self.convOut = torch.nn.Conv2d(16, 1, kernel_size=1)


    def forward(self, x):

        if self.dropout:
            x = F.dropout3d(x)
        UDiv = torch.cat((x.select(1,0).unsqueeze(1), x.select(1,1).unsqueeze(1)), 1)
        flags = x.select(1,2).unsqueeze(1)
        # Apply setWallBcs to zero out obstacle velocities on the boundary

        # Apply scale to input velocity
        s = 0
        UDiv, s = self.scale(UDiv, s, invertScale=True)
        # Flags to occupancy
        flags = fluid.FlagsToOccupancy(flags)
        x[:,0,:,:,:] = UDiv.select(1,0)
        x[:,1,:,:,:] = UDiv.select(1,1)
        x[:,2,:,:,:] = flags.squeeze(1)

        # Squeeze unary dimension as we are in 2D
        x = torch.squeeze(x,2)
        x = F.prelu(self.conv1(x))

        # We divide the network in 3 banks, applying average pooling
        x0 = F.prelu(self.conv2(x))

        x1 = self.modDown1(x)

        x1 = F.relu(self.conv2(x1))
        x2 = self.modDown2(x)
        x2 = F.relu(self.conv2(x2))

        # Process every bank in parallel
        x0 = F.relu(self.conv2(x0))
        x1 = F.relu(self.conv2(x1))
        x2 = F.relu(self.conv2(x2))

        # Upsample banks 1 and 2 to bank 0 size and accumulate inputs
        x1 = self.upscale1(x1)
        x2 = self.upscale2(x2)
        x = x0 + x1 + x2

        # Apply lasts 2 convolutions
        x = F.relu(self.conv3(x))
        x = F.relu(self.convOut(x))

        # Add back the unary dimension
        x = torch.unsqueeze(x, 2)
        x, s = self.scale(x, s, invertScale=False)
        return x

