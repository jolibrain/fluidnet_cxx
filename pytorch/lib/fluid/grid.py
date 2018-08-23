import torch

def getDx(self):
    grid_size_max = max(max(self.size(2), self.size(3)), self.size(4))
    return (1.0 / grid_size_max)

def getCentered(self):

    cuda = torch.device('cuda')
    bsz = self.size(0)
    d = self.size(2)
    h = self.size(3)
    w = self.size(4)

    is3D = (d > 1)

    c_vel_x = torch.zeros_like(self[:,0])
    c_vel_x[:,:,:,:-1] = 0.5 * (self[:,0,:,:,0:-1] + \
                                self[:,0,:,:,1:])
    c_vel_y = torch.zeros_like(self[:,0])
    c_vel_y[:,:,:-1,:] = 0.5 * (self[:,1,:,0:-1,:] + \
                                self[:,1,:,1:,:])
    c_vel_z = torch.zeros_like(self[:,0])

    if (is3D):
        c_vel_z = torch.zeros_like(self[:,0])
        c_vel_z[:,:-1,:,:] = 0.5 * (self[:,2,0:-1,:,:] + \
                                    self[:,2,1:,:,:])

    return torch.stack((c_vel_x, c_vel_y, c_vel_z), dim=1)

