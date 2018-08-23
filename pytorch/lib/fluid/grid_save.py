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
    if is3D:
        d = d -1
    else:
        d = 2
    h -= 1
    w -= 1

    idx_x = torch.arange(1, w, dtype=torch.long, device=cuda).view(1,w-1).expand(bsz,d-1,h-1,w-1)
    idx_y = torch.arange(1, h, dtype=torch.long, device=cuda).view(1,h-1, 1).expand(bsz,d-1,h-1,w-1)
    idx_z = torch.zeros_like(idx_x)
    if (is3D):
       idx_z = torch.arange(1, d, dtype=torch.long, device=cuda).view(1,d-1, 1 , 1).expand(bsz,d-1,h-1,w-1)

    idx_b = torch.arange(0, bsz, dtype=torch.long, device=cuda).view(bsz,1,1,1)
    idx_b = idx_b.expand(bsz,d-1,h-1,w-1)

    idx_c = torch.arange(0, 3, dtype=torch.long, device=cuda).view(1,3,1,1,1)

    c_vel_x = 0.5 * ( self[idx_b,idx_c[:,0] ,idx_z  ,idx_y  ,idx_x  ] + \
                      self[idx_b,idx_c[:,0] ,idx_z  ,idx_y  ,idx_x+1] )
    c_vel_y = 0.5 * ( self[idx_b,idx_c[:,1] ,idx_z  ,idx_y  ,idx_x  ] + \
                      self[idx_b,idx_c[:,1] ,idx_z  ,idx_y+1,idx_x  ] )
    c_vel_z = torch.zeros_like(c_vel_x)

    if (is3D):
      c_vel_z = 0.5 * ( self[idx_b,idx_c.select(1,2) ,idx_z  ,idx_y  ,idx_x  ] + \
                        self[idx_b,idx_c.select(1,2) ,idx_z+1,idx_y  ,idx_x  ] )

    return torch.stack((c_vel_x, c_vel_y, c_vel_z), dim=1)


U = torch.rand(1,2,1,6,6)
U_cent = getCentered(U)
print(U_cent.size())
