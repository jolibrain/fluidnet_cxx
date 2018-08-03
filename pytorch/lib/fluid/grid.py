import torch

def getDx(self):
    grid_size_max = max(max(self.size(2), self.size(3)), self.size(4))
    return (1.0 / grid_size_max)
