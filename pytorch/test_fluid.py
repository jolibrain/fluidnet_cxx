import fluid
import torch

U = torch.rand(1,2,1,5,5)
flags = torch.IntTensor(1,1,1,5,5).random_(1,3)

UDiv = fluid.velocityDivergence(U, flags)
print(UDiv)
