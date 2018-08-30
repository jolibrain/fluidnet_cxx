from load_manta_data import loadMantaFile
import matplotlib.pyplot as plt
import torch

p, U, flags, density, is3D = loadMantaFile('../data_test/test_model/tr/000318/000036.bin')

pDiv, UDiv, flagsDiv, densityDiv, is3DDiv = loadMantaFile('../data_test/test_model/tr/000318/000036_divergent.bin')
torch.set_printoptions(precision=4, threshold=None, edgeitems=10, linewidth=None, profile=None)

print(p)
print(pDiv)
print(UDiv)
