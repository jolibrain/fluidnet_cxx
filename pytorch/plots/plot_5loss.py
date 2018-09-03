import sys
import numpy as np
import matplotlib.pyplot as plt
import glob


# Load numpy arrays
assert (len(sys.argv) == 2), 'Usage: python3 plot_loss.py <plotDirName>'
assert (glob.os.path.exists(sys.argv[1])), 'Directory ' + str(sys.argv[1]) + ' does not exists'

save_dir = sys.argv[1]
file_1 = glob.os.path.join(save_dir, 'FluidNet_Jacobi.npy')
file_2 = glob.os.path.join(save_dir, 'FluidNet_LongTerm.npy')
file_3 = glob.os.path.join(save_dir, 'FluidNet_ShortTerm.npy')
#file_val = glob.os.path.join(save_dir, 'val_loss.npy')
jacobi_div_plot = np.load(file_1)
LT_div_plot = np.load(file_2)
ST_div_plot = np.load(file_3)
#val_loss_plot = np.load(file_val)

init = 0
#print(jacobi_div_plot)

# Plot loss against epochs
plt.plot(LT_div_plot[:,0], LT_div_plot[:,1], label = 'LT')
plt.plot(ST_div_plot[:,0], ST_div_plot[:,1], label = 'ST')
plt.plot(jacobi_div_plot[:,0], jacobi_div_plot[:,1], label = 'Jacobi (28 iterations)')
plt.legend()
plt.ylabel('E(div)')
plt.xlabel('timestep')
plt.show(block=True)

