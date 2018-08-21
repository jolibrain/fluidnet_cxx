import sys
import numpy as np
import matplotlib.pyplot as plt
import glob


# Load numpy arrays
assert (len(sys.argv) == 2), 'Usage: python3 plot_loss.py <plotDirName>'
assert (glob.os.path.exists(sys.argv[1])), 'Directory ' + str(sys.argv[1]) + ' does not exists'

save_dir = sys.argv[1]
file_train = glob.os.path.join(save_dir, 'train_loss.npy')
file_val = glob.os.path.join(save_dir, 'val_loss.npy')
train_loss_plot = np.load(file_train)
val_loss_plot = np.load(file_val)

init = 5

# Plot loss against epochs
plt.plot(train_loss_plot[init:,0], train_loss_plot[init:,1], label = 'Total Training Loss')
plt.plot(train_loss_plot[init:,0], train_loss_plot[init:,2], label = 'L2 pressure Loss')
plt.plot(train_loss_plot[init:,0], train_loss_plot[init:,3], label = 'L2 div Loss')
plt.plot(train_loss_plot[init:,0], train_loss_plot[init:,4], label = 'L1 pressure Loss')
plt.plot(train_loss_plot[init:,0], train_loss_plot[init:,5], label = 'L1 div Loss')
plt.plot(train_loss_plot[init:,0], train_loss_plot[init:,6], label = 'Long Term div Loss')
plt.plot(val_loss_plot[init:,0], val_loss_plot[init:,1], label = 'Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show(block=True)

