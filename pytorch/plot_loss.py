import numpy as np
import matplotlib.pyplot as plt
import glob


# Load numpy arrays
save_dir = 'plot'
file_train = glob.os.path.join(save_dir, 'train_loss.npy')
file_val = glob.os.path.join(save_dir, 'val_loss.npy')
train_loss_plot = np.load(file_train)
val_loss_plot = np.load(file_val)

# Plot loss against epochs
plt.plot(train_loss_plot[:,0], train_loss_plot[:,1], label = 'Training Loss')
plt.plot(val_loss_plot[:,0], val_loss_plot[:,1], label = 'Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show(block=True)

