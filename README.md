fluidnet_cxx
============

This repo is based on Tompson et al. paper:

[Accelerating Eulerian Fluid Simulation With Convolutional Networks, Jonathan Tompson, Kristofer Schlachter, Pablo Sprechmann, Ken Perlin](http://cims.nyu.edu/~schlacht/CNNFluids.htm).

Find the original repository [here](https://github.com/google/FluidNet).

The object of this work is to investigate the architecture of the FluidNet architecture with a C++/Python backend, i.e. using PyTorch and ATen Tensor library, that exposes PyTorch operations in C++11. This allows to replace all Lua code (Torch7) and avoid implementing two distinct kernels for CPU or CUDA.
ATen allows to write generic code that works on both devices.
More information in ATen [repo](https://github.com/zdevito/ATen). It can be called from\PyTorch, using its new extension-cpp.

To install this repo:

#0. Clone this repo:
---------------

```
git clone git@github.com:AAlguacil/fluidnet_cxx.git
```

#1. Install Pytorch 0.4:
---------------

[Pytorch 0.4](https://pytorch.org/)

#2. Install cpp extensions for fluid solver:
---------------

C++ scripts have been written using PyTorch's backend C++ library ATen.
These scripts are used for the advection part of the solver.

Follow these instructions from main directory:
```

cd pytorch/lib/fluid/cpp
python3 setup.py install # if you want to install it on local user, use --user
```

#3. Datatset:
---------------
We use the same dataset as the original FluidNet (generated with MantaFlow) for training
our ConvNet

#4. Training:
---------------
First go to 
```

cd pytorch
```
and open ```config.py``` with your favorite text editor.
Set the location of your dataset in ```dataDir``` and ```dataset``` and the folder to 
save the trained model, configuration files as well as metrics (losses) in ```modelDir ```
To launch the training:
```

python3 fluid_net_train.py
```

In you train for the first time ever with the FluidNet, you will have to preprocess it.
Set ```preprocOnly``` to ```True```.
This will create new torch arrays from the original binary files, being faster reading 
during training. This process can take some time but it is necessary only once!.
For the rest of the work  ```preprocOnly``` must be ```False```.

You can interrupt at any time your training and resume it, by setting ```resumeTraining```
to ```True```.

You can also monitor the loss during training by running in ```/pytorch```

```

python3 plot_loss.py <modelDir> #For total training and validation losses
#or
python3 plot_5loss.py <modelDir> #For each of the losses (e.g: L1(div) and L2(div))

```

It is also possible to take the a saved model and print its output fields and
compare it to targets (Pressure, Velocity, Divergence and Errors):
```

python3 print_output.py <modelDir> <modelFilename>
#example:
python3 print_output.py data/model_pLoss_L2 convModel
```

Finally, you can use your saved trained model to perform a full simulation, i.e. 
advection + pressure projection during several time steps. Run:
```

python3 fluid_net_simulate.py <modelDir> <modelFilename>
```





