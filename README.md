fluidnet_cxx
============

This repo is based on Tompson et al. paper:

[Accelerating Eulerian Fluid Simulation With Convolutional Networks, Jonathan Tompson, Kristofer Schlachter, Pablo Sprechmann, Ken Perlin](http://cims.nyu.edu/~schlacht/CNNFluids.htm).

Find the original repository [here](https://github.com/google/FluidNet).

The object of this work is to investigate the architecture of the FluidNet architecture with a C++/Python backend, i.e. using PyTorch and ATen Tensor library, that exposes PyTorch operations in C++11. This allows to replace all Lua code (Torch7) and avoid implementing two distinct kernels for CPU or CUDA.
ATen allows to write generic code that works on both devices.
More information in ATen [repo](https://github.com/zdevito/ATen). It can be called from PyTorch, using its new extension-cpp.

To install this repo:

#0. Clone this repo:
---------------

```
git clone git@github.com:AAlguacil/fluidnet_cxx.git
```

#1. Install Pytorch 0.4:
---------------

[Pytorch 0.4](https://pytorch.org/)
__NOTE: Training is done in GPUs__

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
We use the same **2D dataset** as the original FluidNet [Section 1: Generating the data - Generating training data](https://github.com/google/FluidNet#1-generating-the-data) (generated with MantaFlow) for training
our ConvNet.

#4. Training:
---------------
First, go to pytorch folder: 
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

In you train for the first time ever with the FluidNet datasset, you will have to **preprocess** it.
Set ```preprocOnly``` to ```True```.
This will create and save the data as torch arrays from the original binary files,
making the reading of data faster during training faster.
This process can take some time but it is necessary only once!.
For the rest of the work  ```preprocOnly``` must be set to ```False```.

You can interrupt at any time your training and resume it, by setting ```resumeTraining```
to ```True```.

You can also monitor the loss during training by running in ```/pytorch```

```
python3 plot_loss.py <modelDir> #For total training and validation losses
#or
python3 plot_5loss.py <modelDir> #For each of the losses (e.g: L1(div) and L2(div))
```

It is also possible to take the saved model and print its output fields and
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

#5. Extending the cpp code:
---------------

The cpp code, written with ATen library, can be compiled, tested and run on its own.
You will need [OpenCV2](https://opencv.org/opencv-2-4-8.html) to visualize output, as matplotlib is unfortunately not available!

**Test**

First, generate the test data from FluidNet
[Section 3. Limitations of the current system - Unit Testing](https://github.com/google/FluidNet#3-limitations-of-the-current-system) and set the location of your folder in:
```
solver_cpp/test/test_fluid.cpp
#define DATA <path_to_data>
```
Then run the following commands:
```
cd solver_cpp/
mkdir build_test
cd build_test
cmake .. -DFLUID_TEST=ON # Default is off is OFF
./test/fluidnet_sim
```
This will test every routine of the solver (advection, divergence calculation, velocity
update, adding of gravity and buoyancy, linear system resolution with Jacobi method).
These tests are taken from FluidNet and compare outputs of Manta to ours, except for 
advection when there is no Manta equivalent. In that case, we compare to the original
FluidNet advection.

**Run**

```
cd solver_cpp/
mkdir build
cd build
cmake .. -DFLUID_TEST=OFF # Default is off is OFF
./simulate/fluidnet_sim
```
Output images will be written in ```build``` folder, and can be converted into gif using
ImageMagick.

**NOTE: For the moment, only 2D simulations and training are supported, as bugs are still
found for the 3D advection.**



