fluidnet_cxx
============

This repo is based on Tompson et al. paper:

[Accelerating Eulerian Fluid Simulation With Convolutional Networks, Jonathan Tompson, Kristofer Schlachter, Pablo Sprechmann, Ken Perlin](http://cims.nyu.edu/~schlacht/CNNFluids.htm).

Find the original repository [here](https://github.com/google/FluidNet).

The object of this work is to investigate the architecture of the FluidNet architecture with a C++/Python backend, i.e. using ATen Tensor library, that exposes PyTorch operations in C++11. This allows to replace all Lua code (Torch7) and avoid implementing two distinct kernels for CPU or CUDA.
ATen allows to write generic code that works on both devices.
More information in ATen [repo](https://github.com/zdevito/ATen).

To install this repo:

#0. Clone this repo:
---------------

```
git clone git@github.com:AAlguacil/fluidnet_cxx.git
```

#1. Clone ATen:
---------------

```
git clone --recurse-submodules -j8 git@github.com:zdevito/ATen.git
```

#2. Install ATen
---------------

From ATen [repo](https://github.com/zdevito/ATen).

TH/THC/THNN/THCUNN are provided (as git subtrees), so the repo is standalone. You will need a C++11 compiler, cmake, and the pyyaml python package.

```

# Install pyyaml used by python code generation to read API declarations

# macOS: if you don't have pip
sudo easy_install pip
# Ubuntu: if you don't have pip
apt-get -y install python-pip

# if you don't have pyyaml
sudo pip install pyyaml

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/where/you/want # specify your dest directory
# cmake .. -DUSE_CUDA=OFF  # for CPU only machines
make install
```

#4. Install OpenCV2 (temporary, must be removed and replaced with scientific plotter)
--------------

Find the releases [here](https://opencv.org/releases.html)

#5. Build and compile fluidnet_cxx
---------------

In main folder

```

mkdir build
cd build
cmake .. -DFLUID_TEST=OFF/ON # specify if you want to build tests
./simulate/fluid_sim # if test = OFF
./test/unit_tests # if test = ON
```




