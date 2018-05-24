#pragma once

#include <string>
#include <fstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>

#include "ATen/ATen.h"

bool load_manta_file
(
  std::string fname,
  at::Tensor& p,
  at::Tensor& U,
  at::Tensor& flags,
  at::Tensor& density,
  bool& is3D
)
{
  // Load files in CPU only!
  auto && Tfloat = CPU(at::kFloat);
  auto && Tint   = CPU(at::kInt);
  auto && Tundef = at::getType(at::Backend::Undefined, at::ScalarType::Undefined);

  if (p.type() != Tundef || U.type() != Tundef || density.type()
      != Tundef || flags.type() != Tundef) {
     AT_ERROR("Load Manta File: input tensors must be Undefined");
  }
  
  std::fstream file;
  file.open(fname, std::ios::in | std::ios::binary);
  
  if(file.fail()) {
      std::cout << "Unable to open the data file!!!" <<std::endl;
      std::cout << "Please make sure your bin file is in the same location as the program!"<< std::endl;
      //std::system("PAUSE");
      return(1);  
  }
  
  int transpose = 0;
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int is3D_ = 0;
  file.read(reinterpret_cast<char*>(&transpose), sizeof(int) );
  file.read(reinterpret_cast<char*>(&nx), sizeof(int) );
  file.read(reinterpret_cast<char*>(&ny), sizeof(int) );
  file.read(reinterpret_cast<char*>(&nz), sizeof(int) );
  file.read(reinterpret_cast<char*>(&is3D_), sizeof(int) );
  is3D = (is3D_ == 1);
  const int numel = nx*ny*nz;
  
  float velx[numel];
  float vely[numel];
  float* velz;
  file.read(reinterpret_cast<char*>(&velx), sizeof(velx) );
  file.read(reinterpret_cast<char*>(&vely), sizeof(vely) );
  at::Tensor Ux = Tfloat.tensorFromBlob(velx, {numel});
  at::Tensor Uy = Tfloat.tensorFromBlob(vely, {numel});
  at::Tensor Uz;
  if (is3D) {
    velz = new float[numel];
    file.read(reinterpret_cast<char*>(&velz), sizeof(velz) );
    Uz = Tfloat.tensorFromBlob(velz, {numel});
  }
  
  float pres[numel];
  file.read(reinterpret_cast<char*>(&pres), sizeof(pres) );
  p = Tfloat.tensorFromBlob(pres, {numel});
  
  int flagIN[numel];
  file.read(reinterpret_cast<char*>(&flagIN), sizeof(flagIN) );
  flags = Tint.tensorFromBlob(flagIN, {numel});
  flags = flags.toType(Tfloat);
  
  float rho[numel];
  file.read(reinterpret_cast<char*>(&rho), sizeof(rho) );
  density = Tfloat.tensorFromBlob(rho, {numel});
  
  Ux.resize_({1, 1, nz, ny, nx});
  Uy.resize_({1, 1, nz, ny, nx});
  if (is3D) {
    Uz.resize_({1, 1, nz, ny, nx});
  }
  p.resize_({1, 1, nz, ny, nx});
  flags.resize_({1, 1, nz, ny, nx});
  density.resize_({1, 1, nz, ny, nx});
  
  if (is3D) {
    U = at::cat({Ux, Uy, Uz}, 1).contiguous();
  } 
  else{ 
    U = at::cat({Ux, Uy}, 1).contiguous();
  }
  
  if (is3D) {
    delete[] velz;
  }
  file.close();
  return true;
}

