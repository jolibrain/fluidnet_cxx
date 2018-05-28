#pragma once

#include <string>
#include <fstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>

#include <glob.h>  // glob(), globfree() to parse through a directory
#include <vector>
#include <string.h> //memset

#include "ATen/ATen.h"
#include "grid/bool_conversion.h"

// Load data from one .bin file generated in python with Manta
bool loadMantaFile
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
      return(false);  
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
  float velz[numel];
  // Ignore border pixels
  //nx -= 1;
  //ny -= 1;

  file.read(reinterpret_cast<char*>(&velx), sizeof(velx) );
  file.read(reinterpret_cast<char*>(&vely), sizeof(vely) );
  at::Tensor Ux = Tfloat.tensorFromBlob(velx, {numel});
  at::Tensor Uy = Tfloat.tensorFromBlob(vely, {numel});
  at::Tensor Uz;
  if (is3D) {
  //  nz -= 1;
    //float velz[numel];
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
  
  file.close();
  return true;
}

// Parse through a directory and store file names in a vector of strings
// https://stackoverflow.com/questions/8401777/simple-glob-in-c-on-unix-system

std::vector<std::string> globVector(const std::string& pattern){
    glob_t glob_result;
    std::memset(&glob_result, 0, sizeof(glob_result));
    
    int rtrn_val = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if(rtrn_val != 0) {
       globfree(&glob_result);
       std::string ss;
       ss = "glob() failed with return value " + std::to_string(rtrn_val);
       const char *css = ss.c_str();
       AT_ERROR(css);
    }

    std::vector<std::string> files;

    for(unsigned int i=0; i<glob_result.gl_pathc; ++i){
        files.push_back(std::string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}

// Funtion to call loadMantaFile on multiple files to concat them into a single batch.
void loadMantaBatch(std::string fn,
                    at::Tensor& p_out,
                    at::Tensor& U_out,
                    at::Tensor& flags_out,
                    at::Tensor& density_out,
                    bool& is3D)
{
   auto && Tundef = at::getType(at::Backend::Undefined, at::ScalarType::Undefined);

   if (p_out.type() != Tundef || U_out.type() != Tundef || density_out.type()
       != Tundef || flags_out.type() != Tundef) {
      AT_ERROR("Load Manta Batch: input tensors must be Undefined");
   }
   std::string path = "../test_data/b*_" + fn;
   std::vector<std::string> files = globVector(path);
   if (files.size() != 16){
     std::string ss;
     ss = "loadMantaBatch(" + fn + ") must have 16 files per batch";
     const char *css = ss.c_str();
     AT_ERROR(css);
   } 
   std::vector<at::Tensor> p;
   std::vector<at::Tensor> U;
   std::vector<at::Tensor> flags;
   std::vector<at::Tensor> density;
   
   for (auto const& file: files){
     at::Tensor curP;
     at::Tensor curU;
     at::Tensor curFlags;
     at::Tensor curDensity;
     bool curIs3D;
     bool succes = loadMantaFile(file, curP, curU, curFlags, curDensity, curIs3D);
     p.push_back(curP);
     U.push_back(curU);
     flags.push_back(curFlags);
     density.push_back(curDensity);
     is3D = curIs3D;
   }
    
   p_out = at::cat(p, 0);
   U_out = at::cat(U, 0);
   flags_out = at::cat(flags, 0);
   density_out = at::cat (density, 0);
}

// Make sure we load samples from file that aren't all the same, otherwise
// we're not really testing the batch dimension.
void assertNotAllEqual(at::Tensor t)
{
   if (t.size(0) == 1){
    return; // Only one sample, so it's correct.
   }
   
   at::Tensor first = t.select(0,0);
   at::Tensor others = t.select(0,10);
   if(fluid::toBool(others - at::max(at::abs(first.expand_as(others))) < 1e-5)){
     AT_ERROR("All samples are equal!");
   }
}
