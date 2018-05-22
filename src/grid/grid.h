#pragma once

#include <iostream>
#include <sstream>

#include "ATen/ATen.h"
#include "cell_type.h"
#include "bool_conversion.h"

namespace fluid {

  typedef at::Tensor T;
  
  class GridBase {
  public:
    explicit GridBase(T & grid, bool is_3d);
  
    int64_t nbatch() const { return tensor_.size(0); }
    int64_t nchan() const  { return tensor_.size(1); }
    int64_t zsize() const  { return tensor_.size(2); }
    int64_t ysize() const  { return tensor_.size(3); }
    int64_t xsize() const  { return tensor_.size(4); }
  
    int64_t numel() const { return xsize() * ysize() * zsize() * nchan()
  			         * nbatch(); }  
  
    bool is_3d() const { return is_3d_; }
    
    T getSize() const;  
    at::Backend getBackend() const { return bckd;}
    at::ScalarType getGridType() const { return real;}
     
    float getDx() const;
  
    bool isInBounds(const T& p, int bnd) const;
  
    friend std::ostream& operator<<(std::ostream& os, const GridBase& outGrid);
    
  private:
    T tensor_;
    const bool is_3d_; 
  
  protected:
    at::Backend bckd;
    at::ScalarType real;
    
    T data(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) const {
      return tensor_[b][c][k][j][i];
    }
  
    T data(T i, T j, T k, T c, T b) const {
      return tensor_[b][c][k][j][i];
    }
  
    T data(const T& pos, T c, T b) const {
      return tensor_[b][c][pos[2]][pos[1]][pos[0]];
    }
    
    void buildIndex(T& posi, T& stf0, T& stf1,
                    const T& pos) const; 
  };
  
  class FlagGrid : public GridBase {
  public:
    explicit FlagGrid(T & grid, bool is_3d);
  
    T operator()(int32_t i, int32_t j, int32_t k, int32_t b) const {
      return data(i, j, k, 0, b);
    }
  
    T operator()(T i, T j, T k, T b) const {
      return data(i, j, k, getType(bckd,at::kInt).scalarTensor(0), b);
    }
    
    bool isFluid(int32_t i, int32_t j, int32_t k, int32_t b) const {
      return toBool(data(i, j, k, 0, b).eq(TypeFluid) );
    }

    bool isFluid(T i, T j, T k, T b) const {
      return toBool(data(i, j, k, getType(bckd,at::kInt).scalarTensor(0), b).eq(TypeFluid) );
    }

    bool isFluid(const T& pos, T b) const {
      return toBool(data(pos, getType(bckd,at::kInt).scalarTensor(0), b).eq(TypeFluid) );
    }
  
    bool isObstacle(int32_t i, int32_t j, int32_t k, int32_t b) const {
      return toBool(data(i, j, k, 0, b).eq(TypeObstacle) );
    }
  
    bool isObstacle(const T& pos, T b) const {
      return toBool(data(pos, getType(bckd,at::kInt).scalarTensor(0), b).eq(TypeObstacle) );
    }
   
    bool isStick(int32_t i, int32_t j, int32_t k, int32_t b) const {
      return toBool(data(i, j, k, 0, b).eq(TypeStick) );
    }
    
    bool isEmpty(int32_t i, int32_t j, int32_t k, int32_t b) const {
      return toBool(data(i, j, k, 0, b).eq(TypeEmpty) );
    }
  
    bool isOutflow(int32_t i, int32_t j, int32_t k, int32_t b) const {
      return toBool(data(i, j, k, 0, b).eq(TypeOutflow) );
    }
  
    bool isOutOfDomain(int32_t i, int32_t j, int32_t k, int32_t b) const {
      return (i < 0 || i >= xsize() || j < 0 || j >= ysize() || k < 0 ||
              k >= zsize() || b < 0 || b >= nbatch());
    }

    bool isOutOfDomain(T i, T j, T k, T b) const {
      return toBool( (i < 0).__or__(i >= xsize()).__or__(j < 0).__or__(j >= ysize()).__or__(k < 0)
             .__or__(k >= zsize()).__or__(b < 0).__or__(b >= nbatch()) );
    }
 };
  
  class RealGrid : public GridBase {
  public:
    explicit RealGrid(T & grid, bool is_3d);
  
    T operator()(int32_t i, int32_t j, int32_t k, int32_t b) const {
      return data(i, j, k, 0, b);
    };
  
    T operator()(T i, T j, T k, T b) const {
      return data(i, j, k, getType(bckd,at::kInt).scalarTensor(0), b);
    }
    
    T getInterpolatedHi(const T& pos, int32_t order, T b) const;
    
    T getInterpolatedWithFluidHi(const FlagGrid& flag, const T& pos,
                                    int32_t order, T b) const;
  
    T interpol(const T& pos, T b) const;
    T interpolWithFluid(const FlagGrid& flag,
                           const T& pos, T ibatch) const;
  private:
    // Interpol1DWithFluid will return:
    // 1. is_fluid = false if a and b are not fluid.
    // 2. is_fluid = true and data(a) if b is not fluid.
    // 3. is_fluid = true and data(b) if a is not fluid.
    // 4. The linear interpolation between data(a) and data(b) if both are fluid.
    void interpol1DWithFluid(const T val_a, const bool is_fluid_a,
                                    const T val_b, const bool is_fluid_b,
                                    const T t_a,   const T t_b,
                                    bool* is_fluid_ab, T& val_ab) const;
  };
  
  class MACGrid : public GridBase {
  public:
    explicit MACGrid(T & grid, bool is_3d);
  
    // Note: as per other functions, we DO NOT bounds check getCentered. You must
    // not call this method on the edge of the simulation domain.
    const T getCentered(T i, T j, T k, T b) const;
   
    const T getCentered(int32_t i, int32_t j, int32_t k, int32_t b) const{
      return getCentered (getType(bckd, at::kInt).scalarTensor(i),
                          getType(bckd, at::kInt).scalarTensor(j),
                          getType(bckd, at::kInt).scalarTensor(k),
                          getType(bckd, at::kInt).scalarTensor(b));
  }
  
    const T getCentered(const T vec, T b) {
      return getCentered(vec[0], vec[1], vec[2], b);
    }
  
    const T operator()(int32_t i, int32_t j,
                                    int32_t k, int32_t b) const {
      
      T ret = getType(bckd, real).tensor({3});
      ret[0] = data(i, j, k, 0, b);
      ret[1] = data(i, j, k, 1, b);
      ret[2] = !is_3d() ? getType(bckd, real).scalarTensor(0) : data(i, j, k, 2, b);
      return ret;
    }
  
    const T operator()(T i, T j, T k, T b) const {
  
      T ret = getType(bckd, real).tensor({3});
      T chan = getType(bckd, at::kInt).arange(3);
  
      ret[0] = data(i, j, k, chan[0], b);
      ret[1] = data(i, j, k, chan[1], b);
      ret[2] = !is_3d() ? getType(bckd, real).scalarTensor(0) : data(i, j, k, chan[2], b);
      return ret;
    }
  
    T operator()(T i, T j, T k, T c, T b) const {
      return data(i, j, k, c, b);
    }
  
    T operator()(int32_t i, int32_t j, int32_t k, 
                 int32_t c, int32_t b) const {
      return data(i, j, k, c, b);
    }
  
    // setSafe will ignore the 3rd component of the input vector if the
    // MACGrid is 2D.
    void setSafe(T i, T j, T k, T b,
                 const T& val);
    
    void setSafe(int32_t i, int32_t j, int32_t k, int32_t b, const T& val) {
      return setSafe(getType(bckd, at::kInt).scalarTensor(i),
                     getType(bckd, at::kInt).scalarTensor(j),
                     getType(bckd, at::kInt).scalarTensor(k),
                     getType(bckd, at::kInt).scalarTensor(b),
                     val);
    }
  
    T getAtMACX(int32_t i, int32_t j, int32_t k, int32_t b) const;
    T getAtMACY(int32_t i, int32_t j, int32_t k, int32_t b) const;
    T getAtMACZ(int32_t i, int32_t j, int32_t k, int32_t b) const;
  
    T getInterpolatedComponentHi(const T& pos,
                                    int32_t order, T c, T b) const;
  private:
    T interpolComponent(const T& pos, T c, T b) const;
  };
  
  class VecGrid : public GridBase {
  public:
    explicit VecGrid(T & grid, bool is_3d);
  
      T operator()(int32_t i, int32_t j,
                                    int32_t k, int32_t b) const {
  
      T ret = getType(bckd, real).tensor({3});
      ret[0] = data(i, j, k, 0, b);
      ret[1] = data(i, j, k, 1, b);
      ret[2] = !is_3d() ? getType(bckd, real).scalarTensor(0) : data(i, j, k, 2, b);
      return ret;
    }
  
      T operator()(T i, T j, T k, T b) const {
  
      T ret = getType(bckd, real).tensor({3});
      T chan = getType(bckd, at::kInt).arange(3);
  
      ret[0] = data(i, j, k, chan[0], b);
      ret[1] = data(i, j, k, chan[1], b);
      ret[2] = !is_3d() ? getType(bckd, real).scalarTensor(0) : data(i, j, k, chan[2], b);
      return ret;
    }
    
    T operator()(T i, T j, T k, T c, T b) const {
      return data(i, j, k, c, b);
    }
  
    T operator()(int32_t i, int32_t j, int32_t k,
                 int32_t c, int32_t b) const {
      return data(i, j, k, c, b);
    }
  
    // setSafe will ignore the 3rd component of the input vector if the
    // VecGrid is 2D, but check that it is non-zero.
    void setSafe(int32_t i, int32_t j, int32_t k, int32_t b,
                 const T& val);
    // set will ignore the 3rd component of the input vector if the VecGrid is 2D
    // and it will NOT check that the component is non-zero.
    void set(int32_t i, int32_t j, int32_t k, int32_t b,
             const T& val);
  
    // Note: you CANNOT call curl on the border of the grid (if you do then
    // the data(...) calls will throw an error.
    // Also note that curl in 2D is a scalar, but we will return a vector anyway
    // with the scalar value being in the 3rd dim.
    T curl(int32_t i, int32_t j, int32_t k, int32_t b);
  };

} // namespace fluid
