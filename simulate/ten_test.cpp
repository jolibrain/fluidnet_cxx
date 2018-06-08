#include <vector>
#include <sstream>

#include "ATen/ATen.h"

#include "grid/cell_type.h"
#include "../test/load_manta_data.h"
#include "../test/plot_utils.h"

typedef at::Tensor T;

T interpol(T& self, T& pos) {

  AT_ASSERT(pos.size(1) == 3, "Input pos must have 3 channels"); 

  bool is3D = (self.size(2) > 1);
  int bsz = pos.size(0);
  int d = pos.size(2);
  int h = pos.size(3);
  int w = pos.size(4);
  
  // 0.5 is defined as the center of the first cell as the scheme shows:
  //   |----x----|----x----|----x----|
  //  x=0  0.5   1   1.5   2   2.5   3
  T p = pos - 0.5;

  // Cast to integer, truncates towards 0.
  T pos0 = p.toType(at::kLong);
 
  T s1 = p.select(1,0) - pos0.select(1,0).toType(at::kFloat);
  T t1 = p.select(1,1) - pos0.select(1,1).toType(at::kFloat);
  T f1 = p.select(1,2) - pos0.select(1,2).toType(at::kFloat);
  T s0 = 1 - s1; 
  T t0 = 1 - t1;
  T f0 = 1 - f1;

  T x0 = pos0.select(1,0).clamp_(0, self.size(4) - 2);
  T y0 = pos0.select(1,1).clamp_(0, self.size(3) - 2);
  T z0 = pos0.select(1,2).clamp_(0, self.size(2) - 2);

  T b_idx = infer_type(x0).arange(0, bsz).view({bsz,1,1,1});
  b_idx = b_idx.expand({bsz,d,h,w});

  s1.clamp_(0, 1);
  t1.clamp_(0, 1);
  f1.clamp_(0, 1);
  s0.clamp_(0, 1);
  t0.clamp_(0, 1);
  f0.clamp_(0, 1);
  
  if (is3D) {
   T Ia= self.index({b_idx, {}, z0  , y0  , x0  }).squeeze(4).unsqueeze(1);
   T Ib= self.index({b_idx, {}, z0  , y0+1, x0  }).squeeze(4).unsqueeze(1);
   T Ic= self.index({b_idx, {}, z0  , y0  , x0+1}).squeeze(4).unsqueeze(1);
   T Id= self.index({b_idx, {}, z0  , y0+1, x0+1}).squeeze(4).unsqueeze(1);
   T Ie= self.index({b_idx, {}, z0+1, y0  , x0  }).squeeze(4).unsqueeze(1);
   T If= self.index({b_idx, {}, z0+1, y0+1, x0  }).squeeze(4).unsqueeze(1);
   T Ig= self.index({b_idx, {}, z0+1, y0  , x0+1}).squeeze(4).unsqueeze(1);
   T Ih= self.index({b_idx, {}, z0+1, y0+1, x0+1}).squeeze(4).unsqueeze(1);

    return ( ((Ia*t0 + Ib*t1)*s0 + (Ic*t0 + Id*t1)*s1)*f0 +
             ((Ie*t0 + If*t1)*s0 + (Ig*t0 + Ih*t1)*s1)*f1 ); 
  } else {
   T Ia= self.index({b_idx, {}, z0+1, y0  , x0  }).squeeze(4).unsqueeze(1);
   T Ib= self.index({b_idx, {}, z0+1, y0+1, x0  }).squeeze(4).unsqueeze(1);
   T Ic= self.index({b_idx, {}, z0+1, y0  , x0+1}).squeeze(4).unsqueeze(1);
   T Id= self.index({b_idx, {}, z0+1, y0+1, x0+1}).squeeze(4).unsqueeze(1);

    return ( (Ia*t0 + Ib*t1)*s0 + (Ic*t0 + Id*t1)*s1 );
  }
}

void interpol1DWithFluid(
    const T& val_a, const T& is_fluid_a,
    const T& val_b, const T& is_fluid_b,
    const T& t_a, const T& t_b,
    T& is_fluid_ab, T& val_ab) {

  T m0 = is_fluid_a.eq(0).__and__(is_fluid_b.eq(0));
  T m1 = is_fluid_a.eq(0).__and__(m0.eq(0));
  T m2 = is_fluid_b.eq(0).__and__(m1.eq(0)).__and__(m0.eq(0));
  T m3 = 1 - (m0.__or__(m1).__or__(m2));

  val_ab = val_a;
  val_ab = val_ab.masked_fill_(m0, 0);
  val_ab = val_ab.masked_scatter_(m1, val_b.masked_select(m1));
  val_ab = val_ab.masked_scatter_(m2, val_a.masked_select(m2));
  val_ab = val_ab.masked_scatter_(m3, (val_a*t_a + val_b*t_b).masked_select(m3));

  is_fluid_ab = m0.eq(0);
}


void interpolWithFluid(T& self, T& flags, T& pos) {

  AT_ASSERT(pos.size(1) == 3, "Input pos must have 3 channels");

  bool is3D = (self.size(2) > 1);
  int bsz = pos.size(0);
  int d = pos.size(2);
  int h = pos.size(3);
  int w = pos.size(4);
  // 0.5 is defined as the center of the first cell as the scheme shows:
  //   |----x----|----x----|----x----|
  //  x=0  0.5   1   1.5   2   2.5   3
  T p = pos - 0.5;

  // Cast to integer, truncates towards 0.
  T pos0 = p.toType(at::kLong);

  T s1 = p.select(1,0) - pos0.select(1,0).toType(at::kFloat);
  T t1 = p.select(1,1) - pos0.select(1,1).toType(at::kFloat);
  T f1 = p.select(1,2) - pos0.select(1,2).toType(at::kFloat);
  T s0 = 1 - s1;
  T t0 = 1 - t1;
  T f0 = 1 - f1;

  T x0 = pos0.select(1,0).clamp_(0, self.size(4) - 2);
  T y0 = pos0.select(1,1).clamp_(0, self.size(3) - 2);
  T z0 = pos0.select(1,2).clamp_(0, self.size(2) - 2);

  T b_idx = infer_type(x0).arange(0, bsz).view({bsz,1,1,1});
  b_idx = b_idx.expand({bsz,d,h,w});

  s1.clamp_(0, 1);
  t1.clamp_(0, 1);
  f1.clamp_(0, 1);
  s0.clamp_(0, 1);
  t0.clamp_(0, 1);
  f0.clamp_(0, 1);

  if (is3D) {
   // val_ab = data(xi, yi, zi, 0, b) * t0 +
   //          data(xi, yi + 1, zi, 0, b) * t1
   T Ia = self.index({b_idx, {}, z0  , y0  , x0  }).squeeze(4).unsqueeze(1);
   T Ib = self.index({b_idx, {}, z0  , y0+1, x0  }).squeeze(4).unsqueeze(1);

   T is_fluid_a = flags.index({b_idx, {}, z0  , y0  , x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);
   T is_fluid_b = flags.index({b_idx, {}, z0  , y0+1, x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);

   T Iab;
   T is_fluid_ab; 
   interpol1DWithFluid(Ia, is_fluid_a, Ib, is_fluid_b, t0, t1, is_fluid_ab, Iab); 

   // val_cd = data(xi + 1, yi, zi, 0, b) * t0 +
   //          data(xi + 1, yi + 1, zi, 0, b) * t1
   T Ic = self.index({b_idx, {}, z0  , y0  , x0+1}).squeeze(4).unsqueeze(1);
   T Id = self.index({b_idx, {}, z0  , y0+1, x0+1}).squeeze(4).unsqueeze(1);

   T is_fluid_c = flags.index({b_idx, {}, z0  , y0  , x0+1}).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);
   T is_fluid_d = flags.index({b_idx, {}, z0  , y0+1, x0+1}).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);

   T Icd;
   T is_fluid_cd; 
   interpol1DWithFluid(Ic, is_fluid_c, Id, is_fluid_d, t0, t1, is_fluid_cd, Icd); 

   // val_ef = data(xi, yi, zi + 1, 0, b) * t0 +
   //          data(xi, yi + 1, zi + 1, 0, b) * t1
   T Ie = self.index({b_idx, {}, z0+1, y0  , x0  }).squeeze(4).unsqueeze(1);
   T If = self.index({b_idx, {}, z0+1, y0+1, x0  }).squeeze(4).unsqueeze(1);

   T is_fluid_c = flags.index({b_idx, {}, z0+1, y0  , x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);
   T is_fluid_d = flags.index({b_idx, {}, z0+1, y0+1, x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);

   T Ief;
   T is_fluid_ef;
   interpol1DWithFluid(Ie, is_fluid_e, If, is_fluid_f, t0, t1, is_fluid_ef, Ief);

   // val_gh = data(xi + 1, yi, zi + 1, 0, b) * t0 +
   //          data(xi + 1, yi + 1, zi + 1, 0, b) * t1
   T Ig = self.index({b_idx, {}, z0+1, y0  , x0+1}).squeeze(4).unsqueeze(1);
   T Ih = self.index({b_idx, {}, z0+1, y0+1, x0+1}).squeeze(4).unsqueeze(1);

   T is_fluid_g = flags.index({b_idx, {}, z0+1, y0  , x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);
   T is_fluid_h = flags.index({b_idx, {}, z0+1, y0+1, x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);

   T Igh;
   T is_fluid_gh;
   interpol1DWithFluid(Ig, is_fluid_g, Ih, is_fluid_h, t0, t1, is_fluid_gh, Igh);

   // val_abcd = val_ab * s0 + val_cd * s1
   T Iabcd;
   T is_fluid_abcd;
   interpol1DWithFluid(Iab, is_fluid_ab, Icd, is_fluid_cd, s0, s1, is_fluid_abcd, Iabcd);

   // val_efgh = val_ef * s0 + val_gh * s1
   T Iefgh;
   T is_fluid_efgh;
   interpol1DWithFluid(Ief, is_fluid_ef, Igh, is_fluid_gh, s0, s1, is_fluid_efgh, Iefgh);
   
   // val = val_abcd * f0 + val_efgh * f1
   T Ival;
   T is_fluid;
   interpol1DWithFluid(Iabcd, is_fluid_abcd, Iefgh, is_fluid_efgh, f0, f1,
                             is_fluid, Ival);

   T no_fluid = is_fluid.eq(0);
   Ival = Ival.masked_scatter_(no_fluid, interpol(self, pos).masked_select(no_fluid));
   return Ival;
  } else {
   // val_ab = data(xi, yi, 0, 0, b) * t0 +
   //          data(xi, yi + 1, 0, 0, b) * t1
   T Ia = self.index({b_idx, {}, z0+1, y0  , x0  }).squeeze(4).unsqueeze(1);
   T Ib = self.index({b_idx, {}, z0+1, y0+1, x0  }).squeeze(4).unsqueeze(1);

   T is_fluid_a = flags.index({b_idx, {}, z0+1, y0  , x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);
   T is_fluid_b = flags.index({b_idx, {}, z0+1, y0+1, x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);

   T Iab;
   T is_fluid_ab;
   interpol1DWithFluid(Ia, is_fluid_a, Ib, is_fluid_b, t0, t1, is_fluid_ab, Iab); 

   // val_cd = data(xi + 1, yi, 0, 0, b) * t0 +
   //          data(xi + 1, yi + 1, 0, 0, b) * t1
   T Ic = self.index({b_idx, {}, z0+1 , y0  , x0+1}).squeeze(4).unsqueeze(1);
   T Id = self.index({b_idx, {}, z0+1, y0+1, x0+1}).squeeze(4).unsqueeze(1);

   T is_fluid_c = flags.index({b_idx, {}, z0+1, y0  , x0+1}).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);
   T is_fluid_d = flags.index({b_idx, {}, z0+1, y0+1, x0+1}).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);

   T Icd;
   T is_fluid_cd;
   interpol1DWithFluid(Ic, is_fluid_c, Id, is_fluid_d, t0, t1, is_fluid_cd, Icd);

   // val = val_ab * s0 + val_cd * s1
   T Ival;
   T is_fluid;
   interpol1DWithFluid(Iab, is_fluid_ab, Icd, is_fluid_cd, s0, s1, is_fluid, Ival);

   T no_fluid = is_fluid.eq(0);
   Ival = Ival.masked_scatter_(no_fluid, interpol(self, pos).masked_select(no_fluid));
   return Ival;
  }
}

int main() {

  auto && Tfloat = CPU(at::kFloat);

  int dim = 2;
 
      std::string fn = std::to_string(dim) + "d_gravity.bin";
      at::Tensor undef1;
      at::Tensor U;
      at::Tensor flags;
      at::Tensor density;
      bool is3D ;
      loadMantaBatch(fn, undef1, U, flags, density, is3D);
      assertNotAllEqual(U);
      assertNotAllEqual(flags);

      AT_ASSERT(is3D == (dim == 3), "Failed assert is3D");
      fn = std::to_string(dim) + "d_correctVelocity.bin";
      at::Tensor undef2;
      at::Tensor pressure;
      at::Tensor UManta;
      at::Tensor flagsManta;
      loadMantaBatch(fn, pressure, UManta, flagsManta, undef2, is3D);
      AT_ASSERT(is3D == (dim == 3), "Failed assert is3D");

      AT_ASSERT(flags.equal(flagsManta), "Flags changed!");
      AT_ASSERT(at::Scalar(at::max(at::abs(U - UManta))).toFloat() > 1e-5, "No velocities changed in Manta!");

//  int b = flags.size(0);
//  int d = flags.size(2);
//  int h = flags.size(3);
//  int w = flags.size(4);
//

  int b = 1;
  int d = 1;
  int h = 4;
  int w = 4;

//  is3D = true;
//  at::Tensor index_x = CPU(at::kInt).arange(0, h).view({h}).expand({d,h,w});
//  at::Tensor index_y = CPU(at::kInt).arange(0, w).view({w, 1}).expand({d,h,w});
//  at::Tensor index_z;
//  if (is3D) {
//     index_z = CPU(at::kInt).arange(0, d).view({d, 1 , 1}).expand({d,h,w});
//  }
//  at::Tensor index_ten;
//
//  if (!is3D) {
//    index_ten = at::stack({index_x, index_y}, 0).view({1,2,d,h,w});
//  }
//  else { 
//    index_ten = at::stack({index_x, index_y, index_z}, 0).view({1,3,d,h,w});
//  }
//
//  index_ten = index_ten.expand_as(pos);
  //index_ten.expand_as(pos);
  // std::cout << pos << std::endl;
   
  T im = Tfloat.rand({b,1,d,h,w}) * 255;
  T flags_rand = CPU(at::kInt).randint(2, {b, 1, d,h,w}) + 1;
  T self = im;
  T pos = CPU(at::kFloat).rand({b,3,d,h,w}) * (h);
  // 0.5 is defined as the center of the first cell as the scheme shows:
  //   |----x----|----x----|----x----|
  //  x=0  0.5   1   1.5   2   2.5   3
  T p = pos - 0.5;

  // Cast to integer, truncates towards 0.
  T x0 = p.toType(at::kLong);

  T s1 = p.select(1,0) - x0.select(1,0).toType(at::kFloat);
  T t1 = p.select(1,1) - x0.select(1,1).toType(at::kFloat);
  T f1 = p.select(1,2) - x0.select(1,2).toType(at::kFloat);
  T s0 = 1 - s1;
  T t0 = 1 - t1;
  T f0 = 1 - f1;

  x0.select(1,0).clamp_(0, self.size(4) - 2);
  x0.select(1,1).clamp_(0, self.size(3) - 2);
  x0.select(1,2).clamp_(0, self.size(2) - 2);

  T second_tensor = CPU(at::kLong).arange(0, x0.size(0)).view({x0.size(0),1,1,1});
  second_tensor = second_tensor.expand({x0.size(0), x0.size(2), x0.size(3), x0.size(4)});
  s1.clamp_(0, 1);
  t1.clamp_(0, 1);
  f1.clamp_(0, 1);
  s0.clamp_(0, 1);
  t0.clamp_(0, 1);
  f0.clamp_(0, 1);

  T Ia = self.index({second_tensor,{},{}, x0.select(1,1)  , x0.select(1,0)}).squeeze(4).squeeze(4).unsqueeze(1);//;//.transpose(3,4).transpose(2,3).transpose(1,2);
  interpolWithFluid(im, flags_rand, pos);
// 
}
