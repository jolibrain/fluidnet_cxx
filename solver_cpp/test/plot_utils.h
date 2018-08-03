#pragma once

#include <string.h>

#include "ATen/ATen.h"

#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void plotTensor2D(at::Tensor Ten, int img_w, int img_h, std::string wname)
{
  at::Tensor T_min = at::min(Ten);
  at::Tensor T_max = at::max(Ten);
  at::Tensor T_scaled = (Ten - T_min) / (T_max - T_min);

    
  int height = Ten.size(Ten.dim()-2);
  int width = Ten.size(Ten.dim()-1);

  cv::Mat img_tensor(width, height, CV_32FC1);
  cv::Mat img_rsz(img_w, img_h, CV_8UC1);
  cv::Mat img8uc;
  cv::Mat img_col;
  
  if (Ten.dim() == 5) {  
    auto T_a = T_scaled.accessor<float,5>();
    for (int h = 0; h < height; h++) {
       for (int w = 0; w < width; w++) {
          
          img_tensor.at<float>(w,h) = T_a[0][0][0][h][w];
       }
    } }
  else if (Ten.dim() == 4) {
    auto T_a = T_scaled.accessor<float,4>();
    for (int h = 0; h < height; h++) {
       for (int w = 0; w < width; w++) {
          img_tensor.at<float>(w,h) = T_a[0][0][h][w];
       }
    } }
  else if (Ten.dim() == 3) {
    auto T_a = T_scaled.accessor<float,3>();
    for (int h = 0; h < height; h++) {
       for (int w = 0; w < width; w++) {
          img_tensor.at<float>(w,h) = T_a[0][h][w];
       }
    } }
  else if (Ten.dim() == 2) {
    auto T_a = T_scaled.accessor<float,2>();
    for (int h = 0; h < height; h++) {
       for (int w = 0; w < width; w++) {
          img_tensor.at<float>(w,h) = T_a[h][w];
       }
    } }
  else {
    AT_ERROR("Plot: error in input dimension!");
  }

  double min;
  double max;
  cv::minMaxLoc(img_tensor, &min, &max);
  
  //CV_8UC1 conversion
  img_tensor.convertTo(img8uc,CV_8UC1, 255/(max-min),-255*min / (max - min));
  resize(img8uc, img_rsz, img_rsz.size(), 0, 0, cv::INTER_NEAREST);
  applyColorMap(img_rsz, img_col, cv::COLORMAP_JET);
  std::string out_name = wname + ".png";
  cv::imwrite(out_name, img_col);
}

