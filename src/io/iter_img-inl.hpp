#ifndef CXXNET_ITER_IMG_INL_HPP
#define CXXNET_ITER_IMG_INL_HPP
#pragma once
/*!
 * \file iter_img-inl.hpp
 * \brief implementation of image iterator
 * \author Tianqi Chen, Naiyan Wang
 */
// use opencv for image loading
#include "data.h"
#include <mshadow/tensor.h>
#include <opencv2/opencv.hpp>

namespace cxxnet{
  /*! \brief simple image iterator that only loads data instance */
class ImageIterator : public IIterator< DataInst >{
public:
  ImageIterator(void) {
    img_.set_pad(false);
    fplst_ = NULL;
    silent_ = 0;
    path_imgdir_ = "";
    path_imglst_ = "img.lst";
    shuffle_ = 0;
    data_index_ = 0;
    label_width_ = 1;
  }
  virtual ~ImageIterator(void) {
    if(fplst_ != NULL) fclose(fplst_);
  }
  virtual void SetParam(const char *name, const char *val) {
    if(!strcmp(name, "image_list"))  path_imglst_ = val;
    if(!strcmp(name, "image_root"))   path_imgdir_ = val;
    if(!strcmp(name, "silent"  ))  silent_ = atoi(val);
    if(!strcmp(name, "shuffle"  ))  shuffle_ = atoi(val);
    if(!strcmp(name, "label_width"  ))  label_width_ = atoi(val);
  }
  virtual void Init(void) {
    fplst_  = utils::FopenCheck(path_imglst_.c_str(), "r");
    if(silent_ == 0) {
      printf("ImageIterator:image_list=%s\n", path_imglst_.c_str());
    }
    unsigned index;
    while (fscanf(fplst_, "%u", &index) == 1) {
      index_list_.push_back(index);
      mshadow::Tensor<cpu, 1> label = mshadow::NewTensor<cpu>(mshadow::Shape1(label_width_), 0.0f);
      for (int i = 0; i < label_width_; ++i) {
        float tmp;
        utils::Check(fscanf(fplst_, "%f", &tmp) == 1,
               "ImageList format:label_width=%d but only have %d labels per line",
               label_width_, i);
        labels_.push_back(tmp);
      }
      char name[256];
      utils::Assert(fscanf(fplst_, "%s\n", name) == 1, "ImageList: no file name");
      filenames_.push_back(name);
    }
    for (size_t i = 0; i < index_list_.size(); ++i) {
      order_.push_back(i);
    }
    this->BeforeFirst();
  }
  virtual void BeforeFirst(void) {
    data_index_ = 0;
    if (shuffle_) {
      std::random_shuffle(order_.begin(), order_.end());
    }
  }
  virtual bool Next(void) {
    if (data_index_ < static_cast<int>(order_.size())) {
      size_t index = order_[data_index_];
      if (path_imgdir_.length() == 0) {
        LoadImage(img_, out_, filenames_[index].c_str());
      } else {
        char sname[256];
        sprintf(sname, "%s%s", path_imgdir_.c_str(), filenames_[index].c_str());
        LoadImage(img_, out_, sname);
      }
      out_.index = index_list_[index];
      mshadow::Tensor<cpu, 1> label_(&(labels_[0]) + label_width_ * index,
        mshadow::Shape1(label_width_));
      out_.label = label_;
      ++data_index_;
      return true;
    }
    return false;
  }
  virtual const DataInst &Value(void) const{
    return out_;
  }
protected:
  inline static void LoadImage(mshadow::TensorContainer<cpu,3> &img, 
          DataInst &out,
          const char *fname) {
    cv::Mat res = cv::imread(fname);
    utils::Assert(res.data != NULL, "LoadImage: Reading image %s failed.\n", fname);
    img.Resize(mshadow::Shape3(3, res.rows, res.cols));
    for(index_t y = 0; y < img.size(1); ++y) {
      for(index_t x = 0; x < img.size(2); ++x) {
        cv::Vec3b bgr = res.at<cv::Vec3b>(y, x);
        // store in RGB order
        img[2][y][x] = bgr[0];
        img[1][y][x] = bgr[1];
        img[0][y][x] = bgr[2];
      }
    }
    out.data = img;
    // free memory
    res.release();
  }
protected:
  // output data
  DataInst out_;
  // silent
  int silent_;
  // file pointer to list file, information file
  FILE *fplst_;
  // prefix path of image folder, path to input lst, format: imageid label path
  std::string path_imgdir_, path_imglst_;
  // temp storage for image
  mshadow::TensorContainer<cpu, 3> img_;
  // whether the data will be shuffled in each epoch
  int shuffle_;
  // denotes the number of labels
  int label_width_;
  // denotes the current data index
  int data_index_;
  // stores the reading orders
  std::vector<int> order_;
  // stores the labels of data
  std::vector<float> labels_;
  // stores the file names of the images
  std::vector<std::string> filenames_;
  // stores the index list of images
  std::vector<int> index_list_;
  };
};
#endif
