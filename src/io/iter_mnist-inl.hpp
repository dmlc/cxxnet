#ifndef CXXNET_ITER_MNIST_INL_HPP_
#define CXXNET_ITER_MNIST_INL_HPP_
/*!
 * \file iter_mnist-inl.hpp
 * \brief iterator that takes mnist dataset
 * \author Tianqi Chen
 */
#include <mshadow/tensor.h>
#include "data.h"
#include "../utils/io.h"
#include "../utils/global_random.h"

namespace cxxnet {
class MNISTIterator: public IIterator<DataBatch> {
 public:
  MNISTIterator(void) {
    img_.dptr_ = NULL;
    mode_ = 1;
    inst_offset_ = 0;
    silent_ = 0;
    shuffle_ = 0;
  }
  virtual ~MNISTIterator(void) {
    if (img_.dptr_ != NULL) delete []img_.dptr_;
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "silent")) silent_ = atoi(val);
    if (!strcmp(name, "batch_size"))   batch_size_ = (index_t)atoi(val); 
    if (!strcmp(name, "input_flat"))   mode_ = atoi(val);
    if (!strcmp(name, "shuffle")) shuffle_ = atoi(val);
    if (!strcmp(name, "index_offset")) inst_offset_ = atoi(val);
    if (!strcmp(name, "path_img"))     path_img = val;
    if (!strcmp(name, "path_label"))   path_label = val;            
  }
  // intialize iterator loads data in
  virtual void Init(void) {
    this->LoadImage();
    this->LoadLabel();
    if (mode_ == 1) {
      out_.data.shape_ = mshadow::Shape4(batch_size_, 1, 1,img_.size(1) * img_.size(2));
    } else {
      out_.data.shape_ = mshadow::Shape4(batch_size_, 1, img_.size(1), img_.size(2));
    }
    out_.inst_index = NULL;
    out_.data.stride_ = out_.data.size(3);
    out_.batch_size = batch_size_;
    if (shuffle_) this->Shuffle();
    if (silent_ == 0) {
      mshadow::Shape<4> s = out_.data.shape_;
      printf("MNISTIterator: load %u images, shuffle=%d, shape=%u,%u,%u,%u\n", 
             (unsigned)img_.size(0), shuffle_, s[0], s[1], s[2], s[3]);
    }
  }
  virtual void BeforeFirst(void) {
    this->loc_ = 0;
  }
  virtual bool Next(void) {
    if (loc_ + batch_size_ <= img_.size(0)) {
      out_.data.dptr_ = img_[loc_].dptr_;
      out_.labels = &labels_[loc_];
      out_.inst_index = &inst_[loc_];
      loc_ += batch_size_;
      return true;
    } else{
      return false;
    }
  }
  virtual const DataBatch &Value(void) const {
    return out_;
  }
 private:
  inline void LoadImage(void) {
    utils::GzFile gzimg(path_img.c_str(), "rb");
    ReadInt(gzimg);
    int image_count = ReadInt(gzimg);
    int image_rows  = ReadInt(gzimg);
    int image_cols  = ReadInt(gzimg);
            
    img_.shape_ = mshadow::Shape3(image_count, image_rows, image_cols);
    img_.stride_ = img_.size(2);
            
    // allocate continuous memory
    img_.dptr_ = new float[img_.MSize()];
    for (int i = 0; i < image_count; ++i) {
      for (int j = 0; j < image_rows; ++j) {
        for (int k = 0; k < image_cols; ++k) {
          img_[i][j][k] = gzimg.ReadType<unsigned char>();
        }
      }
    }
    // normalize to 0-1
    img_ *= 1.0f / 256.0f;
  }        
  inline void LoadLabel(void) {
    utils::GzFile gzlabel(path_label.c_str(), "rb");
    ReadInt(gzlabel);
    int labels_count =ReadInt(gzlabel);

    labels_.resize(labels_count);
    for (int i = 0; i < labels_count; ++i) {
      labels_[i] = gzlabel.ReadType<unsigned char>();
      inst_.push_back((unsigned)i + inst_offset_);
    }
  }
  inline void Shuffle(void) {
    utils::Shuffle(inst_);
    std::vector<float> tmplabel(labels_.size());
    mshadow::TensorContainer<cpu,3> tmpimg(img_.shape_);
    for (size_t i = 0; i < inst_.size(); ++ i) {
      unsigned ridx = inst_[i] - inst_offset_;
      mshadow::Copy(tmpimg[i], img_[ridx]);
      tmplabel[i] = labels_[ridx];
    }
    // copy back
    mshadow::Copy(img_, tmpimg);
    labels_ = tmplabel;
  }
 private:
  inline static int ReadInt(utils::IStream &fi) {
    unsigned char buf[4];
    utils::Assert(fi.Read(buf, sizeof(buf)) == sizeof(buf), "Failed to read an int\n");
    return int(buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3]);
  }
 private:
  // silent
  int silent_;
  // path
  std::string path_img, path_label;
  // output 
  DataBatch out_;
  // whether do shuffle
  int shuffle_;
  // data mode
  int mode_;
  // current location
  index_t loc_;
  // batch size
  index_t batch_size_;
  // image content 
  mshadow::Tensor<cpu,3> img_;
  // label content
  std::vector<float> labels_;
  // instance index offset
  unsigned inst_offset_;
  // instance index
  std::vector<unsigned> inst_; 
}; //class MNISTIterator
}  // namespace cxxnet
#endif  // CXXNET_ITER_MNIST_INL_HPP_
