#ifndef CXXNET_MSHADOW_ITER_INL_HPP
#define CXXNET_MSHADOW_ITER_INL_HPP
#pragma once

#include "mshadow/tensor_container.h"
#include "cxxnet_data.h"
#include "../utils/cxxnet_io_utils.h"
#include "../utils/cxxnet_global_random.h"

namespace cxxnet {
// load from mshadow binary data
class MShadowIterator: public IIterator<DataInst> {
public:
  MShadowIterator(void) {
    img_.dptr = NULL;
    labels_.dptr = NULL;
    mode_ = 1;
    inst_offset_ = 0;
    silent_ = 0;
    shuffle_ = 0;
  }
  virtual ~MShadowIterator(void) {
    if (img_.dptr == NULL) {
      mshadow::FreeSpace(img_);
      mshadow::FreeSpace(labels_);
    }
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "silent"))       silent_ = atoi(val);
    if (!strcmp(name, "shuffle"))      shuffle_ = atoi(val);
    if (!strcmp(name, "index_offset")) inst_offset_ = atoi(val);
    if (!strcmp(name, "path_img"))     path_img_ = val;
    if (!strcmp(name, "path_label"))   path_label_ = val;
    if (!strcmp(name, "type")) {
      if (!strcmp(val, "mshadow")) type_ = 0;
      else if (!strcmp(val, "mnist")) type_ = 1;
      else if (!strcmp(val, "npy")) type_ = 2;
    }
  }
  // intialize iterator loads data in
  virtual void Init(void) {
    if (type_ == 0) {
      this->LoadImageMShadow();
      this->LoadLabelMShadow();
    } else if (type_ == 1) {
      this->LoadImageMNIST();
      this->LoadLabelMNIST();
    }
    utils::Assert(img_.shape[3] == labels_.shape[0], "label and image much match each other");
    if (shuffle_) this->Shuffle();
    if (silent_ == 0) {
      mshadow::Shape<4> s = img_.shape;
      printf("MShadowTIterator: shuffle=%d, data=%u,%u,%u,%u\n",
             shuffle_, s[3], s[2], s[1], s[0]);
    }
  }
  virtual void BeforeFirst(void) {
    this->loc_ = 0;
  }
  virtual bool Next(void) {
    if (loc_ < img_.shape[3]) {
      out_.label = labels_[loc_];
      out_.index = inst_[loc_];
      out_.data =  img_[loc_];
      ++ loc_; return true;
    } else {
      return false;
    }
  }
  virtual const DataInst &Value(void) const {
    return out_;
  }
private:
  inline void LoadImageMShadow(void) {
    mshadow::utils::FileStream fs(utils::FopenCheck(path_img_.c_str(), "rb"));
    mshadow::LoadBinary(fs, img_, false);
    fs.Close();
  }
  inline void LoadLabelMShadow(void) {
    mshadow::utils::FileStream fs(utils::FopenCheck(path_label_.c_str(), "rb"));
    mshadow::LoadBinary(fs, labels_, false);
    fs.Close();
    inst_.resize(labels_.shape[0]);
    for (size_t i = 0; i < inst_.size(); ++ i) {
      inst_[i] = (unsigned)i + inst_offset_;
    }
  }
  inline void LoadImageMNIST(void) {
    utils::StdFile fi = utils::StdFile(path_img_.c_str(), "rb");
    unsigned char buf[4];
    std::vector<unsigned> t_data;
    int img_count = 0;
    int img_rows = 0;
    int img_cols = 0;
    fi.Read(buf, 4);
    fi.Read(buf, 4);
    img_count = Pack(buf);
    fi.Read(buf, 4);
    img_rows = Pack(buf);
    fi.Read(buf, 4);
    img_cols = Pack(buf);

    int step = img_rows * img_cols;
    t_data.resize(img_count * step);
    fi.Read(&t_data[0], img_count * step);
    fi.close();

    img_.shape = mshadow::Shape4(1, image_count, image_rows, image_cols);
    img_.shape.stride_ = img_.shape[0];
    img_.dptr = new float[ img_.shape.MSize() ];
    int loc = 0;
    for (int i = 0; i < image_count; ++i) {
      for (int j = 0; j < image_rows; ++j) {
        for (int k = 0; k < image_cols; ++k) {
          img_[1][i][j][k] = static_cast<float>(t_data[loc++]);
        }
      }
    }
    // normalize to 0-1
    img_ *= 1.0f / 256.0f;
  }
  inline void LoadLabelMNIST(void) {
    utils::StdFile fi = utils::StdFile(path_label_, "rb");
    unsigned char buf[4];
    int img_count = 0;
    fi.Read(buf, 4);
    fi.Read(buf, 4);
    img_count = Pack(buf);
    vector<unsigned> t_data(img_count);
    fi.Read(&t_data[0], img_count);
    fi.close();

    labels_.shape = mshadow::Shape1(img_count);
    labels_.shape.stride_ = 1;
    labels_.dptr = new float[labels_.shape.MSize()];
    for (int i = 0; i < img_count; ++i) {
      labels_[i] = static_cast<float>(t_data[i]);
    }

  }
  inline void LoadImageNpy(void) {
    utils::Assert(1<0, "TODO");
  }
  inline void LoadLabelNpy(void) {
    utils::Assert(1<0, "TODO");
  }
  inline void Shuffle(void) {
    utils::Shuffle(inst_);
    mshadow::TensorContainer<cpu, 1> tmplabel(labels_.shape);
    mshadow::TensorContainer<cpu, 4> tmpimg(img_.shape);
    for (size_t i = 0; i < inst_.size(); ++ i) {
      unsigned ridx = inst_[i] - inst_offset_;
      mshadow::Copy(tmpimg[i], img_[ridx]);
      tmplabel[i] = labels_[ ridx ];
    }
    // copy back
    mshadow::Copy(img_, tmpimg);
    mshadow::Copy(labels_, tmplabel);
  }
  inline int Pack(unsigned char zz[4]) {
    return (int)(zz[3])
           | (((int)(zz[2])) << 8)
           | (((int)(zz[1])) << 16)
           | (((int)(zz[0])) << 24);
  }
private:
  // silent
  int silent_;
  // path
  std::string path_img_, path_label_;
  // output
  DataInst out_;
  // whether do shuffle
  int shuffle_;
  // data mode
  int mode_;
  // current location
  index_t loc_;
  // image content
  mshadow::Tensor<cpu, 4> img_;
  // label content
  mshadow::Tensor<cpu, 1> labels_;
  // instance index offset
  unsigned inst_offset_;
  // instance index
  std::vector<unsigned> inst_;
  // file format type
  int type_;
}; //class MShadowIterator
}; // namespace cxxnet
#endif // CXXNET_MSHADOW_ITER_INL_HPP
