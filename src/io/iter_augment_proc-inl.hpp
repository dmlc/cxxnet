#ifndef CXXNET_ITER_AUGMENT_INL_HPP_
#define CXXNET_ITER_AUGMENT_INL_HPP_
/*!
 * \file iter_augment_proc-inl.hpp
 * \brief processing unit to do data augmention
 * \author Tianqi Chen, Bing Xu, Naiyan Wang
 */
#include <mshadow/tensor.h>
#include "data.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../utils/random.h"
#include "../utils/thread_buffer.h"

#if CXXNET_USE_OPENCV
#include "./image_augmenter-inl.hpp"
#endif

namespace cxxnet {
/*! \brief create a batch iterator from single instance iterator */
class AugmentIterator: public IIterator<DataInst> {
public:
  AugmentIterator(IIterator<DataInst> *base, int no_aug=0)
      : base_(base), no_aug_(no_aug) {
    rand_crop_ = 0;
    rand_mirror_ = 0;
    crop_y_start_ = -1;
    crop_x_start_ = -1;
    // scale data
    scale_ = 1.0f;
    // silent
    silent_ = 0;
    // by default, not mean image file
    name_meanimg_ = "";
    meanfile_ready_ = false;
    mean_r_ = 0.0f;
    mean_g_ = 0.0f;
    mean_b_ = 0.0f;
    mirror_ = 0;
    max_random_illumination_ = 0.0f;
    max_random_contrast_ = 0.0f;
    rnd.Seed(kRandMagic);
  }
  virtual ~AugmentIterator(void) {
    delete base_;
  }
  virtual void SetParam(const char *name, const char *val) {
    base_->SetParam(name, val);
    if (!strcmp(name, "input_shape")) {
      utils::Check(sscanf(val, "%u,%u,%u", &shape_[0], &shape_[1], &shape_[2]) == 3,
                   "input_shape must be three consecutive integers without space example: 1,1,200 ");
    }
    if (!strcmp(name, "seed_data")) rnd.Seed(kRandMagic + atoi(val));
    if (!strcmp(name, "rand_crop")) rand_crop_ = atoi(val);
    if (!strcmp(name, "silent")) silent_ = atoi(val);
    if (!strcmp(name, "divideby")) scale_ = static_cast<real_t>(1.0f / atof(val));
    if (!strcmp(name, "scale")) scale_ = static_cast<real_t>(atof(val));
    if (!strcmp(name, "image_mean")) name_meanimg_ = val;
    if (!strcmp(name, "crop_y_start")) crop_y_start_ = atoi(val);
    if (!strcmp(name, "crop_x_start")) crop_x_start_ = atoi(val);
    if (!strcmp(name, "rand_mirror")) rand_mirror_ = atoi(val);
    if (!strcmp(name, "mirror")) mirror_ = atoi(val);
    if (!strcmp(name, "max_random_contrast")) max_random_contrast_ = atof(val);
    if (!strcmp(name, "max_random_illumination")) max_random_illumination_ = atof(val);
    if (!strcmp(name, "mean_value")) {
      utils::Check(sscanf(val, "%f,%f,%f", &mean_b_, &mean_g_, &mean_r_) == 3,
                   "mean value must be three consecutive float without space example: 128,127.5,128.2 ");
    }
#if CXXNET_USE_OPENCV
    aug.SetParam(name, val);
#endif
  }
  virtual void Init(void) {
    base_->Init();
    if (name_meanimg_.length() != 0) {
      FILE *fi = fopen64(name_meanimg_.c_str(), "rb");
      if (fi == NULL) {
        this->CreateMeanImg();
      } else {
        if (silent_ == 0) {
          printf("loading mean image from %s\n", name_meanimg_.c_str());
        }
        utils::FileStream fs(fi) ;
        meanimg_.LoadBinary(fs);
        fclose(fi);
        meanfile_ready_ = true;
      }
    }
  }
  virtual void BeforeFirst(void) {
    base_->BeforeFirst();
  }
  virtual const DataInst &Value(void) const {
    return out_;
  }
  virtual bool Next(void) {
    if (!this->Next_()) return false;
    return true;
  }

private:
  inline void SetData(const DataInst &d) {
    using namespace mshadow::expr;
    out_.label = d.label;
    out_.index = d.index;
    mshadow::Tensor<cpu, 3> data = d.data;
#if CXXNET_USE_OPENCV
    if (!no_aug_) data = aug.Process(data, &rnd);
#endif

    img_.Resize(mshadow::Shape3(data.shape_[0], shape_[1], shape_[2]));
    if (shape_[1] == 1) {
      img_ = data * scale_;
    } else {
      utils::Assert(data.size(1) >= shape_[1] && data.size(2) >= shape_[2],
                    "Data size must be bigger than the input size to net.");
      mshadow::index_t yy = data.size(1) - shape_[1];
      mshadow::index_t xx = data.size(2) - shape_[2];
      if (rand_crop_ != 0 && (yy != 0 || xx != 0)) {
        yy = rnd.NextUInt32(yy + 1);
        xx = rnd.NextUInt32(xx + 1);
      } else {
        yy /= 2; xx /= 2;
      }
      if (data.size(1) != shape_[1] && crop_y_start_ != -1) {
        yy = crop_y_start_;
      }
      if (data.size(2) != shape_[2] && crop_x_start_ != -1) {
        xx = crop_x_start_;
      }
      float contrast = rnd.NextDouble() * max_random_contrast_ * 2 - max_random_contrast_ + 1;
      float illumination = rnd.NextDouble() * max_random_illumination_ * 2 - max_random_illumination_;
      if (mean_r_ > 0.0f || mean_g_ > 0.0f || mean_b_ > 0.0f) {
        // substract mean value
        data[0] -= mean_b_; data[1] -= mean_g_; data[2] -= mean_r_;
        if ((rand_mirror_ != 0 && rnd.NextDouble() < 0.5f) || mirror_ == 1) {
          img_ = mirror(crop(data * contrast + illumination, img_[0].shape_, yy, xx)) * scale_;
        } else {
          img_ = crop(data * contrast + illumination, img_[0].shape_, yy, xx) * scale_ ;
        }
      } else if (!meanfile_ready_ || name_meanimg_.length() == 0) {
        // do not substract anything
        if (rand_mirror_ != 0 && rnd.NextDouble() < 0.5f) {
          img_ = mirror(crop(data, img_[0].shape_, yy, xx)) * scale_;
        } else {
          img_ = crop(data, img_[0].shape_, yy, xx) * scale_ ;
        }
      } else {
        // substract mean image
        if ((rand_mirror_ != 0 && rnd.NextDouble() < 0.5f) || mirror_ == 1) {
          if (data.shape_ == meanimg_.shape_){
            img_ = mirror(crop((data - meanimg_) * contrast + illumination, img_[0].shape_, yy, xx)) * scale_;
          } else {
            img_ = (mirror(crop(data, img_[0].shape_, yy, xx) - meanimg_) * contrast + illumination) * scale_;
          }
        } else {
          if (data.shape_ == meanimg_.shape_){
            img_ = crop((data - meanimg_) * contrast + illumination, img_[0].shape_, yy, xx) * scale_ ;
          } else {
            img_ = ((crop(data, img_[0].shape_, yy, xx) - meanimg_) * contrast + illumination) * scale_;
          }
        }
      }
    }
    out_.data = img_;
  }
  inline bool Next_(void) {
    if (!base_->Next()) {
      return false;
    }
    const DataInst &d = base_->Value();
    this->SetData(d);
    return true;
  }
  inline void CreateMeanImg(void) {
    if (silent_ == 0) {
      printf("cannot find %s: create mean image, this will take some time...\n", name_meanimg_.c_str());
    }
    time_t start = time(NULL);
    unsigned long elapsed = 0;
    size_t imcnt = 1;

    utils::Assert(this->Next_(), "input iterator failed.");
    meanimg_.Resize(mshadow::Shape3(shape_[0], shape_[1], shape_[2]));
    mshadow::Copy(meanimg_, img_);
    while (this->Next()) {
      meanimg_ += img_; imcnt += 1;
      elapsed = (long)(time(NULL) - start);
      if (imcnt % 1000 == 0 && silent_ == 0) {
        printf("\r                                                               \r");
        printf("[%8lu] images processed, %ld sec elapsed", imcnt, elapsed);
        fflush(stdout);
      }
    }
    meanimg_ *= (1.0f / imcnt);
    utils::StdFile fo(name_meanimg_.c_str(), "wb");
    meanimg_.SaveBinary(fo);
    if (silent_ == 0) {
      printf("save mean image to %s..\n", name_meanimg_.c_str());
    }
    meanfile_ready_ = true;
    this->BeforeFirst();
  }
private:
  /*! \brief base iterator */
  IIterator<DataInst> *base_;
  /*! \brief input shape */
  mshadow::Shape<4> shape_;
  /*! \brief output data */
  DataInst out_;
  /*! \brief silent */
  int silent_;
  /*! \brief scale of data */
  real_t scale_;
  /*! \brief whether we do random cropping */
  int rand_crop_;
  /*! \brief whether we do nonrandom croping */
  int crop_y_start_;
  /*! \brief whether we do nonrandom croping */
  int crop_x_start_;
  /*! \brief whether we do random mirroring */
  int rand_mirror_;
  /*! \brief mean image, if needed */
  mshadow::TensorContainer<cpu, 3> meanimg_;
  /*! \brief temp space */
  mshadow::TensorContainer<cpu, 3> img_;
  /*! \brief mean image file, if specified, will generate mean image file, and substract by mean */
  std::string name_meanimg_;
  /*! \brief mean value for r channel */
  float mean_r_;
  /*! \brief mean value for g channel */
  float mean_g_;
  /*! \brief mean value for b channel */
  float mean_b_;
  /*! \brief maximum ratio of contrast variation */
  float max_random_contrast_;
  /*! \brief maximum value of illumination variation */
  float max_random_illumination_;
  /*! \brief whether to mirror the image */
  int mirror_;
  /*! \brief whether mean file is ready */
  bool meanfile_ready_;
  int no_aug_;
  // augmenter
#if CXXNET_USE_OPENCV
  ImageAugmenter aug;
#endif
  // random sampler
  utils::RandomSampler rnd;
  // random magic number of this iterator
  static const int kRandMagic = 0;
};  // class AugmentIterator
}  // namespace cxxnet
#endif
