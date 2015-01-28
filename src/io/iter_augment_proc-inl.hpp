#ifndef CXXNET_ITER_PROC_INL_HPP_
#define CXXNET_ITER_PROC_INL_HPP_
/*!
 * \file iter_augment_proc-inl.hpp
 * \brief processing unit to do data augmention
 * \author Tianqi Chen, Bing Xu, Naiyan Wang
 */
#include <mshadow/tensor.h>
#include "data.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../utils/global_random.h"
#include "../utils/thread_buffer.h"

#if CXXNET_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace cxxnet {
/*! \brief create a batch iterator from single instance iterator */
class AugmentIterator: public IIterator<DataInst> {
public:
  AugmentIterator(IIterator<DataInst> *base): base_(base) {
    rand_crop_ = 0;
    rand_mirror_ = 0;
    // skip read, used for debug
    test_skipread_ = 0;
    // scale data
    scale_ = 1.0f;
    // use round roubin to handle overflow batch
    round_batch_ = 0;
    // number of overflow instances that readed in round_batch mode
    num_overflow_ = 0;
    // silent
    silent_ = 0;
    // by default, not mean image file
    name_meanimg_ = "";
    crop_y_start_ = -1;
    crop_x_start_ = -1;
    max_rotate_angle_ = -1;
    max_aspect_ratio_ = 0.0f;
    max_shear_ratio_ = 0.0f;
    min_crop_size_ = -1;
    max_crop_size_ = -1;
    mean_r_ = 0.0f;
    mean_g_ = 0.0f;
    mean_b_ = 0.0f;
    flip_ = 0;
  }
  virtual ~AugmentIterator(void) {
    delete base_;
  }
  virtual void SetParam(const char *name, const char *val) {
    base_->SetParam(name, val);
    if (!strcmp(name, "input_shape")) {
      utils::Assert(sscanf(val, "%u,%u,%u", &shape_[0], &shape_[1], &shape_[2]) == 3,
                    "input_shape must be three consecutive integers without space example: 1,1,200 ");
    }
    if (!strcmp(name, "round_batch")) round_batch_ = atoi(val);
    if (!strcmp(name, "rand_crop"))   rand_crop_ = atoi(val);
    if (!strcmp(name, "crop_y_start"))  crop_y_start_ = atoi(val);
    if (!strcmp(name, "crop_x_start"))  crop_x_start_ = atoi(val);
    if (!strcmp(name, "rand_mirror")) rand_mirror_ = atoi(val);
    if (!strcmp(name, "silent"))      silent_ = atoi(val);
    if (!strcmp(name, "divideby"))    scale_ = static_cast<real_t>(1.0f / atof(val));
    if (!strcmp(name, "scale"))       scale_ = static_cast<real_t>(atof(val));
    if (!strcmp(name, "image_mean"))   name_meanimg_ = val;
    if (!strcmp(name, "max_rotate_angle")) max_rotate_angle_ = atof(val);
    if (!strcmp(name, "max_shear_ratio"))  max_shear_ratio_ = atof(val);
    if (!strcmp(name, "max_aspect_ratio"))  max_aspect_ratio_ = atof(val);
    if (!strcmp(name, "test_skipread"))    test_skipread_ = atoi(val);
    if (!strcmp(name, "min_crop_size"))     min_crop_size_ = atoi(val);
    if (!strcmp(name, "max_crop_size"))     max_crop_size_ = atoi(val);
    if (!strcmp(name, "flip")) flip_ = atoi(val);
    if (!strcmp(name, "mean_value")) {
      utils::Assert(sscanf(val, "%f,%f,%f", &mean_b_, &mean_g_, &mean_r_) == 3,
                    "mean value must be three consecutive float without space example: 128,127.5,128.2 ");
    }
  }
  virtual void Init(void) {
    base_->Init();
    printf("In augment init.\n");
    meanfile_ready_ = false;
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
      }
      meanfile_ready_ = true;
    }
  }
  virtual void BeforeFirst(void) {
    base_->BeforeFirst();
  }
  virtual bool Next(void) {
    if (!base_->Next()){
      return false;
    }
    const DataInst &d = base_->Value();
    this->SetData(d);
    return true;
  }
  virtual const DataInst &Value(void) const {
    return out_;
  }
private:
  inline void SetData(const DataInst &d) {
    using namespace mshadow::expr;
    bool cut = false;
    out_.label = d.label;
    out_.index = d.index;
    img_.Resize(mshadow::Shape3(d.data.shape_[0], shape_[1], shape_[2]));
    if (shape_[1] == 1) {
      img_ = d.data * scale_;
    } else {
      utils::Assert(d.data.size(1) >= shape_[1] && d.data.size(2) >= shape_[2],
        "Data size must be bigger than the input size to net.");
      #if CXXNET_USE_OPENCV
      cv::Mat res(d.data.size(1), d.data.size(2), CV_8UC3);
      index_t out_h = d.data.size(1);
      index_t out_w = d.data.size(2);
      for (index_t i = 0; i < d.data.size(1); ++i) {
        for (index_t j = 0; j < d.data.size(2); ++j) {
          res.at<cv::Vec3b>(i, j)[0] = d.data[2][i][j];
          res.at<cv::Vec3b>(i, j)[1] = d.data[1][i][j];
          res.at<cv::Vec3b>(i, j)[2] = d.data[0][i][j];
        }
      }
      if (max_rotate_angle_ > 0.0f || max_shear_ratio_ > 0.0f) {
        int angle = utils::NextUInt32(max_rotate_angle_ * 2) - max_rotate_angle_;
        int len = std::max(res.cols, res.rows);
        cv::Point2f pt(len / 2.0f, len / 2.0f);
        cv::Mat M(2, 3, CV_32F);
        float cs = cos(angle / 180.0 * M_PI);
        float sn = sin(angle / 180.0 * M_PI);
        float q = utils::NextDouble() * max_shear_ratio_ * 2 - max_shear_ratio_;
        M.at<float>(0, 0) = cs;
        M.at<float>(0, 1) = sn;
        M.at<float>(0, 2) = 0.0f;
        M.at<float>(1, 0) = q * cs - sn;
        M.at<float>(1, 1) = q * sn + cs;
        M.at<float>(1, 2) = 0.0f;
        cv::Mat temp;
        cv::warpAffine(res, temp, M, cv::Size(len, len),
              cv::INTER_CUBIC,
              cv::BORDER_CONSTANT,
              cv::Scalar(255, 255, 255));
        res = temp;
      }
      if (min_crop_size_ > 0 && max_crop_size_ > 0) {
        int crop_size_x = utils::NextUInt32(max_crop_size_ - min_crop_size_ + 1) + \
                                       min_crop_size_;
        int crop_size_y = crop_size_x * (1 + utils::NextDouble() * \
                                                      max_aspect_ratio_ * 2 - max_aspect_ratio_);
        crop_size_y = std::max(min_crop_size_, std::min(crop_size_y, max_crop_size_));
        mshadow::index_t y = res.rows - crop_size_y;
        mshadow::index_t x = res.cols - crop_size_x;
        if (rand_crop_ != 0) {
          y = utils::NextUInt32(y + 1);
          x = utils::NextUInt32(x + 1);
        } else {
          y /= 2; x /= 2;
        }
        if (crop_y_start_ != -1) y = crop_y_start_;
        if (crop_x_start_ != -1) x = crop_x_start_;
        cv::Rect roi(y, x, crop_size_y, crop_size_x);
        res = res(roi);
        cut = true;
        cv::resize(res, res, cv::Size(shape_[1], shape_[2]));
        out_h = shape_[1];
        out_w = shape_[2];
      }
      for (index_t i = 0; i < out_h; ++i) {
        for (index_t j = 0; j < out_w; ++j) {
          cv::Vec3b bgr = res.at<cv::Vec3b>(i, j);
          d.data[0][i][j] = bgr[2];
          d.data[1][i][j] = bgr[1];
          d.data[2][i][j] = bgr[0];
        }
      }
      res.release();
      #endif
      mshadow::index_t yy = d.data.size(1) - shape_[1];
      mshadow::index_t xx = d.data.size(2) - shape_[2];
      if (rand_crop_ != 0) {
        yy = utils::NextUInt32(yy + 1);
        xx = utils::NextUInt32(xx + 1);
      } else {
        yy /= 2; xx /= 2;
      }
      if (cut) {
        yy = 0;
        xx = 0;
      } else {
        if (crop_y_start_ != -1) yy = crop_y_start_;
        if (crop_x_start_ != -1) xx = crop_x_start_;
      }
      if (mean_r_ > 0.0f || mean_g_ > 0.0f || mean_b_ > 0.0f) {
        d.data[0] -= mean_b_; d.data[1] -= mean_g_; d.data[2] -= mean_r_;
        if ((rand_mirror_ != 0 && utils::NextDouble() < 0.5f) || flip_ == 1) {
          img_ = mirror(crop(d.data, img_[0].shape_, yy, xx)) * scale_;
        } else {
          img_ = crop(d.data, img_[0].shape_, yy, xx) * scale_ ;
        }
      } else if (!meanfile_ready_ || name_meanimg_.length() == 0) {
        if (rand_mirror_ != 0 && utils::NextDouble() < 0.5f) {
          img_ = mirror(crop(d.data, img_[0].shape_, yy, xx)) * scale_;
        } else {
          img_ = crop(d.data, img_[0].shape_, yy, xx) * scale_ ;
        }
      } else {
        // substract mean image
        if ((rand_mirror_ != 0 && utils::NextDouble() < 0.5f) || flip_ == 1) {
          if (d.data.shape_ == meanimg_.shape_){
            img_ = mirror(crop(d.data - meanimg_, img_[0].shape_, yy, xx)) * scale_;
          } else {
            img_ = mirror(crop(d.data, img_[0].shape_, yy, xx) - meanimg_) * scale_;
          }
        } else {
          if (d.data.shape_ == meanimg_.shape_){
            img_ = crop(d.data - meanimg_, img_[0].shape_, yy, xx) * scale_ ;
          } else {
            img_ = (crop(d.data, img_[0].shape_, yy, xx) - meanimg_) * scale_;
          }
        }
      }
    }
    out_.data = img_;
  }
  inline void CreateMeanImg(void) {
    if (silent_ == 0) {
      printf("cannot find %s: create mean image, this will take some time...\n", name_meanimg_.c_str());
    }
    time_t start = time(NULL);
    unsigned long elapsed = 0;
    size_t imcnt = 1;

    utils::Assert(this->Next(), "input iterator failed.");
    meanimg_.Resize(mshadow::Shape3(shape_[0], shape_[1], shape_[2]));
    mshadow::Copy(meanimg_, this->Value().data);
    while (this->Next()) {
      meanimg_ += this->Value().data; imcnt += 1;
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
    this->BeforeFirst();
  }
private:
  // base iterator
  IIterator<DataInst> *base_;
  // input shape
  mshadow::Shape<4> shape_;
  // output data
  DataInst out_;
  // skip read
  int test_skipread_;
  // silent
  int silent_;
  // scale of data
  real_t scale_;
  // whether we do random cropping
  int rand_crop_;
  // whether we do random mirroring
  int rand_mirror_;
  // use round roubin to handle overflow batch
  int round_batch_;
  // whether we do nonrandom croping
  int crop_y_start_;
  // whether we do nonrandom croping
  int crop_x_start_;
  // number of overflow instances that readed in round_batch mode
  int num_overflow_;
  // mean image, if needed
  mshadow::TensorContainer<cpu, 3> meanimg_;
  // temp space
  mshadow::TensorContainer<cpu, 3> img_;
  // mean image file, if specified, will generate mean image file, and substract by mean
  std::string name_meanimg_;
  // Indicate the max ratation angle for augmentation, we will random rotate
  // [-max_rotate_angle, max_rotate_angle]
  int max_rotate_angle_;
  // max aspect ratio
  float max_aspect_ratio_;
  // max shear ratio
  // will random shear the image [-max_shear_ratio, max_shear_ratio]
  float max_shear_ratio_;
  int max_crop_size_;
  int min_crop_size_;
  float mean_r_;
  float mean_g_;
  float mean_b_;
  bool meanfile_ready_;
  int flip_;
};  // class AugmentIterator
}  // namespace cxxnet
#endif
