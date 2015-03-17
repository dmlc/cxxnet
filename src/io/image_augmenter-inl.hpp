#ifndef IMAGE_AUGMENTER_OPENCV_HPP_
#define IMAGE_AUGMENTER_OPENCV_HPP_
/*!
 * \file image_augmenter_opencv.hpp
 * \brief threaded version of page iterator
 * \author Naiyan Wang, Tianqi Chen
 */
#include <opencv2/opencv.hpp>
#include "../utils/random.h"

namespace cxxnet {
/*! \brief helper class to do image augmentation */
class ImageAugmenter {
 public:
  // contructor
  ImageAugmenter(void)
      : tmpres(false),
        rotateM(2, 3, CV_32F) {
    rand_crop_ = 0;
    crop_y_start_ = -1;
    crop_x_start_ = -1;
    max_rotate_angle_ = 0.0f;
    max_aspect_ratio_ = 0.0f;
    max_shear_ratio_ = 0.0f;
    min_crop_size_ = -1;
    max_crop_size_ = -1;
    rotate_ = -1.0f;
    max_random_scale_ = 1.0f;
    min_random_scale_ = 1.0f;
    min_img_size_ = 0.0f;
    max_img_size_ = 1e10f;
    fill_value_ = 255;
  }
  virtual ~ImageAugmenter() {
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "input_shape")) {
      utils::Check(sscanf(val, "%u,%u,%u", &shape_[0], &shape_[1], &shape_[2]) == 3,
                   "input_shape must be three consecutive integers without space example: 1,1,200 ");
    }
    if (!strcmp(name, "rand_crop")) rand_crop_ = atoi(val);
    if (!strcmp(name, "crop_y_start")) crop_y_start_ = atoi(val);
    if (!strcmp(name, "crop_x_start")) crop_x_start_ = atoi(val);
    if (!strcmp(name, "max_rotate_angle")) max_rotate_angle_ = atof(val);
    if (!strcmp(name, "max_shear_ratio")) max_shear_ratio_ = atof(val);
    if (!strcmp(name, "max_aspect_ratio")) max_aspect_ratio_ = atof(val);
    if (!strcmp(name, "min_crop_size")) min_crop_size_ = atoi(val);
    if (!strcmp(name, "max_crop_size")) max_crop_size_ = atoi(val);
    if (!strcmp(name, "min_random_scale")) min_random_scale_ = atof(val);
    if (!strcmp(name, "max_random_scale")) max_random_scale_ = atof(val);
    if (!strcmp(name, "min_img_size")) min_img_size_ = atof(val);
    if (!strcmp(name, "max_img_size")) max_img_size_ = atof(val);
    if (!strcmp(name, "fill_value")) fill_value_ = atoi(val);
    if (!strcmp(name, "mirror")) mirror_ = atoi(val);
    if (!strcmp(name, "rotate")) rotate_ = atoi(val);
    if (!strcmp(name, "rotate_list")) {
      const char *end = val + strlen(val);
      char buf[128];
      while (val < end) {
        sscanf(val, "%[^,]", buf);
        val += strlen(buf) + 1;
        rotate_list_.push_back(atoi(buf));
      }
    }
  }
  /*!
   * \brief augment src image, store result into dst
   *   this function is not thread safe, and will only be called by one thread
   *   however, it will tries to re-use memory space as much as possible
   * \param src the source image
   * \param source of random number
   * \param dst the pointer to the place where we want to store the result
   */
  virtual cv::Mat Process(const cv::Mat &src,
                          utils::RandomSampler *prnd) {
    // shear
    float s = prnd->NextDouble() * max_shear_ratio_ * 2 - max_shear_ratio_;
    // rotate
    int angle = prnd->NextUInt32(max_rotate_angle_ * 2) - max_rotate_angle_;
    if (rotate_ > 0) angle = rotate_;
    if (rotate_list_.size() > 0) {
      angle = rotate_list_[prnd->NextUInt32(rotate_list_.size() - 1)];
    }
    float a = cos(angle / 180.0 * M_PI);
    float b = sin(angle / 180.0 * M_PI);
    // scale
    float scale = prnd->NextDouble() * (max_random_scale_ - min_random_scale_) + min_random_scale_;
    // aspect ratio
    float ratio = prnd->NextDouble() * max_aspect_ratio_ * 2 - max_aspect_ratio_ + 1;
    float hs = 2 * scale / (1 + ratio);
    float ws = ratio * hs;
    // new width and height
    float new_width = std::max(min_img_size_, std::min(max_img_size_, scale * src.cols));
    float new_height = std::max(min_img_size_, std::min(max_img_size_, scale * src.rows));
    //printf("%f %f %f %f %f %f %f %f %f\n", s, a, b, scale, ratio, hs, ws, new_width, new_height);
    cv::Mat M(2, 3, CV_32F);
    M.at<float>(0, 0) = hs * a - s * b * ws;
    M.at<float>(1, 0) = -b * ws;
    M.at<float>(0, 1) = hs * b + s * a * ws;
    M.at<float>(1, 1) = a * ws;
    float ori_center_width = M.at<float>(0, 0) * src.cols + M.at<float>(0, 1) * src.rows;
    float ori_center_height = M.at<float>(1, 0) * src.cols + M.at<float>(1, 1) * src.rows;
    M.at<float>(0, 2) = (new_width - ori_center_width) / 2;
    M.at<float>(1, 2) = (new_height - ori_center_height) / 2;
    cv::warpAffine(src, temp, M, cv::Size(new_width, new_height),
                     cv::INTER_CUBIC,
                     cv::BORDER_CONSTANT,
                     cv::Scalar(fill_value_, fill_value_, fill_value_));
    cv::Mat res = temp;
    mshadow::index_t y = res.rows - shape_[2];
    mshadow::index_t x = res.cols - shape_[1];
    if (rand_crop_ != 0) {
      y = prnd->NextUInt32(y + 1);
      x = prnd->NextUInt32(x + 1);
    } else {
      y /= 2; x /= 2;
    }
    cv::Rect roi(x, y, shape_[1], shape_[2]);
    res = res(roi);
    return res;
  }
  /*!
   * \brief augment src image, store result into dst
   *   this function is not thread safe, and will only be called by one thread
   *   however, it will tries to re-use memory space as much as possible
   * \param src the source image
   * \param source of random number
   * \param dst the pointer to the place where we want to store the result
   */
  virtual mshadow::Tensor<cpu, 3> Process(mshadow::Tensor<cpu, 3> data,
                                          utils::RandomSampler *prnd) {
    if (!NeedProcess()) return data;
    cv::Mat res(data.size(1), data.size(2), CV_8UC3);
    for (index_t i = 0; i < data.size(1); ++i) {
      for (index_t j = 0; j < data.size(2); ++j) {
        res.at<cv::Vec3b>(i, j)[0] = data[2][i][j];
        res.at<cv::Vec3b>(i, j)[1] = data[1][i][j];
        res.at<cv::Vec3b>(i, j)[2] = data[0][i][j];
      }
    }
    res = this->Process(res, prnd);
    tmpres.Resize(mshadow::Shape3(3, res.rows, res.cols));
    for (index_t i = 0; i < tmpres.size(1); ++i) {
      for (index_t j = 0; j < tmpres.size(2); ++j) {
        cv::Vec3b bgr = res.at<cv::Vec3b>(i, j);
        tmpres[0][i][j] = bgr[2];
        tmpres[1][i][j] = bgr[1];
        tmpres[2][i][j] = bgr[0];
      }
    }
    return tmpres;
  }

 private:
  // whether skip processing
  inline bool NeedProcess(void) const {
    if (max_rotate_angle_ > 0 || max_shear_ratio_ > 0.0f
        || rotate_ > 0 || rotate_list_.size() > 0) return true;
    if (min_crop_size_ > 0 && max_crop_size_ > 0) return true;
    return false;
  }
  // temp input space
  mshadow::TensorContainer<cpu, 3> tmpres;
  // temporal space
  cv::Mat temp0, temp, temp2;
  // rotation param
  cv::Mat rotateM;
  // parameters
  /*! \brief input shape */
  mshadow::Shape<4> shape_;
  /*! \brief whether we do random cropping */
  int rand_crop_;
  /*! \brief whether we do nonrandom croping */
  int crop_y_start_;
  /*! \brief whether we do nonrandom croping */
  int crop_x_start_;
  /*! \brief Indicate the max ratation angle for augmentation, we will random rotate */
  /*! \brief [-max_rotate_angle, max_rotate_angle] */
  int max_rotate_angle_;
  /*! \brief max aspect ratio */
  float max_aspect_ratio_;
  /*! \brief random shear the image [-max_shear_ratio, max_shear_ratio] */
  float max_shear_ratio_;
  /*! \brief max crop size */
  int max_crop_size_;
  /*! \brief min crop size */
  int min_crop_size_;
  /*! \brief max scale ratio */
  float max_random_scale_;
  /*! \brief min scale_ratio */
  float min_random_scale_;
  /*! \brief min image size */
  float min_img_size_;
  /*! \brief max image size */
  float max_img_size_;
  /*! \brief whether to mirror the image */
  int mirror_;
  /*! \brief rotate angle */
  int rotate_;
  /*! \brief filled color while padding */
  int fill_value_;
  /*! \brief list of possible rotate angle */
  std::vector<int> rotate_list_;
};
}  // namespace cxxnet
#endif
