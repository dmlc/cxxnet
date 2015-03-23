#ifndef CXXNET_ITER_CSV_INL_HPP
#define CXXNET_ITER_CSV_INL_HPP
#pragma once
/*!
 * \file iter_csv-inl.hpp
 * \brief implementation of csv iterator
 * \author Naiyan Wang
 */
#include "data.h"
#include <mshadow/tensor.h>
#include <opencv2/opencv.hpp>

namespace cxxnet{
  /*! \brief simple csv iterator that only loads data instance */
class CSVIterator : public IIterator<DataInst> {
public:
  CSVIterator(void) {
    data_.set_pad(false);
    fplst_ = NULL;
    silent_ = 0;
    filename_ = "";
    data_index_ = 0;
    label_width_ = 1;
    has_header_ = 0;
  }
  virtual ~CSVIterator(void) {
    if(fplst_ != NULL) fclose(fplst_);
  }
  virtual void SetParam(const char *name, const char *val) {
    if(!strcmp(name, "filename"))  filename_ = val;
    if(!strcmp(name, "has_header"))   has_header_ = atoi(val);
    if(!strcmp(name, "silent"  ))  silent_ = atoi(val);
    if(!strcmp(name, "label_width"  ))  label_width_ = atoi(val);
    if (!strcmp(name, "input_shape")) {
      utils::Check(sscanf(val, "%u,%u,%u", &shape_[0], &shape_[1], &shape_[2]) == 3,
                   "input_shape must be three consecutive integers without space example: 1,1,200 ");
    }
  }
  virtual void Init(void) {
    fplst_  = utils::FopenCheck(filename_.c_str(), "r");
    if(silent_ == 0) {
      printf("CSVIterator:filename=%s\n", filename_.c_str());
    }
    // Skip the header
    if (has_header_) {
      char ch;
      while ((ch = fgetc(fplst_)) != EOF) {
        if (ch == '\r' || ch == '\n') {
          break;
        }
      }
    }
    labels_.resize(label_width_);
    data_.Resize(shape_);
    this->BeforeFirst();
  }
  virtual void BeforeFirst(void) {
    data_index_ = 0;
    fseek(fplst_, 0, SEEK_SET);
  }
  virtual bool Next(void) {
    if (fscanf(fplst_, "%f", BeginPtr(labels_)) != EOF) {
      for (int i = 1; i < label_width_; ++i){
        utils::Check(fscanf(fplst_, ",%f", &labels_[i]) == 1,
          "CSVIterator: Error when reading label. Possible incorrect file or label_width.");
      }
      for (index_t i = 0; i < shape_[0]; ++i) {
        for (index_t j = 0; j < shape_[1]; ++j) {
          for (index_t k = 0; k < shape_[2]; ++k) {
            utils::Check(fscanf(fplst_, ",%f", &data_[i][j][k]) == 1,
              "CSVIterator: Error when reading data. Possible incorrect file or input_shape.");
          }
        }
      }
      out_.data = data_;
      out_.index = data_index_++;
      mshadow::Tensor<cpu, 1> label_(&(labels_[0]), mshadow::Shape1(label_width_));
      out_.label = label_;
      return true;

    }
    return false;
  }
  virtual const DataInst &Value(void) const{
    return out_;
  }

protected:
  // output data
  DataInst out_;
  // silent
  int silent_;
  // file pointer to list file, information file
  FILE *fplst_;
  // prefix path of image folder, path to input lst, format: imageid label path
  std::string filename_;
  // temp storage for image
  mshadow::TensorContainer<cpu, 3> data_;
  // whether the data will be shuffled in each epoch
  int shuffle_;
  // denotes the number of labels
  int label_width_;
  // denotes the current data index
  int data_index_;
  // stores the labels of data
  std::vector<float> labels_;
  // brief input shape
  mshadow::Shape<3> shape_;
  // has header
  int has_header_;
  };
};
#endif