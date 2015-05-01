#ifndef CXXNET_ATTACH_TXT_ITER_INL_HPP_
#define CXXNET_ATTACH_TXT_ITER_INL_HPP_
/*!
 * \file iter_attach_txt-inl.hpp
 * \brief iterator that attach additional data store in txt file
 * \author Naiyan Wang
 */
#include <map>
#include <mshadow/tensor.h>
#include "./data.h"
#include "../utils/utils.h"
#include "../utils/io.h"

namespace cxxnet {
class AttachTxtIterator : public IIterator<DataBatch> {
 public:
  AttachTxtIterator(IIterator<DataBatch> *base)
      : base_(base) {
    file_ = NULL;
    batch_size_ = 0;
    round_batch_ = 0;
  }
  virtual void SetParam(const char *name, const char *val) {
    base_->SetParam(name, val);
    if (!strcmp(name, "filename")) filename_ = val;
    if (!strcmp(name, "batch_size"))  batch_size_ = (index_t)atoi(val);
    if (!strcmp(name, "round_batch")) round_batch_ = atoi(val);
  }
  virtual ~AttachTxtIterator(void) {
    delete base_;
    mshadow::FreeSpace(&extra_data_);
  }
  virtual void Init(void) {
    base_->Init();
    file_ = fopen(filename_.c_str(), "r");
    CHECK(file_ != NULL)
        << "AttachTxt: Open file failed: " << filename_;
    CHECK(fscanf(file_, "%d", &dim_) == 1)
        << "AttachTxt: First line should indicate the data dim.";
    extra_data_ = mshadow::NewTensor<cpu>(
            mshadow::Shape4(batch_size_, 1, 1, dim_), 0.0f, false);
    int cnt = 0;
    int data_id = 0;
    while (fscanf(file_, "%d", &data_id) == 1) {
      id_map_[data_id] = cnt++;
      for (int i = 0; i < dim_; ++i) {
        float tmp;
        utils::Check(fscanf(file_, "%f", &tmp) == 1,
                     "AttachTxt: data do not match dimension specified");
        all_data_.push_back(tmp);
      }
    }
    fclose(file_);
  }
  virtual void BeforeFirst(void) {
    base_->BeforeFirst();
  }
  virtual bool Next(void) {
    if (base_->Next()) {
      out_ = base_->Value();
      out_.extra_data.clear();
      out_.extra_data.push_back(extra_data_);
      for (int top = 0; top < batch_size_; ++top) {
        if (id_map_.find(out_.inst_index[top]) != id_map_.end()) {
          int start = id_map_[out_.inst_index[top]] * dim_;
          for (int i = 0; i < dim_; ++i) {
            extra_data_[top][0][0][i] = all_data_[start++];  
          }
        }
      }
      return true;
    } else {
      return false;
    }
  }
  virtual const DataBatch &Value(void) const {
    return out_;
  }

 private:
  /*! \brief dim of the additional data */
  int dim_;
  /*! \brief batch size */
  int batch_size_;
  /*! \brief whether to use round robin to pad batch */
  int round_batch_;
  /*! \brief the output data batch */
  DataBatch out_;
  /*! \brief filename of the extra data */
  std::string filename_;
  /*! \brief file pointer of the file */
  FILE* file_;
  /*! \brief file pointer of the file */
  mshadow::Tensor<cpu, 4> extra_data_;
  /*! \brief base iterator */
  IIterator<DataBatch> *base_;
  std::map<int, int> id_map_;
  std::vector<float> all_data_;
};
}  // namespace cxxnet
#endif  // CXXNET_ATTACH_TXT_ITER_INL_HPP_
