#ifndef CXXNET_ATTACH_TXT_ITER_INL_HPP_
#define CXXNET_ATTACH_TXT_ITER_INL_HPP_
/*!
 * \file iter_attach_txt-inl.hpp
 * \brief iterator that attach additional data store in txt file
 * \author Naiyan Wang
 */
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
    if (file_ != NULL) fclose(file_);
    if (out_.inst_index != NULL) delete[] out_.inst_index;
    mshadow::FreeSpace(&extra_data_);
  }
  virtual void Init(void) {
    base_->Init();
    file_ = fopen(filename_.c_str(), "r");
    utils::Assert(file_ != NULL,
      "AttachTxt: Open file failed: %s", filename_.c_str());
    utils::Assert(fscanf(file_, "%d", &dim_) == 1,
      "AttachTxt: First line should indicate the data dim.");
    extra_data_ = mshadow::NewTensor<cpu>(
            mshadow::Shape4(batch_size_, 1, 1, dim_), 0.0f, false);
    //out_.extra_data.push_back(extra_data_);
  }
  virtual void BeforeFirst(void) {
    base_->BeforeFirst();
    if (file_ != NULL){
      fseek(file_, 0, SEEK_SET);
    }
    fscanf(file_, "%d", &dim_);
  }
  virtual bool Next(void) {
    if (base_->Next()){
      out_ = base_->Value();
      out_.extra_data.clear();
      out_.extra_data.push_back(extra_data_);
      bool failed = false;
      int top = 0;
      for (top = 0; top < batch_size_; ++top){
        for (int j = 0; j < dim_; ++j){
          if (fscanf(file_, "%f", &(out_.extra_data[0][top][0][0][j])) != 1){
            failed = true; break;
          }
        }
        if (failed){
          break;
        }
      }
      if (!failed) return true;
      int read = top;
      if (round_batch_ != 0){
        for (; top < batch_size_; ++top){
          for (int j = 0; j < dim_; ++j){
            fscanf(file_, "%f", &(out_.extra_data[0][top][0][0][j]));
          }
        }
      }
      out_.num_batch_padd = batch_size_ - read;
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
};
}  // namespace cxxnet
#endif  // CXXNET_ATTACH_TXT_ITER_INL_HPP_
