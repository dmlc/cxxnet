#ifndef CXXNET_ITER_MEM_BUFFER_INL_HPP_
#define CXXNET_ITER_MEM_BUFFER_INL_HPP_
/*!
 * \file iter_mem_buffer-inl.hpp
 * \brief iterator that gets limited number of batch into memory,
 *        and only return these data
 * \author Tianqi Chen
 */
#include <mshadow/tensor.h>
#include "./data.h"
#include "../utils/utils.h"
#include "../utils/io.h"

namespace cxxnet {
/*! \brief iterator that gets limitted number of batch into memory */
class DenseBufferIterator : public IIterator<DataBatch> {
 public:
  DenseBufferIterator(IIterator<DataBatch> *base)
      : base_(base) {
    max_nbatch_ = 100;
    data_index_ = 0;
    silent_ = 0;
  }
  virtual void SetParam(const char *name, const char *val) {
    base_->SetParam(name, val);
    if (!strcmp(name, "max_nbatch")) {
      max_nbatch_ = static_cast<size_t>(atol(val));
    }
    if (!strcmp(name, "silent")) silent_ = atoi(val);
  }
  virtual void Init(void) {
    base_->Init();
    while (base_->Next()) {
      const DataBatch &batch = base_->Value();
      utils::Assert(batch.label.dptr_ != NULL, "need dense");
      DataBatch v;
      v.AllocSpaceDense(batch.data.shape_, batch.batch_size, batch.label.size(1));
      v.CopyFromDense(batch);
      buffer_.push_back(v);
      if (buffer_.size() >= max_nbatch_) break;
    }
    if (silent_ == 0) {
      printf("DenseBufferIterator: load %d batches\n",
             static_cast<int>(buffer_.size()));
    }
  }
  virtual void BeforeFirst(void) {
    data_index_ = 0;
  }
  virtual bool Next(void) {
    if (data_index_ < buffer_.size()) {
      data_index_ += 1;
      return true;
    } else {
      return false;
    }
  }
  virtual const DataBatch &Value(void) const {
    utils::Assert(data_index_ > 0,
                  "Iterator.Value: at beginning of iterator");
    return buffer_[data_index_ - 1];
  }

 private:
  /*! \brief silent */
  int silent_;
  /*! \brief maximum number of batch in buffer */
  size_t max_nbatch_;
  /*! \brief data index */
  size_t data_index_;
  /*! \brief base iterator */
  IIterator<DataBatch> *base_;
  /*! \brief data content */
  std::vector<DataBatch> buffer_;
};
}  // namespace cxxnet
#endif  // CXXNET_ITER_BATCH_PROC_INL_HPP_
