#ifndef CXXNET_ITER_BATCH_PROC_INL_HPP_
#define CXXNET_ITER_BATCH_PROC_INL_HPP_
/*!
 * \file iter_batch_proc-inl.hpp
 * \brief definition of preprocessing iterators that takes an iterator and do some preprocessing
 * \author Tianqi Chen
 */
#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include "./data.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../utils/thread_buffer.h"

namespace cxxnet {
/*! \brief create a batch iterator from single instance iterator */
class BatchAdaptIterator: public IIterator<DataBatch> {
public:
  BatchAdaptIterator(IIterator<DataInst> *base): base_(base) {
    // skip read, used for debug
    test_skipread_ = 0;
    // use round roubin to handle overflow batch
    round_batch_ = 0;
    // number of overflow instances that readed in round_batch mode
    num_overflow_ = 0;
    // silent
    silent_ = 0;
    // label width
    label_width_ = 1;
  }
  virtual ~BatchAdaptIterator(void) {
    delete base_;
    out_.FreeSpaceDense();
  }
  virtual void SetParam(const char *name, const char *val) {
    base_->SetParam(name, val);
    if (!strcmp(name, "batch_size"))  batch_size_ = (index_t)atoi(val);
    if (!strcmp(name, "input_shape")) {
      CHECK(sscanf(val, "%u,%u,%u", &shape_[1], &shape_[2], &shape_[3]) == 3)
          << "input_shape must be three consecutive integers without space example: 1,1,200 ";
    }
    if (!strcmp(name, "label_width")) {
      label_width_ = static_cast<index_t>(atoi(val));
    }
    if (!strcmp(name, "round_batch")) round_batch_ = atoi(val);
    if (!strcmp(name, "silent")) silent_ = atoi(val);
    if (!strcmp(name, "test_skipread")) test_skipread_ = atoi(val);
  }
  virtual void Init(void) {
    base_->Init();
    mshadow::Shape<4> tshape = shape_;
    tshape[0] = batch_size_;
    out_.AllocSpaceDense(tshape, batch_size_, label_width_, false);
  }

  virtual void BeforeFirst(void) {
    if (round_batch_ == 0 || num_overflow_ == 0) {
      // otherise, we already called before first
      base_->BeforeFirst();
    } else {
      num_overflow_ = 0;
    }
    head_ = 1;
  }
  virtual bool Next(void) {
    out_.num_batch_padd = 0;

    // skip read if in head version
    if (test_skipread_ != 0 && head_ == 0) return true;
    else this->head_ = 0;

    // if overflow from previous round, directly return false, until before first is called
    if (num_overflow_ != 0) return false;
    index_t top = 0;

    while (base_->Next()) {
      const DataInst& d = base_->Value();
      mshadow::Copy(out_.label[top], d.label);
      out_.inst_index[top] = d.index;
      //out_.data[top] = d.data;
      Copy(out_.data[top], d.data);

      if (++ top >= batch_size_) return true;
    }
    if (top != 0) {
      if (round_batch_ != 0) {
        num_overflow_ = 0;
        base_->BeforeFirst();
        for (; top < batch_size_; ++top, ++num_overflow_) {
          CHECK(base_->Next()) << "number of input must be bigger than batch size";
          const DataInst& d = base_->Value();
          mshadow::Copy(out_.label[top], d.label);
          out_.inst_index[top] = d.index;
          out_.data[top] = d.data;
        }
        out_.num_batch_padd = num_overflow_;
      } else {
        out_.num_batch_padd = batch_size_ - top;
      }
      return true;
    }
    return false;
  }
  virtual const DataBatch &Value(void) const {
    CHECK(head_ == 0) << "must call Next to get value";
    return out_;
  }
private:
  /*! \brief base iterator */
  IIterator<DataInst> *base_;
  /*! \brief batch size */
  index_t batch_size_;
  /*! \brief input shape */
  mshadow::Shape<4> shape_;
  /*! \brief label width */
  index_t label_width_;
  /*! \brief output data */
  DataBatch out_;
  /*! \brief on first */
  int head_;
  /*! \brief skip read */
  int test_skipread_;
  /*! \brief silent */
  int silent_;
  /*! \brief use round roubin to handle overflow batch */
  int round_batch_;
  /*! \brief number of overflow instances that readed in round_batch mode */
  int num_overflow_;
}; // class BatchAdaptIterator

/*! \brief thread buffer iterator */
class ThreadBufferIterator: public IIterator< DataBatch > {
public :
  ThreadBufferIterator(IIterator<DataBatch> *base) {
    silent_ = 0;
    itr.get_factory().base_ = base;
    itr.SetParam("buffer_size", "2");
  }
  virtual ~ThreadBufferIterator() {
    itr.Destroy();
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "silent")) silent_ = atoi(val);
    itr.SetParam(name, val);
  }
  virtual void Init(void) {
    CHECK(itr.Init()) << "iterator init fail";
    printf("In batch init.\n");
    if (silent_ == 0) {
      printf("ThreadBufferIterator: buffer_size=%d\n", itr.buf_size);
    }
  }
  virtual void BeforeFirst() {
    itr.BeforeFirst();
  }
  virtual bool Next() {
    if (itr.Next(out_)) {
      return true;
    } else {
      return false;
    }
  }
  virtual const DataBatch &Value() const {
    return out_;
  }
private:
  struct Factory {
  public:
    IIterator<DataBatch> *base_;
  public:
    Factory(void) {
      base_ = NULL;
    }
    inline void SetParam(const char *name, const char *val) {
      base_->SetParam(name, val);
    }
    inline bool Init() {
      base_->Init();
      CHECK(base_->Next()) << "ThreadBufferIterator: input can not be empty";
      oshape_ = base_->Value().data.shape_;
      batch_size_ = base_->Value().batch_size;
      label_width_ = base_->Value().label.size(1);
      for (size_t i = 0; i < base_->Value().extra_data.size(); ++i){
        extra_shape_.push_back(base_->Value().extra_data[i].shape_);
      }
      base_->BeforeFirst();
      return true;
    }
    inline bool LoadNext(DataBatch &val) {
      if (base_->Next()) {
        val.CopyFromDense(base_->Value());
        return true;
      } else {
        return false;
      }
    }
    inline DataBatch Create(void) {
      DataBatch a; a.AllocSpaceDense(oshape_, batch_size_, label_width_, extra_shape_);
      return a;
    }
    inline void FreeSpace(DataBatch &a) {
      a.FreeSpaceDense();
    }
    inline void Destroy() {
      if (base_ != NULL) delete base_;
    }
    inline void BeforeFirst() {
      base_->BeforeFirst();
    }
  private:
    mshadow::index_t batch_size_;
    mshadow::index_t label_width_;
    mshadow::Shape<4> oshape_;
    std::vector<mshadow::Shape<4> > extra_shape_;
  };
private:
  int silent_;
  DataBatch out_;
  utils::ThreadBuffer<DataBatch, Factory> itr;
}; // class ThreadBufferIterator
}  // namespace cxxnet
#endif  // CXXNET_ITER_BATCH_PROC_INL_HPP_
