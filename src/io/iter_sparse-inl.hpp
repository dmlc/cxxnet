#ifndef ITER_SPARSE_INL_HPP
#define ITER_SPARSE_INL_HPP
#pragma once
/*!
 * \file cxxnet_iter_sparse-inl.hpp
 * \brief definition of iterator that relates to sparse data format
 * \author Tianqi Chen
 */
#include "data.h"
#include "../utils/thread_buffer.h"
#include "../utils/io.h"
#include "../utils/utils.h"

namespace cxxnet {
/*! \brief a simple adapter */
class Dense2SparseAdapter: public IIterator<DataBatch> {
public:
  Dense2SparseAdapter(IIterator<DataBatch> *base): base_(base) {
  }
  virtual ~Dense2SparseAdapter(void) {
    delete base_;
  }
  virtual void SetParam(const char *name, const char *val) {
    base_->SetParam(name, val);
  }
  virtual void Init(void) {
    base_->Init();
  }
  virtual void BeforeFirst(void) {
    base_->BeforeFirst();
  }
  virtual bool Next(void) {
    if (base_->Next()) {
      // convert the space
      const DataBatch &din = base_->Value();
      out_.labels = din.labels;
      out_.inst_index = din.inst_index;
      out_.batch_size = din.batch_size;
      out_.num_batch_padd = din.num_batch_padd;
      row_ptr.clear(); row_ptr.push_back(0); data.clear();

      utils::Assert(din.batch_size == din.data.shape[1], "Dense2SparseAdapter: only support 1D input");
      for (mshadow::index_t i = 0; i < din.batch_size; ++i) {
        mshadow::Tensor<mshadow::cpu, 1> row = din.data[0][0][i];
        for (mshadow::index_t j = 0; j < row.shape[0]; ++j) {
          data.push_back(SparseInst::Entry(j, row[j]));
        }
        row_ptr.push_back(row_ptr.back() + row.shape[0]);
      }
      out_.sparse_row_ptr = &row_ptr[0];
      out_.sparse_data = &data[0];
      return true;
    } else {
      return false;
    }
  }
  virtual const DataBatch &Value(void) const {
    return out_;
  }
private:
  // base iterator
  IIterator<DataBatch> *base_;
  // output data
  DataBatch out_;
  // actual content of row ptr
  std::vector<size_t> row_ptr;
  // actual content of sparse data storage
  std::vector<SparseInst::Entry> data;
}; // class Dense2SparseAdapter

/*! \brief adapter to adapt instance iterator into batch iterator */
class SparseBatchAdapter: public IIterator<DataBatch> {
public:
  SparseBatchAdapter(IIterator<SparseInst> *base): base_(base) {
    batch_size_ = 0;
    scale_fvalue_ = 1.0f;
  }
  virtual ~SparseBatchAdapter(void) {
    delete base_;
  }
  virtual void SetParam(const char *name, const char *val) {
    base_->SetParam(name, val);
    if (!strcmp(name, "round_batch")) round_batch_ = atoi(val);
    if (!strcmp(name, "batch_size")) batch_size_ = (unsigned)atoi(val);
    if (!strcmp(name, "scale_fvalue")) scale_fvalue_ = (float)atof(val);
  }
  virtual void Init(void) {
    base_->Init();
    labels.resize(batch_size_);
    inst_index.resize(batch_size_);
    out_.labels = &labels[0];
    out_.inst_index = &inst_index[0];
    printf("SparseBatchAdapter: batch_size=%u,scale=%f\n", batch_size_, scale_fvalue_);
  }
  virtual void BeforeFirst(void) {
    if (round_batch_ == 0 || num_overflow_ == 0) {
      // otherise, we already called before first
      base_->BeforeFirst();
    } else {
      num_overflow_ = 0;
    }
  }
  virtual bool Next(void) {
    data.clear();
    row_ptr.clear(); row_ptr.push_back(0);
    out_.num_batch_padd = 0;
    out_.batch_size = batch_size_;

    while (base_->Next()) {
      this->Add(base_->Value());
      if (row_ptr.size() > batch_size_) {
        out_.sparse_row_ptr = &row_ptr[0];
        out_.sparse_data = &data[0];
        return true;
      }
    }

    if (row_ptr.size() == 1) return false;
    out_.num_batch_padd = batch_size_ + 1 - row_ptr.size();
    if (round_batch_ != 0) {
      num_overflow_ = 0;
      base_->BeforeFirst();
      for (; row_ptr.size() <= batch_size_;  ++num_overflow_) {
        utils::Assert(base_->Next(), "number of input must be bigger than batch size");
        this->Add(base_->Value());
      }
    } else {
      while (row_ptr.size() <= batch_size_) {
        row_ptr.push_back(row_ptr.back());
      }
    }
    utils::Assert(row_ptr.size() == batch_size_ + 1, "BUG");
    out_.sparse_row_ptr = &row_ptr[0];
    out_.sparse_data = &data[0];
    return true;
  }
  virtual const DataBatch &Value(void) const {
    return out_;
  }
private:
  inline void Add(const SparseInst &line) {
    labels[row_ptr.size() - 1] = line.label;
    inst_index[row_ptr.size() - 1] = line.index;
    for (unsigned i = 0; i < line.length; ++ i) {
      SparseInst::Entry e = line[i];
      e.fvalue  *= scale_fvalue_;
      data.push_back(e);
    }
    row_ptr.push_back(row_ptr.back() + line.length);
  }
private:
  // base iterator
  IIterator<SparseInst> *base_;
  // batch size
  mshadow::index_t batch_size_;
  // scaling factor
  float scale_fvalue_;
  // use round batch mode to reuse further things
  int round_batch_;
  // number of overflow items
  int num_overflow_;
private:
  // output data
  DataBatch out_;
  // actual content of row ptr
  std::vector<size_t> row_ptr;
  std::vector<float>  labels;
  std::vector<unsigned> inst_index;
  // actual content of sparse data storage
  std::vector<SparseInst::Entry> data;
}; // class SparseBatchAdapter

/*! \brief constant size binary page that can */
struct SparseInstObj {
private:
  unsigned *dptr;
  unsigned length;
public:
  SparseInstObj(const SparseInst &line) {
    utils::Assert(sizeof(SparseInst::Entry) % sizeof(unsigned) == 0, "TODO");
    dptr = new unsigned[(line.length * sizeof(SparseInst::Entry) + 2 * sizeof(unsigned)) / sizeof(unsigned) ];
    this->label() = line.label;
    this->index() = line.index;
    this->length = line.length;
    memcpy(dptr + 2, line.data, sizeof(SparseInst::Entry)*line.length);

  }
  SparseInstObj(const utils::BinaryPage::Obj &obj) {
    this->dptr = (unsigned *)obj.dptr;
    this->length = (obj.sz - 2 * sizeof(unsigned)) / sizeof(SparseInst::Entry);
  }
  inline void FreeSpace(void) {
    delete [] dptr;
  }
  inline float &label(void) {
    return *(float *)(dptr + 0);
  }
  inline unsigned &index(void) {
    return *(unsigned *)(dptr + 1);
  }
  /*! \brief get sparse inst from this one */
  inline SparseInst GetInst(void) {
    SparseInst inst;
    inst.label = this->label();
    inst.index = this->index();
    inst.length = this->length;
    inst.data  = (const SparseInst::Entry *)(dptr + 2);
    return inst;
  }
  /*! \brief get binary obj view of this object */
  inline utils::BinaryPage::Obj GetObj(void) {
    return utils::BinaryPage::Obj(dptr, length * sizeof(SparseInst::Entry) + 2 * sizeof(unsigned));
  }
}; // struct SparseInstObj


/*! \brief thread buffer iterator */
class ThreadSparsePageIterator: public IIterator< SparseInst > {
public:
  ThreadSparsePageIterator(void) {
    silent_ = 0;
    path_bin_ = "data.bin";
    itr.SetParam("buffer_size", "4");
    page_.page = NULL;
    isend = false;
  }
  virtual ~ThreadSparsePageIterator(void) {
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "path_data")) path_bin_ = val;
    if (!strcmp(name, "silent")) silent_ = atoi(val);
  }
  virtual void Init(void) {
    if (silent_ == 0) {
      printf("ThreadSparsePageIterator: bin=%s\n", path_bin_.c_str());
    }
    itr.get_factory().fi.Open(path_bin_.c_str(), "rb");
    itr.Init();
    this->BeforeFirst();
  }
  virtual void BeforeFirst(void) {
    itr.BeforeFirst();
    isend = false;
    utils::Assert(this->LoadNextPage(), "BUG");
  }
  virtual bool Next(void) {
    while (page_.page == NULL || ptop_ >= page_.page->Size() || isend) {
      if (!this->LoadNextPage()) return false;
    }
    out_ = SparseInstObj((*page_.page)[ ptop_ ]).GetInst();
    ++ ptop_;
    return true;
  }
  virtual const SparseInst &Value(void) const {
    return out_;
  }
  inline bool LoadNextPage(void) {
    ptop_ = 0;
    bool ret = itr.Next(page_);
    isend = !ret;
    return ret;
  }
protected:
  // output data
  SparseInst out_;
  bool isend;
  // silent
  int silent_;
  // prefix path of binary buffer
  std::string path_bin_;
private:
  struct PagePtr {
    utils::BinaryPage *page;
  };
  struct Factory {
  public:
    utils::StdFile fi;
  public:
    Factory() {}
    inline bool Init() {
      return true;
    }
    inline void SetParam(const char *name, const char *val) {}
    inline bool LoadNext(PagePtr &val) {
      return val.page->Load(fi);
    }
    inline PagePtr Create(void) {
      PagePtr a; a.page = new utils::BinaryPage();
      return a;
    }
    inline void FreeSpace(PagePtr &a) {
      delete a.page;
    }
    inline void Destroy() {
    }
    inline void BeforeFirst() {
      fi.Seek(0);
    }
  };
protected:
  PagePtr page_;
  int     ptop_;
  utils::ThreadBuffer<PagePtr, Factory> itr;
}; // class ThreadSparsePageIterator
}; // namespace cxxnet
#endif
