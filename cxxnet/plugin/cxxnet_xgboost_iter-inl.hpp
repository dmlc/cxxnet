#ifndef CXXNET_XGBOOST_ITER_INL_HPP
#define CXXNET_XGBOOST_ITER_INL_HPP
#pragma once
/*!
 * \file cxxnet_xgboost_iter-inl.hpp
 * \brief adapt input format of xgboost DMatrix format into cxxnet
 * \author Tianqi Chen
 */

#include <string>
#include <cstring>
#include "../io/cxxnet_data.h"
#include "io/io.h"
#include "io/page_dmatrix-inl.hpp"

namespace cxxnet {
class XGBoostPageIterator : public IIterator<SparseInst> {
 public:
  XGBoostPageIterator(void) {
    silent_ = 0;
    index_offset_ = 0;
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "path_data")) fname_ = val;
    if (!strcmp(name, "silent")) silent_ = atoi(val);    
  }
  virtual void Init( void ) {
    ::xgboost::utils::FileStream fs(utils::FopenCheck(fname_.c_str(), "rb"));
    dmat_.Load(fs, false, fname_.c_str(), true);
    iter_ = dmat_.fmat()->RowIterator();
  }
  virtual void BeforeFirst( void ) {
    iter_->BeforeFirst();
    top_ = 0;
    batch_.size = 0;
  }        
  virtual bool Next( void ) {
    if (top_ >= batch_.size) {
      if (!iter_->Next()) return false;
      batch_ = iter_->Value();
      top_ = 0;
    }
    ++top_;
    ::xgboost::RowBatch::Inst inst = batch_[top_];
    size_t ridx = batch_.base_rowid + top_;
    entry_.clear();
    for (unsigned i = 0; i < inst.length; ++i) {
      entry_.push_back(SparseInst::Entry(inst[i].index, inst[i].fvalue));
    }
    out_.data = xgboost::BeginPtr(entry_);
    out_.index = ridx + index_offset_;
    out_.length = static_cast<unsigned>(entry_.size());
    if (dmat_.info.labels.size() != 0) out_.label = dmat_.info.labels[ridx];

    return true;
  }
  virtual const SparseInst& Value(void) const {
    return out_;
  }
 private:
  int silent_;
  std::string fname_;
  size_t index_offset_;
  // internal data
  size_t top_;
  SparseInst out_;
  ::xgboost::RowBatch batch_;
  std::vector<SparseInst::Entry> entry_;
  /*! \brief internal iterator */
  ::xgboost::io::DMatrixPage dmat_;
  ::xgboost::utils::IIterator< ::xgboost::RowBatch > *iter_;
};
}
#endif
