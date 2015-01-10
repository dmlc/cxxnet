#ifndef CXXNET_LAYER_PAIRTEST_LAYER_INL_HPP_
#define CXXNET_LAYER_PAIRTEST_LAYER_INL_HPP_
/*!
 * \file pairtest-inl.hpp
 * \brief module to do pairtest, used to compare layer implementations
 * \author Tianqi Chen, Bing Xu
 */
#include <mshadow/tensor.h>
#include "./layer.h"
#include "./visitor.h"

namespace cxxnet {
namespace layer {
template<typename xpu>
class PairTestLayer : public ILayer<xpu> {
 public:
  PairTestLayer(ILayer<xpu> *master, ILayer<xpu> *slave) 
      : master_(master) {
    slave_.layer = slave;
  }
  // virtual destructor
  virtual ~PairTestLayer(void) {
    delete master_;
    delete slave_.layer;
    for (size_t i = 0; i < snodes_in_.size(); ++i) {
      snodes_in_[i].FreeSpace();
    }
    for (size_t i = 0; i < snodes_out_.size(); ++i) {
      snodes_out_[i].FreeSpace();
    }
  }
  virtual bool AllowSharing(void) const {
    return false;
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    master_->InitConnection(nodes_in, nodes_out, p_cstate);
    snodes_in_.resize(nodes_in.size());
    snodes_out_.resize(nodes_out.size());
    slave_.nodes_in.resize(nodes_in.size());
    slave_.nodes_out.resize(nodes_out.size());
    for (size_t i = 0; i < nodes_in.size(); ++i) {
      snodes_in_[i].data.shape_ = nodes_in[i]->data.shape_;
      mshadow::AllocSpace(&snodes_in_[i].data);
      slave_.nodes_in[i] = &snodes_in_[i];
    }
    slave_.layer->InitConnection(slave_.nodes_in,
                                 slave_.nodes_out,
                                 &slave_.state);    
    for (size_t i = 0; i < nodes_out.size(); ++i) {
      utils::Check(snodes_out_[i].data.shape_ == nodes_out[i]->data.shape_,
                   "PairTestLayer.InitConnection: shape inconsistent");          
      mshadow::AllocSpace(&snodes_out_[i].data);
      slave_.nodes_out[i] = &snodes_out_[i];
    }
  }
  
  virtual void OnBatchSizeChanged(const std::vector<Node<xpu>*> &nodes_in,
                                  const std::vector<Node<xpu>*> &nodes_out,
                                  ConnectState<xpu> *p_cstate) {
    master_->OnBatchSizeChanged(nodes_in, nodes_out, p_cstate);
    for (size_t i = 0; i < nodes_in.size(); ++i) {
      snodes_in_[i].data.shape_ = nodes_in[i]->data.shape_;
    }
    for (size_t i = 0; i < nodes_out.size(); ++i) {
      snodes_out_[i].data.shape_ = nodes_out[i]->data.shape_;
    }
    slave_.layer->OnBatchSizeChanged(slave_.nodes_in,
                                     slave_.nodes_out,
                                     &slave_.state);
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    this->Cmp("Before-Forward:weight", "weight");
    master_->Forward(is_train, nodes_in, nodes_out, p_cstate);
    for (size_t i = 0; i < nodes_in.size(); ++i) {
      mshadow::Copy(snodes_in_[i].data, nodes_in[i]->data,
                    nodes_in[i]->data.stream_);
    }
    slave_.layer->Forward(is_train,
                          slave_.nodes_in,
                          slave_.nodes_out,
                          &slave_.state);
    utils::Check(nodes_out.size() == snodes_out_.size(),
                 "Forward: size mismatch");
    for (size_t i = 0; i < nodes_out.size(); ++i) {
      CmpResult(nodes_out[i]->data.FlatTo2D(),
                snodes_out_[i].data.FlatTo2D(), "Forward");
    }
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    master_->Backprop(prop_grad, nodes_in, nodes_out, p_cstate);
    for (size_t i = 0; i < nodes_in.size(); ++i) {
      mshadow::Copy(snodes_out_[i].data, nodes_out[i]->data,
                    nodes_out[i]->data.stream_);
    }
    slave_.layer->Backprop(prop_grad,
                           slave_.nodes_in,
                           slave_.nodes_out,
                           &slave_.state);
    this->Cmp("After-Backprop:grad", "grad");
    if (prop_grad) {
      utils::Check(nodes_in.size() == snodes_in_.size(),
                   "Backprop: size mismatch");
      for (size_t i = 0; i < nodes_in.size(); ++i) {
        CmpResult(nodes_in[i]->data.FlatTo2D(),
                  snodes_in_[i].data.FlatTo2D(), "Forward");
      }
    }
  }
  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    master_->SetStream(stream);
    slave_.SetStream(stream);
  }
  virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {
    master_->ApplyVisitor(pvisitor);
    slave_.layer->ApplyVisitor(pvisitor);
  }
  virtual void SetParam(const char *name, const char* val) {
    master_->SetParam(name, val);
    slave_.layer->SetParam(name, val);
    if (!strncmp( name, "master:", 7)) {
      master_->SetParam(name + 7, val);
    }
    if (!strncmp( name, "slave:", 6)) {
      slave_.layer->SetParam(name + 6, val);
    }
  }
  virtual void InitModel(void) {
    master_->InitModel();
    slave_.layer->InitModel();
    this->Sync("weight");
  }
  virtual void SaveModel(utils::IStream &fo) const {
    master_->SaveModel(fo);
    slave_.layer->SaveModel(fo);
  }
  virtual void LoadModel(utils::IStream &fi) {
    master_->LoadModel(fi);
    slave_.layer->SaveModel(fi);
  }

 private:
  ILayer<xpu> *master_;
  Connection<xpu> slave_;
  std::vector<Node<xpu> > snodes_in_, snodes_out_;
  // synchronize weight or gradient
  inline void Sync(const char *dtype) {
    GetWeightVisitor<xpu> vg(dtype);
    master_->ApplyVisitor(&vg);
    SetWeightVisitor<xpu> vs(vg.data, dtype);
    slave_.layer->ApplyVisitor(&vs);
  }
  inline void Cmp(const char *method, const char *dtype) {
    GetWeightVisitor<xpu> vm(dtype), vs(dtype);
    master_->ApplyVisitor(&vm);
    slave_.layer->ApplyVisitor(&vs);
    utils::Check(vm.data.size() == vs.data.size(),
                 "%s: number of %s mismatch", method, dtype);
    for (size_t i = 0; i < vm.data.size(); ++i) {
      CmpResult(vm.data[i], vs.data[i], method);
    }
  }
  inline static void CmpResult(mshadow::Tensor<xpu, 2> dmaster,
                               mshadow::Tensor<xpu, 2> dslave,
                               const char *tag) {
    mshadow::TensorContainer<cpu, 2> tmst(false), tslv(false);
    mshadow::Stream<xpu> stream;
    tmst.Resize(dmaster.shape_);
    tslv.Resize(dslave.shape_);
    mshadow::Copy(tmst, dmaster, &stream);
    mshadow::Copy(tslv, dslave, &stream);
    index_t count = tmst.shape_.Size();
    double diff = 0.0, ssum = 0.0, maxdiff = 0.0;
    index_t mxidx = 0;
    for (index_t i = 0; i < count; ++i) {
      double d = std::abs(tmst.dptr_[i] - tslv.dptr_[i]);
      if (d > maxdiff) {
        maxdiff = d; mxidx = i;
      }
      diff += d;
      ssum += std::abs(tmst.dptr_[i]);
    }
    // relative absolute error
    double rerr = diff / ssum;
    if (rerr > 1e-5 || diff != diff) {
      fprintf(stderr, "%s: err=%f, maxd[%u]=%f, diff=%f, ssum=%f\n", tag, rerr, mxidx, maxdiff, diff, ssum);
    }
  }
  
};
}  // namespace layer
}  // namespace cxxnet
#endif  // CXXNET_LAYER_PAIRTEST_LAYER_INL_HPP_
