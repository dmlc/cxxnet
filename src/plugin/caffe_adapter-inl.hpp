#ifndef CXXNET_CAFFE_ADAPTER_INL_HPP
#define CXXNET_CAFEE_ADAPTER_INL_HPP
#pragma once
/*!
 * \file cxxnet_caffee_adapter-inl.hpp
 * \brief try to adapt caffe layers, this code comes as plugin of cxxnet, and by default not included in the code.
 * \author Tianqi Chen
 */
#include <climits>
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "mshadow/tensor.h"
#include "mshadow/tensor_container.h"
#include <google/protobuf/text_format.h>

namespace cxxnet {
namespace layer {
using namespace caffe;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;
/*!
 * \brief adapter from caffe, will cost a extra blob memory,
 *        but allows some correct comparisons
 */
template<typename xpu>
class CaffeLayer: public ILayer<xpu>{
 public:
  CaffeLayer(){
    this->base_ = NULL;
    this->mode_ = -1;
    this->blb_in_ = NULL;
    this->blb_out_ = NULL;
  }
  virtual ~CaffeLayer(void) {
    this->FreeSpace();
    if (blb_in_ != NULL)  delete blb_in_;
    if (blb_out_ != NULL) delete blb_out_;
  }

  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    this->stream_ = stream;
  }

  virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {
    const std::vector<boost::shared_ptr<caffe::Blob<real_t> > > &blobs = base_->blobs();
    for(size_t i = 0; i < blobs.size(); ++ i) {
      // Assume that blobs do not change
      char tag[ 256 ];
      sprintf(tag, "blob%d", (int)i);
      index_t count = blobs[i]->count();
      if (xpu::kDevCPU) {
        mshadow::Tensor<xpu,1> weight(blobs[i]->mutable_cpu_data(), Shape1(count));
        mshadow::Tensor<xpu,1> grad(blobs[i]->mutable_cpu_diff(), Shape1(count));
        weight.set_stream(stream_);
        grad.set_stream(stream_);
        pvisitor->Visit(tag, weight, grad);
      }else{
        mshadow::Tensor<xpu,1> weight(blobs[i]->mutable_gpu_data(), Shape1(count));
        mshadow::Tensor<xpu,1> grad(blobs[i]->mutable_gpu_diff(), Shape1(count));
        weight.set_stream(stream_);
        grad.set_stream(stream_);
        pvisitor->Visit(tag, weight, grad);
      }
    }
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    for (index_t i = 0; i < nodes_in.size(); ++i){
      mshadow::Shape<4> shape_in = nodes_in[i]->data.shape_;
      if (xpu::kDevCPU) {
        mshadow::Tensor<xpu,4> tbin(vec_in_[i]->mutable_cpu_data(), shape_in);
        tbin.set_stream(stream_);
        mshadow::Copy(tbin, nodes_in[i]->data, stream_);
      }else{
        mshadow::Tensor<xpu,4> tbin(blb_in_->mutable_gpu_data(), shape_in);
        tbin.set_stream(stream_);
        mshadow::Copy(tbin, nodes_in[i]->data, stream_);
      }
    }
    base_->Forward(vec_in_, &vec_out_);
    for (index_t i = 0; i < nodes_out.size(); ++i){
      mshadow::Shape<4> shape_ou = nodes_out[i]->data.shape_;
      if (xpu::kDevCPU) {
        mshadow::Tensor<xpu,4> tbout(vec_out_[i]->mutable_cpu_data(), shape_ou);
        tbout.set_stream(stream_);
        mshadow::Copy(nodes_out[i]->data, tbout, stream_);
      } else {
        mshadow::Tensor<xpu,4> tbout(vec_out_[i]->mutable_gpu_data(), shape_ou);
        tbout.set_stream(stream_);
        mshadow::Copy(nodes_out[i]->data, tbout, stream_);
      }
    }
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    for (index_t i = 0; i < nodes_out.size(); ++i){
      mshadow::Shape<4> shape_ou = nodes_out[i]->data.shape_;
      if (xpu::kDevCPU) {
        mshadow::Tensor<xpu,4> tbout(vec_out_[i]->mutable_cpu_diff(), shape_ou);
        tbout.set_stream(stream_);
        mshadow::Copy(tbout, nodes_out[i]->data, stream_);
      } else {
        mshadow::Tensor<xpu,4> tbout(vec_out_[i]->mutable_gpu_diff(), shape_ou);
        tbout.set_stream(stream_);
        mshadow::Copy(tbout, nodes_out[i]->data, stream_);
      }
    }

    base_->Backward(vec_out_, prop_grad, &vec_in_);
    if (prop_grad) {
      for (index_t i = 0; i < nodes_in.size(); ++i){
        mshadow::Shape<4> shape_in = nodes_in[i]->data.shape_;
        if (xpu::kDevCPU) {
          mshadow::Tensor<xpu,4> tbin(vec_in_[i]->mutable_cpu_diff(), shape_in);
          tbin.set_stream(stream_);
          mshadow::Copy(nodes_in[i]->data, tbin, stream_);
        } else {
          mshadow::Tensor<xpu,4> tbin(vec_in_[i]->mutable_gpu_diff(), shape_in);
          tbin.set_stream(stream_);
          mshadow::Copy(nodes_in[i]->data, tbin, stream_);
        }
      }  
    }
  }
  
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Assert(mode_ != -1, "CaffeLayer: must specify mode: 0:flatten, 1:conv-channels");
    vec_in_.clear();
    Caffe::SetDevice(0);
    for (index_t i = 0; i < nodes_in.size(); ++i){
      mshadow::Shape<4> ishape = nodes_in[i]->data.shape_;
      if (mode_ == 0) {
        utils::Assert(ishape[0] == 1 && ishape[1] == 1, "the input is not flattened, forget a FlattenLayer?");
        batch_size_ = ishape[2];
        blb_in_  = new caffe::Blob<real_t>(ishape[2], ishape[3], 1, 1);
        blb_out_ = new caffe::Blob<real_t>();
      }else{
        batch_size_ = ishape[0];
        blb_in_  = new caffe::Blob<real_t>(ishape[0], ishape[1], ishape[2], ishape[3]);
        blb_out_ = new caffe::Blob<real_t>();
      }
      vec_in_.push_back(blb_in_);
    }

    vec_out_.clear();
    for (index_t i = 0; i < nodes_out.size(); ++i){
      blb_out_ = new caffe::Blob<real_t>();
      vec_out_.push_back(blb_out_);
    }    
    if (base_ == NULL) {
      base_ = caffe::GetLayer<real_t>(param_);
    }
    
    base_->SetUp(vec_in_, &vec_out_);
    utils::Assert(nodes_out.size() == vec_out_.size(), "CaffeLayer: Number of output inconsistent.");
    for (index_t i = 0; i < nodes_out.size(); ++i){
      if (mode_ == 0 || mode_ == 2) {
        nodes_out[i]->data.shape_ = mshadow::Shape4(1, 1, vec_out_[i]->num(), vec_out_[i]->channels());
      }else{
        nodes_out[i]->data.shape_ = mshadow::Shape4(vec_out_[i]->num(), vec_out_[i]->channels(),
          vec_out_[i]->height(), vec_out_[i]->width());
      }
    }
  }
  virtual void SetParam(const char *name, const char* val) {
    if (!strcmp(name, "proto")) {
      google::protobuf::TextFormat::ParseFromString(std::string(val), &param_);
    }
    if (!strcmp(name, "mode")) {
      mode_ = atoi(val);
    }
    if (!strcmp(name, "dev")) {
      if (!strcmp(val, "cpu")) caffe::Caffe::set_mode(caffe::Caffe::CPU);
      if (!strcmp(val, "gpu")) caffe::Caffe::set_mode(caffe::Caffe::GPU);
    }
  }
  virtual void InitModel(void) {
  }
  virtual void SaveModel(mshadow::utils::IStream &fo) const {
    std::vector<char> buf;
    caffe::LayerParameter lparam = base_->layer_param();
    base_->ToProto(&lparam);            
    int msize = lparam.ByteSize();
    buf.resize(msize);
    fo.Write(&msize, sizeof(int));
    utils::Assert(lparam.SerializeToArray(&buf[0], msize ), "CaffeLayer::SaveModel");
    fo.Write(&buf[0], msize);
  }
  virtual void LoadModel(mshadow::utils::IStream &fi) {
    int msize;
    std::vector<char> buf;
    fi.Read(&msize, sizeof(int));
    buf.resize(msize);
    utils::Assert(fi.Read(&buf[0], msize)!= 0, "CaffeLayer::LoadModel");
    param_.ParseFromArray(&buf[0], msize);
    this->FreeSpace();
    base_ = caffe::GetLayer<real_t>(param_);
  }
 private:
  inline void FreeSpace(void) {
    if (base_ != NULL) delete base_;
    base_ = NULL;
  }
 private:
  /*!\brief mini batch size*/
  int batch_size_;
  /*!\brief whether it is fullc or convolutional layer */
  int mode_;
  /*! \brief caffe's layer parametes */
  caffe::LayerParameter param_;
  /*! \brief caffe's impelementation */
  caffe::Layer<real_t>* base_;
  /*! \brief blob data */
  caffe::Blob<real_t>* blb_in_;
  caffe::Blob<real_t>* blb_out_;
  /*!\ brief stores blb in */
  std::vector< caffe::Blob<real_t>* > vec_in_;
  std::vector< caffe::Blob<real_t>* > vec_out_;
  /*!\ brief the stream_ used for the tensor*/
  mshadow::Stream<xpu> *stream_;
};
}  // namespace layer
}  // namespace cxxnet
#endif
