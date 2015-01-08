#ifndef CXXNET_LAYER_CONVOLUTION_LAYER_INL_HPP_
#define CXXNET_LAYER_CONVOLUTION_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include "../utils/utils.h"

namespace cxxnet {
namespace layer {

template<typename xpu>
class ConvolutionLayer : public ILayer<xpu> {
 public:
  ConvolutionLayer(mshadow::Random<xpu> *p_rnd) 
      : prnd_(p_rnd), wmat_(false), bias_(false), gwmat_(false), gbias_(false) {}
  virtual ~ConvolutionLayer(void) {}
  virtual void SetParam(const char *name, const char* val) {
    param_.SetParam(name, val);
  }
  virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {
    pvisitor->Visit("wmat", wmat_, gwmat_);
    if (param_.no_bias == 0) {
      pvisitor->Visit("bias", bias_, gbias_);
    }
  }
  virtual void InitModel(void) {
    // resize to correct shape, use 2d to store the weight, since we use dot
    wmat_.Resize(mshadow::Shape3(param_.num_group, param_.num_channel / param_.num_group,
                                 param_.num_input_channel / param_.num_group * param_.kernel_height * param_.kernel_width));
    bias_.Resize(mshadow::Shape1(param_.num_channel));    
    param_.RandInitWeight(this->prnd_, wmat_, wmat_.size(2), wmat_.size(1));
    bias_ = param_.init_bias;
    // setup gradient
    gwmat_.Resize(wmat_.shape_);
    gbias_.Resize(bias_.shape_);
    gwmat_ = 0.0f; gbias_ = 0.0f;
  }
  virtual void SaveModel(utils::IStream &fo) const {
    fo.Write(&param_, sizeof(LayerParam));
    wmat_.SaveBinary(fo);
    bias_.SaveBinary(fo);
  }
  virtual void LoadModel(utils::IStream &fi) {
    utils::Check(fi.Read(&param_, sizeof(LayerParam)) != 0,
                  "ConvolutionLayer: LoadModel invalid model file");
    wmat_.LoadBinary(fi);
    bias_.LoadBinary(fi);
    // setup gradient
    gwmat_.Resize(wmat_.shape_);
    gbias_.Resize(bias_.shape_);
    gwmat_ = 0.0f; gbias_ = 0.0f;
  }
  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    gwmat_.set_stream(stream);
    gbias_.set_stream(stream);
    temp_dst_.set_stream(stream);
    temp_col_.set_stream(stream);
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "ConvolutionLayer Layer only support 1-1 connection");
    const index_t ksize_y = static_cast<index_t>(param_.kernel_height);
    const index_t ksize_x = static_cast<index_t>(param_.kernel_width);
    const index_t kstride = static_cast<index_t>(param_.stride);
    mshadow::Shape<4> ishape = nodes_in[0]->data.shape_;
    utils::Check(ishape[1] % param_.num_group == 0,  "input channels must divide group size");
    utils::Check(param_.num_channel % param_.num_group == 0, "output channels must divide group size");
    utils::Check(param_.num_channel > 0, "must set nchannel correctly");
    utils::Check(param_.kernel_height > 0 && param_.kernel_width > 0, "must set kernel_size correctly");
    utils::Check(ksize_x <= ishape[3] && ksize_y <= ishape[2], "kernel size exceed input");    
    mshadow::Shape<4> oshape = mshadow::
        Shape4(ishape[0], param_.num_channel,
                (ishape[2] + 2 * param_.pad_y - ksize_y) / kstride + 1,
                (ishape[3] + 2 * param_.pad_x - ksize_x) / kstride + 1);
    nodes_out[0]->data.shape_ = oshape;

    if (param_.num_input_channel == 0) {
      param_.num_input_channel = static_cast<int>(ishape[1]);
    } else {
      utils::Check(param_.num_input_channel == static_cast<int>(ishape[1]),
                   "ConvolutionLayer: number of input channels is not consistent");
    }    
    this->InitTemp(ishape, oshape);
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;    
    mshadow::Tensor<xpu, 4> &in = nodes_in[0]->data;
    mshadow::Tensor<xpu, 4> &out = nodes_out[0]->data;
    this->InitTemp(in.shape_, out.shape_);
    const index_t nbatch = in.size(0);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      // resize, incase last batch is smaller
      const index_t step = std::min(nstep_, nbatch-i);
      temp_col_.Resize(mshadow::Shape2(shape_colunit_[0], shape_colunit_[1]*step));
      temp_dst_.Resize(mshadow::Shape3(shape_dstunit_[0], shape_dstunit_[1], shape_dstunit_[2]*step));
      
      if (param_.pad_x == 0 && param_.pad_y == 0) {
        temp_col_ = unpack_patch2col(in.Slice(i, i+step), param_.kernel_height, param_.kernel_width, param_.stride);
      }else{
        temp_col_ = unpack_patch2col(pad(in.Slice(i,i+step), param_.pad_y, param_.pad_x),
                                     param_.kernel_height, param_.kernel_width, param_.stride);
      }
      
      const index_t gstride = temp_col_.size(0) / param_.num_group;
      for (int gid = 0; gid < param_.num_group; ++ gid) {
        mshadow::Tensor<xpu,2> tmpc = temp_col_.Slice(gstride * gid, gstride * (gid + 1));
        temp_dst_[gid] = dot(wmat_[gid], tmpc);
      }
      out.Slice(i, i + step) =
          swapaxis<1,0>(reshape(temp_dst_,
                                mshadow::Shape4(param_.num_channel, step, out.size(2), out.size(3))));
    }
    if (param_.no_bias == 0) {
      // add bias, broadcast bias to dim 1: channel
      out += broadcast<1>(bias_, out.shape_);
    }
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> &in = nodes_in[0]->data;
    mshadow::Tensor<xpu, 4> &out = nodes_out[0]->data;
    this->InitTemp(in.shape_, out.shape_);
    const index_t nbatch = in.size(0);
    
    if (param_.no_bias == 0) {
      gbias_ += sumall_except_dim<1>(out);
    }
    
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch-i);
      temp_col_.Resize(mshadow::Shape2(shape_colunit_[0], shape_colunit_[1] * step));
      temp_dst_.Resize(mshadow::Shape3(shape_dstunit_[0], shape_dstunit_[1], shape_dstunit_[2] * step));
      
      temp_dst_ = reshape(swapaxis<1,0>(out.Slice(i, i + step)), temp_dst_.shape_);
      
      if (param_.pad_x == 0 && param_.pad_y == 0) {
        temp_col_ = unpack_patch2col(in.Slice(i, i + step), param_.kernel_height, param_.kernel_width, param_.stride);
      }else{
        temp_col_ = unpack_patch2col(pad(in.Slice(i,i + step),param_.pad_y, param_.pad_x), param_.kernel_height, param_.kernel_width, param_.stride);
      }
      
      const index_t gstride = temp_col_.size(0) / param_.num_group;
      for (int gid = 0; gid < param_.num_group; ++ gid) {
        mshadow::Tensor<xpu,2> tmpc = temp_col_.Slice(gstride * gid, gstride * (gid+1));
        gwmat_[gid] += dot(temp_dst_[gid], tmpc.T());
      }
      
      if (prop_grad) {
        for (int gid = 0; gid < param_.num_group; ++ gid) {
          mshadow::Tensor<xpu,2> tmpc = temp_col_.Slice(gstride * gid, gstride * (gid+1));
          tmpc = dot(wmat_[gid].T(), temp_dst_[gid]);
        }
        
        if (param_.pad_x == 0 && param_.pad_y == 0) {
          in.Slice(i,i+step) = pack_col2patch(temp_col_, in.Slice(i, i + step).shape_, param_.kernel_height, param_.kernel_width, param_.stride);
        }else{
          mshadow::Shape<4> pshape = in.Slice(i, i + step).shape_;
          pshape[2] += 2 * param_.pad_y; pshape[3] += 2 * param_.pad_x;
          in.Slice(i, i + step) = crop(pack_col2patch(temp_col_, pshape, param_.kernel_height, param_.kernel_width, param_.stride), in[i][0].shape_);
        }
      }
    }
  }

 private:
  inline void InitTemp(mshadow::Shape<4> ishape, mshadow::Shape<4> oshape) {
    const index_t ksize_y = static_cast<index_t>(param_.kernel_height);
    const index_t ksize_x = static_cast<index_t>(param_.kernel_width);

    // this is the unit size of eacj temp structure
    shape_colunit_ = mshadow::Shape2(ishape[1] * ksize_y * ksize_x, oshape[2] * oshape[3]);
    shape_dstunit_ = mshadow::Shape3(param_.num_group, param_.num_channel/param_.num_group, oshape[2] * oshape[3]);
    nstep_ = std::max(std::min((index_t)(param_.temp_col_max / shape_colunit_.Size()), ishape[0]), 1U);
    // make nstep more balanced,  nstep will use exactly same number of operations to finish,
    index_t nop = (ishape[0]+nstep_-1) / nstep_;
    nstep_ = (ishape[0] + nop - 1)/ nop;
    utils::Assert(nstep_ > 0, "InitLayer_: nstep check");    
    // helper structure
    temp_col_.Resize(mshadow::Shape2(shape_colunit_[0], shape_colunit_[1] * nstep_));
    temp_dst_.Resize(mshadow::Shape3(shape_dstunit_[0], shape_dstunit_[1], shape_dstunit_[2] * nstep_));    
  }

  /*! \brief random number generator */
  mshadow::Random<xpu> *prnd_;  
  /*! \brief parameters that potentially be useful */
  LayerParam param_;
  /*! \brief weight matrix */
  mshadow::TensorContainer<xpu,3> wmat_;
  /*! \brief bias */
  mshadow::TensorContainer<xpu,1> bias_;
  /*! \brief accumulates the gradient of weight matrix */
  mshadow::TensorContainer<xpu,3> gwmat_;
  /*! \brief accumulates the gradient of bias */
  mshadow::TensorContainer<xpu,1> gbias_;
  /*! \brief temporary data structure to store patches */
  mshadow::TensorContainer<xpu,2> temp_col_;
  /*! \brief temporary data structure to store results */
  mshadow::TensorContainer<xpu,3> temp_dst_;
  /*! \brief shape of column unit */
  mshadow::Shape<2> shape_colunit_;
  /*! \brief shape of dst unit */
  mshadow::Shape<3> shape_dstunit_;
  /*! \brief how many number of batches to be unpacked together */
  mshadow::index_t nstep_;
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_CONVOLUTION_LAYER_INL_HPP_
