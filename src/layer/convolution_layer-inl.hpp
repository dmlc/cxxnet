#ifndef CXXNET_LAYER_CONVOLUTION_LAYER_INL_HPP_
#define CXXNET_LAYER_CONVOLUTION_LAYER_INL_HPP_

#include "./layer.h"
#include "./param.h"
#include "mshadow/tensor_container.h"
#include "../utils/utils.h"

namespace cxxnet {
namespace layer {

template<typename xpu>
class ConvolutionLayer : public CommonLayerBase<xpu> {
 public:
  ConvolutionLayer(mshadow::Random<xpu> *p_rnd, Node<xpu> *p_in, Node<xpu> *p_out)
      : CommonLayerBase<xpu>(p_rnd, p_in, p_out) {}
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
                                 this->pin_->data.shape[2] / param_.num_group * param_.kernel_height * param_.kernel_width));
    bias_.Resize(mshadow::Shape1(param_.num_channel));    
    param_.RandInitWeight(this->prnd_, wmat_, wmat_.shape[0], wmat_.shape[1]);
    bias_ = param_.init_bias;
    // setup gradient
    gwmat_.Resize(wmat_.shape);
    gbias_.Resize(bias_.shape);
    gwmat_ = 0.0f; gbias_ = 0.0f;
  }
  virtual void SaveModel(mshadow::utils::IStream &fo) const {
    fo.Write(&param_, sizeof(LayerParam));
    wmat_.SaveBinary(fo);
    bias_.SaveBinary(fo);
  }
  virtual void LoadModel(mshadow::utils::IStream &fi) {
    utils::Check(fi.Read(&param_, sizeof(LayerParam)) != 0,
                  "CommonLayerBase: LoadModel invalid model file");
    wmat_.LoadBinary(fi);
    bias_.LoadBinary(fi);
    // setup gradient
    gwmat_.Resize(wmat_.shape);
    gbias_.Resize(bias_.shape);
    gwmat_ = 0.0f; gbias_ = 0.0f;
  }

 protected:
  virtual void InitLayer_(const Node<xpu> &node_in,
                          Node<xpu> *pnode_out) {
    const index_t ksize_y = static_cast<index_t>(param_.kernel_height);
    const index_t ksize_x = static_cast<index_t>(param_.kernel_width);
    const index_t kstride = static_cast<index_t>(param_.stride);
    utils::Check(node_in.data.shape[2] % param_.num_group == 0,  "input channels must divide group size");
    utils::Check(param_.num_channel % param_.num_group == 0, "output channels must divide group size");
    utils::Check(param_.num_channel > 0, "must set nchannel correctly");
    utils::Check(param_.kernel_height > 0 && param_.kernel_width > 0, "must set kernel_size correctly");
    utils::Check(ksize_x <= node_in.data.shape[0] && ksize_y <= node_in.data.shape[1], "kernel size exceed input");
    
    mshadow::Shape<4> oshape = mshadow::
        Shape4(node_in.data.shape[3], param_.num_channel,
                (node_in.data.shape[1] + 2 * param_.pad_y - ksize_y)/kstride + 1,
                (node_in.data.shape[0] + 2 * param_.pad_x - ksize_x)/kstride + 1);
    pnode_out->data.shape = oshape;
    // this is the unit size of eacj temp structure
    shape_colunit_ = mshadow::Shape2(node_in.data.shape[2] * ksize_y * ksize_x, oshape[1] * oshape[0]);
    shape_dstunit_ = mshadow::Shape3(param_.num_group, param_.num_channel/param_.num_group, oshape[1]*oshape[0]);
    nstep_ = std::max(std::min((index_t)(param_.temp_col_max / shape_colunit_.Size()), node_in.data.shape[3]), 1U);
    // make nstep more balanced,  nstep will use exactly same number of operations to finish,
    index_t nop = (node_in.data.shape[3]+nstep_-1) / nstep_;
    nstep_ = (node_in.data.shape[3] + nop - 1)/ nop;
    utils::Assert(nstep_ > 0, "InitLayer_: nstep check");
    
    // helper structure
    temp_col_.Resize(mshadow::Shape2(shape_colunit_[1], shape_colunit_[0] * nstep_));
    temp_dst_.Resize(mshadow::Shape3(shape_dstunit_[2], shape_dstunit_[1], shape_dstunit_[0] * nstep_));
    
    if (param_.silent == 0) {
      utils::Printf("ConvolutionLayer: nstep=%u\n", nstep_);
    }
  }  
  virtual void Forward_(bool is_train,
                        Node<xpu> *pnode_in,
                        Node<xpu> *pnode_out) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> &in = pnode_in->data;
    mshadow::Tensor<xpu, 4> &out = pnode_out->data;

    const index_t nbatch = in.shape[3];
    for (index_t i = 0; i < nbatch; i += nstep_) {
      // resize, incase last batch is smaller
      const index_t step = std::min(nstep_, nbatch-i);
      temp_col_.Resize(mshadow::Shape2(shape_colunit_[1], shape_colunit_[0]*step));
      temp_dst_.Resize(mshadow::Shape3(shape_dstunit_[2], shape_dstunit_[1], shape_dstunit_[0]*step));
      
      if (param_.pad_x == 0 && param_.pad_y == 0) {
        temp_col_ = unpack_patch2col(in.Slice(i, i+step), param_.kernel_height, param_.kernel_width, param_.stride);
      }else{
        temp_col_ = unpack_patch2col(pad(in.Slice(i,i+step),param_.pad_y, param_.pad_x), param_.kernel_height, param_.kernel_width, param_.stride);
      }
      
      const index_t gstride = temp_col_.shape[1] / param_.num_group;
      for (int gid = 0; gid < param_.num_group; ++ gid) {
        mshadow::Tensor<xpu,2> tmpc = temp_col_.Slice(gstride*gid, gstride*(gid+1));
        temp_dst_[ gid ] = dot(wmat_[gid], tmpc);
      }
      out.Slice(i,i+step)  = swapaxis<2,3>(reshape(temp_dst_, mshadow::Shape4(param_.num_channel, step, out.shape[1], out.shape[0])));
    }
    if (param_.no_bias == 0) {
      // add bias, broadcast bias to dim 2: channel
      out += broadcast<2>(bias_, out.shape);
    }
  }
  virtual void Backprop_(bool prop_grad,
                         Node<xpu> *pnode_in,
                         Node<xpu> *pnode_out) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> &in = pnode_in->data;
    mshadow::Tensor<xpu, 4> &out = pnode_out->data;
    const index_t nbatch = in.shape[3];
    
    if (param_.no_bias == 0) {
      gbias_ += sumall_except_dim<2>(out);
    }
    
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch-i);
      temp_col_.Resize(mshadow::Shape2(shape_colunit_[1], shape_colunit_[0]*step));
      temp_dst_.Resize(mshadow::Shape3(shape_dstunit_[2], shape_dstunit_[1], shape_dstunit_[0]*step));
      
      temp_dst_ = reshape(swapaxis<2,3>(out.Slice(i,i+step)), temp_dst_.shape);
      
      if (param_.pad_x == 0 && param_.pad_y == 0) {
        temp_col_ = unpack_patch2col(in.Slice(i, i+step), param_.kernel_height, param_.kernel_width, param_.stride);
      }else{
        temp_col_ = unpack_patch2col(pad(in.Slice(i,i+step),param_.pad_y, param_.pad_x), param_.kernel_height, param_.kernel_width, param_.stride);
      }
      
      const index_t gstride = temp_col_.shape[1] / param_.num_group;
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
          in.Slice(i,i+step) = pack_col2patch(temp_col_, in.Slice(i,i+step).shape, param_.kernel_height, param_.kernel_width, param_.stride);
        }else{
          mshadow::Shape<4> pshape = in.Slice(i,i+step).shape; pshape[0] += 2 * param_.pad_y; pshape[1] += 2 * param_.pad_x;
          in.Slice(i,i+step) = crop(pack_col2patch(temp_col_, pshape, param_.kernel_height, param_.kernel_width, param_.stride), in[i][0].shape);
        }
      }
    }
  }

 private:
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
