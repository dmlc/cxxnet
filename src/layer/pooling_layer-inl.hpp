#include "./layer.h"
#include "./param.h"
#include "mshadow/tensor_container.h"

namespace cxxnet {
namespace layer {

template<typename Reducer, bool scalebysize, typename xpu>
class PoolingLayer : public CommonLayerBase<xpu> {
 public:
  PoolingLayer(mshadow::Random<xpu> *p_rnd, Node<xpu> *p_in, Node<xpu> *p_out)
      : CommonLayerBase<xpu>(p_rnd, p_in, p_out) {}
  virtual ~PoolingLayer(void) {}
  virtual void SetParam(const char *name, const char* val) {
    param_.SetParam( name, val );
  }
  
 protected:
  virtual void InitLayer_(const Node<xpu> &node_in,
                          Node<xpu> *pnode_out) {
    
    const index_t ksize_y   = static_cast<index_t>( param_.kernel_height );
    const index_t ksize_x   = static_cast<index_t>( param_.kernel_width );
    const index_t kstride = static_cast<index_t>( param_.stride );
    utils::Check(param_.kernel_height > 0 && param_.kernel_width > 0, "must set kernel_size correctly" );
    utils::Check(ksize_x <= node_in.data.shape[0] && ksize_y <= node_in.data.shape[1],
                 "kernel size exceed input" );
    
    // conform to same shape style as caffe, though maybe not necessary
    mshadow::Shape<4> oshape = mshadow::
        Shape4( node_in.data.shape[3], node_in.data.shape[2],
                std::min(node_in.data.shape[1] - ksize_y + kstride-1, node_in.data.shape[1] - 1) / kstride + 1,
                std::min(node_in.data.shape[0] - ksize_x + kstride-1, node_in.data.shape[0] - 1) / kstride + 1);
    tmp_.Resize(oshape); pnode_out->data.shape = oshape;
  }
  virtual void Forward_(bool is_train,
                        Node<xpu> *pnode_in,
                        Node<xpu> *pnode_out) {
    using namespace mshadow::expr;
    const int ksize_y = param_.kernel_height;
    const int ksize_x = param_.kernel_width;
    mshadow::Shape<2> pshape = pnode_out->data[0][0].shape;
    if (!scalebysize) {
      tmp_ = pool<Reducer>(pnode_in->data, pshape, ksize_y, ksize_x, param_.stride);
    }else{
      tmp_ = pool<Reducer>(pnode_in->data, pshape, ksize_y, ksize_x, param_.stride) * (1.0f/(ksize_y*ksize_x) );
    }
    mshadow::Copy(pnode_out->data, tmp_);
  }
  virtual void Backprop_(bool prop_grad,
                         Node<xpu> *pnode_in,
                         Node<xpu> *pnode_out) {
    using namespace mshadow::expr;
    if (prop_grad) {
      const int ksize_y = param_.kernel_height;
      const int ksize_x = param_.kernel_width;
      if( !scalebysize ){
        pnode_in->data = unpool<Reducer>(pnode_in->data, tmp_, pnode_out->data, ksize_y, ksize_x, param_.stride);
      }else{
        pnode_in->data = unpool<Reducer>(pnode_in->data, tmp_, pnode_out->data, ksize_y, ksize_x, param_.stride) * (1.0f / (ksize_y * ksize_x));
      }
    }
  }

 private:
  /*! \brief parameters that potentially be useful */
  LayerParam param_;
  /*! \brief pooled result */
  mshadow::TensorContainer<xpu, 4> tmp_;
};   // class PoolingLayer
}  // namespace layer
}  // namespace cxxnet
