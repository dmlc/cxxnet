#ifndef INSANITY_POOLING_LAYER_INL_HPP
#define INSANITY_POOLING_LAYER_INL_HPP
#pragma once

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"

namespace mshadow {
namespace expr {

template<typename Reducer, typename SrcExp, typename MaskExp, typename DType, int srcdim>
struct InsanityPoolingExp :
  public MakeTensorExp<InsanityPoolingExp<Reducer, SrcExp, MaskExp, DType, srcdim>,
                       SrcExp, srcdim, DType> {
  const SrcExp &src_;
  const MaskExp &mask_;
  index_t ksize_y_;
  index_t ksize_x_;
  index_t kstride_;
  index_t src_height_;
  index_t src_width_;
  DType p_keep_;
  InsanityPoolingExp(const SrcExp &src, const MaskExp &mask,
                     index_t ksize_y, index_t ksize_x, index_t kstride, DType p_keep)
    : src_(src), mask_(mask),
      ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride), p_keep_(p_keep) {
      Shape<srcdim> sshape = ShapeCheck<srcdim, SrcExp>::Check(src_);
      Shape<srcdim> smshape = ShapeCheck<srcdim, MaskExp>::Check(mask_);
      utils::Check(sshape == smshape, "Incorrect shape");
      utils::Check(sshape[srcdim - 1] >= ksize_x && sshape[srcdim - 2] >= ksize_y,
                   "InsanityPoolingExp: kernel must be smaller than image");
      this->src_height_ = sshape[srcdim - 2];
      this->src_width_  = sshape[srcdim - 1];
      this->shape_ = sshape;
      this->shape_[srcdim - 2] = std::min(src_height_ - ksize_y + kstride - 1, src_height_ - 1) / kstride + 1,
      this->shape_[srcdim - 1] = std::min(src_width_ - ksize_x + kstride-1, src_width_ - 1) / kstride + 1;
  }
}; // struct InsanityPoolingExp

template<typename Reducer, typename SrcExp, typename MaskExp, typename DType, int etype>
inline InsanityPoolingExp<Reducer, SrcExp, MaskExp, DType, ExpInfo<SrcExp>::kDim>
insanity_pool(const Exp<SrcExp, DType, etype> &src,
              const Exp<MaskExp, DType, etype> &mask,
              index_t ksize_y, index_t ksize_x, index_t kstride, DType p_keep) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>::Error_Expression_Does_Not_Meet_Dimension_Req();
  return InsanityPoolingExp<Reducer, SrcExp, MaskExp, DType, ExpInfo<SrcExp>::kDim>
    (src.self(), mask.self(), ksize_y, ksize_x, kstride, p_keep);
}


template<typename Reducer, typename SrcExp, typename MaskExp, typename DType, int srcdim>
struct Plan<InsanityPoolingExp<Reducer, SrcExp, MaskExp, DType, srcdim>, DType> {
  public:
    explicit Plan(const InsanityPoolingExp<Reducer, SrcExp, MaskExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)),
        mask_(MakePlan(e.mask_)),
        ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_), kstride_(e.kstride_),
        src_height_(e.src_height_), src_width_(e.src_width_),
        new_height_(e.shape_[srcdim - 2]),
        p_keep_(e.p_keep_), delta_((1.0f - e.p_keep_) / 4.0f) {}

    MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
      using namespace std;
      const index_t py = i % new_height_;
      const index_t y_start = py * kstride_;
      const index_t y_end = min(y_start + ksize_y_, src_height_);
      const index_t px = j;
      const index_t x_start = px * kstride_;
      const index_t x_end = min(x_start + ksize_x_, src_width_);
      const index_t c = i / new_height_;

      DType res; Reducer::SetInitValue(res);
      for (index_t y = y_start; y < y_end; ++y) {
        for (index_t x = x_start; x < x_end; ++x) {
          index_t loc_y = y;
          index_t loc_x = x;
          DType flag = mask_.Eval(c * src_height_ + y, x);
          if (flag < p_keep_) {
            ;
          } else if (flag < p_keep_ + delta_) {
            loc_y = loc_y > 0 ? loc_y - 1 : loc_y;
          } else if (flag < p_keep_ + delta_ * 2.0f) {
            loc_y = loc_y + 1 < src_height_ ? loc_y + 1 : src_height_ - 1;
          } else if (flag < p_keep_ + delta_ * 3.0f) {
            loc_x = loc_x > 0 ? loc_x - 1 : loc_x;
          } else {
            loc_x = loc_x + 1 < src_width_ ? loc_x + 1 : src_width_ - 1;
          }
          Reducer::Reduce(res, src_.Eval(c * src_height_ + loc_y, loc_x));
        }
      }
      return res;
    }
  private:
    Plan<SrcExp, DType> src_;
    Plan<MaskExp, DType> mask_;
    const index_t ksize_y_, ksize_x_, kstride_;
    const index_t src_height_, src_width_;
    const index_t new_height_;
    const DType p_keep_;
    const DType delta_;
}; // struct Plan

} // namespace expr
} // namespace mshadow


namespace mshadow {
namespace expr {

template<typename Reducer, typename SrcExp, typename MaskExp, typename DType, int srcdim>
struct InsanityUnPoolingExp:
  public MakeTensorExp<InsanityUnPoolingExp<Reducer, SrcExp, MaskExp, DType, srcdim>,
          SrcExp, srcdim, DType> {
  const SrcExp &data_src_;
  const SrcExp &data_pooled_;
  const SrcExp &grad_pooled_;
  const MaskExp &mask_;
  index_t pshape_y_;
  index_t pshape_x_;
  index_t ksize_y_;
  index_t ksize_x_;
  index_t kstride_;
  DType p_keep_;
  InsanityUnPoolingExp(const SrcExp &data_src,
                       const SrcExp &data_pooled,
                       const SrcExp &grad_pooled,
                       const MaskExp &mask,
                       index_t ksize_y, index_t ksize_x, index_t kstride, DType p_keep)
    : data_src_(data_src), data_pooled_(data_pooled), grad_pooled_(grad_pooled),
      mask_(mask), ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride), p_keep_(p_keep) {
    Shape<srcdim> pshape = ShapeCheck<srcdim, SrcExp>::Check(grad_pooled);
    utils::Check(pshape == ShapeCheck<srcdim, SrcExp>::Check(data_pooled),
                 "UnPoolingExp: pooled shape mismatch");
    Shape<srcdim> sshape = ShapeCheck<srcdim, SrcExp>::Check(data_src);
    Shape<srcdim> smshape = ShapeCheck<srcdim, MaskExp>::Check(mask_);
    utils::Check(sshape == smshape, "Incorrect shape");
    for (int k = 0;  k < srcdim - 2; ++k) {
      utils::Check(pshape[k] == sshape[k],
                  "UnPoolingExp: pool and src shape mismatch");
    }
    pshape_x_ = pshape[srcdim - 1];
    pshape_y_ = pshape[srcdim - 2];
    this->shape_ = sshape;
  }
}; // struct InsanityUnPoolingExp

template<typename Reducer, typename SrcExp, typename MaskExp, typename DType, int etype>
inline InsanityUnPoolingExp<Reducer, SrcExp, MaskExp, DType, ExpInfo<SrcExp>::kDim>
insanity_unpool(const Exp<SrcExp, DType, etype> &data_src,
                const Exp<SrcExp, DType, etype> &data_pooled,
                const Exp<SrcExp, DType, etype> &grad_pooled,
                const Exp<MaskExp, DType, etype> &mask,
                index_t ksize_y, index_t ksize_x, index_t kstride, DType p_keep) {
  return InsanityUnPoolingExp<Reducer, SrcExp, MaskExp, DType, ExpInfo<SrcExp>::kDim>
    (data_src.self(), data_pooled.self(), grad_pooled.self(), mask.self(),
     ksize_y, ksize_x, kstride, p_keep);
}

template<typename Reducer, typename SrcExp, typename MaskExp, typename DType, int srcdim>
struct Plan<InsanityUnPoolingExp<Reducer, SrcExp, MaskExp, DType, srcdim>, DType> {
  public:
    explicit Plan(const InsanityUnPoolingExp<Reducer, SrcExp, MaskExp, DType, srcdim> &e)
      : data_src_(e.data_src_), data_pooled_(e.data_pooled_), grad_pooled_(e.grad_pooled_),
        mask_(e.mask_), sshape_y_(e.shape_[srcdim - 2]), sshape_x_(e.shape_[srcdim - 3]),
        pshape_y_(e.pshape_y_),  pshape_x_(e.pshape_x_),
        ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_), kstride_(e.kstride_),
        p_keep_(e.p_keep_), delta_((1.0f - p_keep_) / 4.0f) {}
    MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
      using namespace std;
      const index_t x = j;
      const index_t y = i % sshape_y_;
      const index_t c = i / sshape_y_;
      const DType flag = mask_.Eval(i, j);
      index_t loc_x = x;
      index_t loc_y = y;
      if (flag < p_keep_) {
        ;
      } else if (flag < p_keep_ + delta_) {
        loc_y = loc_y > 0 ? loc_y - 1 : loc_y;
      } else if (flag < p_keep_ + delta_ * 2.0f) {
        loc_y = loc_y + 1 < sshape_y_ ? loc_y + 1 : sshape_y_ - 1;
      } else if (flag < p_keep_ + delta_ * 3.0f) {
        loc_x = loc_x > 0 ? loc_x - 1 : loc_x;
      } else {
        loc_x = loc_x + 1 < sshape_x_ ? loc_x + 1 : sshape_x_ - 1;
      }
      const DType vsrc = data_src_.Eval(c * sshape_y_ + loc_y ,loc_x);
      const index_t py_min =
        y < ksize_y_ ? 0 : (y - ksize_y_ + kstride_) / kstride_;
      const index_t px_min =
        x < ksize_x_ ? 0 : (x - ksize_x_ + kstride_) / kstride_;
      const index_t py_max = min((y + kstride_) / kstride_, pshape_y_);
      const index_t px_max = min((x + kstride_) / kstride_, pshape_x_);
      DType val = static_cast<DType>(0);
      for (index_t py = py_min; py < py_max; ++py) {
        for (index_t px = px_min; px < px_max; ++px) {
          val += Reducer::PartialGrad(vsrc,
                                      data_pooled_.Eval(c * pshape_y_ + py, px)) *
                                      grad_pooled_.Eval(c * pshape_y_ + py, px);
        }
      }
      return val;
    }
  private:
    Plan<SrcExp, DType> data_src_, data_pooled_, grad_pooled_;
    Plan<MaskExp, DType> mask_;
    const index_t sshape_y_, sshape_x_, pshape_y_, pshape_x_;
    const index_t ksize_y_, ksize_x_;
    const index_t kstride_;
    const DType p_keep_;
    const DType delta_;
}; // struct Plan

} // namespace expr
} // namespace mshadow

namespace cxxnet {
namespace layer {

template<typename Reducer, int mode, typename xpu>
class InsanityPoolingLayer : public PoolingLayer<Reducer, mode, xpu> {
  private:
    typedef PoolingLayer<Reducer, mode, xpu> Parent;
  public:
    InsanityPoolingLayer(mshadow::Random<xpu> *p_rnd) : prnd(p_rnd) {p_keep = 1.0f;};
    virtual ~InsanityPoolingLayer() {}
    virtual void SetParam(const char *name, const char *val) {
      Parent::SetParam(name, val);
      if (!strcmp(name, "keep")) p_keep = atof(val);
    }
    virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                                const std::vector<Node<xpu>*> &nodes_out,
                                ConnectState<xpu> *p_cstate) {
      this->InitNode(nodes_in, nodes_out, p_cstate);
    }
    virtual void Forward(bool is_train,
                         const std::vector<Node<xpu>*> &nodes_in,
                         const std::vector<Node<xpu>*> &nodes_out,
                         ConnectState<xpu> *p_cstate) {
      mshadow::TensorContainer<xpu,4> &tmp = p_cstate->states[0];
      mshadow::TensorContainer<xpu,4> &mask = p_cstate->states[1];
      mshadow::Shape<2> pshape = nodes_out[0]->data[0][0].shape_;
      using namespace mshadow::expr;
      if (is_train) {
        mask = prnd->uniform(mask.shape_);
        tmp = insanity_pool<Reducer>(nodes_in[0]->data, mask,
                                     Parent::param_.kernel_height,
                                     Parent::param_.kernel_width,
                                     Parent::param_.stride,
                                     p_keep);
      } else {
        tmp = pool<Reducer>(nodes_in[0]->data, pshape,
                            Parent::param_.kernel_height,
                            Parent::param_.kernel_width,
                            Parent::param_.stride);
      }
      mshadow::Copy(nodes_out[0]->data, tmp, nodes_out[0]->data.stream_);
    }
    virtual void Backprop(bool prop_grad,
                          const std::vector<Node<xpu>*> &nodes_in,
                          const std::vector<Node<xpu>*> &nodes_out,
                          ConnectState<xpu> *p_cstate) {
      mshadow::TensorContainer<xpu,4> &tmp = p_cstate->states[0];
      mshadow::TensorContainer<xpu,4> &mask = p_cstate->states[1];
      using namespace mshadow::expr;
      nodes_in[0]->data = insanity_unpool<Reducer>(nodes_in[0]->data, tmp,
                                                   nodes_out[0]->data, mask,
                                                   Parent::param_.kernel_height,
                                                   Parent::param_.kernel_width,
                                                   Parent::param_.stride,
                                                   p_keep);
    }
  protected:
    inline void InitNode(const std::vector<Node<xpu>*> &nodes_in,
                         const std::vector<Node<xpu>*> &nodes_out,
                         ConnectState<xpu> *p_cstate) {
      Parent::InitNode(nodes_in, nodes_out, p_cstate);
      p_cstate->states.push_back(mshadow::TensorContainer<xpu,4>(false));
      p_cstate->states[1].Resize(nodes_in[0]->data.shape_);
    }
  private:
    mshadow::Random<xpu> *prnd;
    float p_keep;
}; // class InsanityPoolingLayer

} // namespace layer
} // namespace cxxnet
#endif // INSANITY_POOLING_LAYER_INL_HPP
