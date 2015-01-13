#ifndef CXXNET_LAYER_VISITOR_H_
#define CXXNET_LAYER_VISITOR_H_
#include <vector>
#include <string>
#include "./layer.h"
/*!
 * \file visitor.h
 * \brief implementation of visitor of layer weights
 * tihs file gives util to set/get weights from/to a layer
 * \author Tianqi Chen
 */
namespace cxxnet {
namespace layer {
/*!
 * \brief visitor used to get weight/gradient
 *  from the layer the weight is flattened to 2D shape
 *
 *  Usage Example:
 *     GetWeightVisitor vs("weight");
 *     layer->ApplyVisitor(&vs);
 *     std::vector<mshadow::Tensor<xpu, 2> > weights = vs.data;
 *
 * \tparam xpu the device the data contents lies on
 */
template<typename xpu>
class GetWeightVisitor : public ILayer<xpu>::IVisitor {
 public:
  /*! \brief the weight contents of the layer */
  std::vector<mshadow::Tensor<xpu, 2> > data;
  /*! \brief field name of each of the data */
  std::vector<std::string> fields;
  /*!
   * \brief constructor of visitor,
   * \param data_type can only be "grad" or "weight"
   * \param prefix set the prefix to only fetch data whose field name
   *  have the prefix
   */
  GetWeightVisitor(const char *data_type, const char *prefix = "")
      : mode_(0), prefix_(prefix) {
    if (!strcmp(data_type, "weight")) mode_ = 0;
    if (!strcmp(data_type, "grad")) mode_ = 1;
    utils::Assert(mode_ == 0 || mode_ == 1,
      "GetWeightVisitor: do not support data_type %s", data_type);
  }
  // visit
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu, 1> weight,
                     mshadow::Tensor<xpu, 1> grad) {
    this->Visit_(field_name, weight, grad);
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu, 2> weight,
                     mshadow::Tensor<xpu, 2> grad) {
    this->Visit_(field_name, weight, grad);
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu, 3> weight,
                     mshadow::Tensor<xpu, 3> grad) {
    this->Visit_(field_name, weight, grad);
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu, 4> weight,
                     mshadow::Tensor<xpu, 4> grad) {
    this->Visit_(field_name, weight, grad);
  }

 private:
  // internal mode
  int mode_;
  // prefix to match the field name
  std::string prefix_;
  // local visiting function
  template<int dim>
  inline void Visit_(const char *field_name,
                     mshadow::Tensor<xpu, dim> weight,
                     mshadow::Tensor<xpu, dim> grad) {
    if (strncmp(prefix_.c_str(), field_name, prefix_.length()) != 0) return;
    fields.push_back(std::string(field_name));
    if (mode_ == 0) {
      data.push_back(weight.FlatTo2D());
    } else {
      data.push_back(grad.FlatTo2D());
    }
  }
};
/*!
 * \brief set used to set weight/gradient into layer
 *  the weight must be flattened to 2D to input
 *
 *  Usage Example:
 *     GetSetVisitor vs(data, "weight");
 *     layer->ApplyVisitor(&vs);
 *
 * \tparam xpu the device the data contents lies on
 */
template<typename xpu>
class SetWeightVisitor : public ILayer<xpu>::IVisitor {
 public:
  /*!
   * \brief constructor of visitor,
   * \param data_type can only be "grad" or "weight"
   * \param prefix set the prefix to only fetch data whose field name
   *  have the prefix
   */
  SetWeightVisitor(const std::vector<mshadow::Tensor<xpu, 2> > &data,
                   const char *data_type, const char *prefix = "")
      : data_(data), prefix_(prefix), counter_(0) {
    if (!strcmp(data_type, "weight")) mode_ = 0;
    if (!strcmp(data_type, "grad")) mode_ = 1;
    utils::Assert(mode_ == 0 || mode_ == 1,
      "SetWeightVisitor: do not support data_type %s", data_type);
  }
  // visit
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu, 1> weight,
                     mshadow::Tensor<xpu, 1> grad) {
    this->Visit_(field_name, weight, grad);
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu, 2> weight,
                     mshadow::Tensor<xpu, 2> grad) {
    this->Visit_(field_name, weight, grad);
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu, 3> weight,
                     mshadow::Tensor<xpu, 3> grad) {
    this->Visit_(field_name, weight, grad);
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu, 4> weight,
                     mshadow::Tensor<xpu, 4> grad) {
    this->Visit_(field_name, weight, grad);
  }

 private:
  /*! \brief the weight contents of the layer */
  std::vector<mshadow::Tensor<xpu, 2> > data_;
  // internal mode
  int mode_;
  // prefix to match the field name
  std::string prefix_;
  // index counter
  size_t counter_;
  template<int dim>
  inline void Visit_(const char *field_name,
                     mshadow::Tensor<xpu, dim> weight,
                     mshadow::Tensor<xpu, dim> grad) {
    using mshadow::expr::reshape;
    if (strncmp(prefix_.c_str(), field_name, prefix_.length()) != 0) return;
    utils::Check(counter_ < data_.size(),
                 "SetWeightVisitor: not enough input data");
    if (mode_ == 0) {
      weight = reshape(data_[counter_], weight.shape_);
    } else {
      grad = reshape(data_[counter_], grad.shape_);
    }
    counter_ += 1;
  }
};
}  // namespace layer
}  // namespace cxxnet
#endif  // CXXNET_LAYER_VISITOR_H_
