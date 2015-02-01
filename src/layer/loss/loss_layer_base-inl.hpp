#ifndef CXXNET_LAYER_LOSS_LAYER_BASE_INL_HPP_
#define CXXNET_LAYER_LOSS_LAYER_BASE_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"

namespace cxxnet {
namespace layer {
/*! \brief loss function layer */
template<typename xpu>
class LossLayerBase: public ILayer<xpu> {
 public:
  LossLayerBase(const LabelInfo *label_info)
      : stream_(NULL) {
    this->plabelinfo = label_info;
    this->target = "label";
    update_period = 1;
    grad_scale = 1.0f;
  }
  virtual ~LossLayerBase(void) {
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "batch_size")) batch_size = atoi(val);
    if (!strcmp(name, "update_period")) update_period = atoi(val);
    if (!strcmp(name, "target")) target = val;
    if (!strcmp(name, "grad_scale")) grad_scale = atof(val);  
  }
  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    this->stream_ = stream;
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "LossLayer: only support 1-1 connection");
    utils::Check(nodes_in[0] == nodes_out[0], "LossLayer is an self-loop Layer");
    utils::Assert(plabelinfo->name2findex != NULL,
                  "LossLayer: LabelInfo.name2findex == NULL");
    std::map<std::string, size_t>::const_iterator it =
        plabelinfo->name2findex->find(target);
    utils::Check(it != plabelinfo->name2findex->end() &&
                 it->first == target, "LossLayer: unknown target=%s",
                 target.c_str());
    target_index = it->second;
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    this->Forward_(nodes_in[0]->mat(), stream_);
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    utils::Assert(target_index < plabelinfo->fields.size(),
                  "target index exceed bound");
    this->SetGrad(nodes_in[0]->mat(),
                  plabelinfo->fields[target_index],
                  stream_);                  
    // scale gradient by dividing global batch size
    nodes_in[0]->mat() *= (grad_scale / (batch_size * update_period));
  }
  
 protected:
  // the child class can override the following functions in
  // protected fields
  /*!
   * \brief forward transformation called by loss layer
   * \param inout_data the data used as both input and output
   * \param stream the computing stream
   */
  virtual void Forward_(mshadow::Tensor<xpu, 2> inout_data,
                        mshadow::Stream<xpu> *stream) {
  }
  /*!
   * \brief set the gradient value given input data,
   *  this is the function that child class must implement in loss layer
   *
   *  when the function is called, inout_data contains the forward value
   *  This function need to set the content of the inout_data to be the
   *  gradient value given the input data and label
   * \param inout_data the data used as both input and output
   * \param label label sequence of the data
   * \param stream the computing stream
   */
  virtual void SetGrad(mshadow::Tensor<xpu, 2> inout_data,
                       const LabelRecord &label,
                       mshadow::Stream<xpu> *stream) {
    temp_.Resize(inout_data.shape_);
    mshadow::Copy(temp_, inout_data, stream);
    // wait till copy finish
    if (stream != NULL) stream->Wait();
    this->SetGradCPU(temp_, label);
    mshadow::Copy(inout_data, temp_, stream);
  }
  /*!
   * \brief same as SetGrad, but everything is now on CPU
   * normally you only need to implement this function
   *
   *  when the function is called, inout_data contains the forward value
   *  This function need to set the content of the inout_data to be the
   *  gradient value given the input data and label
   * \param inout_data the data used as both input and output
   * \param label label sequence of the data
   */
  virtual void SetGradCPU(mshadow::Tensor<cpu, 2> inout_data,
                          const LabelRecord &label) {
    utils::Error("LossLayerBase::SetGradCPU not implemented");
  }
 private:
  /*! \brief stream used for internal computation */
  mshadow::Stream<xpu> *stream_;
  /*! \brief temp memory to do CPU side computation*/
  mshadow::TensorContainer<cpu, 2> temp_;
  /*!
   * \brief global batch_size set by user, this 
   *        is not necessarily the batch_size in plabelinfo,
   *        since a batch can be divided 
   *        into subbatch to layers in different devices
   */
  int batch_size;
  /*! \brief target field of loss */
  std::string target;
  /*! \brief remembered target index in label info */
  size_t target_index;
  /*! \brief reference to label information */
  const LabelInfo *plabelinfo;
  // update period, used to do scaling
  int update_period;
  /*! \brief gradient scaling of the loss */
  float grad_scale;
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_LOSS_LAYER_BASE_INL_HPP_
