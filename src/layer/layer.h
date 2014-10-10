#ifndef CXXNET_LAYER_LAYER_H
#define CXXNET_LAYER_LAYER_H
/*!
 * \file layer.h
 * \brief abstract interface of layer
 * \author Tianqi Chen, Bing Xu
 */
#include <vector>
#include <mshadow/tensor.h>
#include "../global.h"
#include "../utils/utils.h"
#include "../utils/io.h"

/*! \brief namespace of cxxnet */
namespace cxxnet {
/*! \brief namespace of layer defintiion */
namespace layer {
/*! 
 * \brief node structure, this is used to store forward activation,
 *    and backproped gradient in the network
 * \tparam xpu the device name it lies in, can be cpu/gpu
 */
template<typename xpu>
struct Node {
  /*! 
   * \brief content of the node 
   *  layout: 
   *     images (batch_size, nchannel, height, width)
   *     matrix (batch_size, 1, 1, length-of-vector)
   */
  mshadow::Tensor<xpu,4> data;
  Node(void) {
    data.shape = mshadow::Shape4(0,0,0,0);
  }
  /*! \brief matrix view of the node */
  inline mshadow::Tensor<xpu,2> mat(void) {
    return data.FlatTo2D();
  }
  /*! \brief check whether it holds a matrix data */
  inline bool is_mat(void) const {
    return data.shape[2] == 1 && data.shape[1] == 1;
  }
  inline void FreeSpace(void) {
    mshadow::FreeSpace(data);
  }  
}; // struct Node

/*! 
 * \brief data structure to hold additional information about label of instances
 * this information is used by layers that computes the gradient over objectibe functions,
 * this data structure  will be evolving, to meet needs of different kinds of supervision signals
 */
struct LabelInfo {
  /*! \brief pointer to the label fields */
  const float *labels;
  /*! \brief the size of the batch */
  mshadow::index_t batch_size;
  // constructor
  LabelInfo(void) : labels(NULL), batch_size(0) {
  }
  /*! 
   * \brief slice the label information to take [begin, end)
   * \param begin beginning of index
   * \param end end of index
   */
  inline LabelInfo Slice(mshadow::index_t begin, mshadow::index_t end) const {
    LabelInfo ret;
    ret.labels = labels + begin;
    ret.batch_size = end - begin;
    return ret;
  }
};

/*! 
 * \brief connection states 
 *   temporal state space that can be used to share information between forward and backprop
 *   not every layer needs this, this is used 
 */
template<typename xpu>
struct ConnectState {
  /*! \brief the contents of states */
  std::vector< mshadow::TensorContainer<xpu, 4> > states;
};

/*! 
 * \brief Interface of layer
 *    this is a pure interface and there is not data memember 
 *    in ILayer. However, there can be common pattern of memembers in a layer,
 *    see the following notes
 *
 *  Connection and Layer:
 *     In the current design of cxxnet, there is concept of Connection, and Layer
 *     A Layer is defines set of of functions Forward and Backprop, given input/output nodes
 *     A Layer is not attached to specific pair of nodes, while Connection is.
 *     Connection is the connection between nodes, whose function is backed by Layers. 
 *     Different connection can share a same Layer
 *
 *     This means Layer can not contain any node specific state(for example, dropout mask) in the class. 
 *     The Connection specific states are maintained by Connection, and passed to Layer during Forward/Backprop
 *      
 *  Weights and gradient:
 *     Some layers can have connection weight parameters, and gradient of weights.
 *     These weights are hold in the specific implementation. 
 *     They can be accesed by using a vistor, see also IVisitor
 *     SaveModel and LoadModel are used to serialize and deserialize them
 * 
 *  Parameters:
 *     Parameters related to each layer (e.g. number of hidden nodes), can be set by calling SetParam
 *
 * \tparam xpu the device name it lies in, can be cpu/gpu
 * \sa CommonLayerBase, ILayer::IVisitor, ConnectState
 */
template<typename xpu>
class ILayer {
 public:
  /*!
   * \brief visitor to the layer
   *    visits the weight and gradient in the layer
   */
  struct IVisitor {
    /*!
     * \brief visit content of the layer, this is called by Layer
     *    when ApplyVisitor is called
     *
     *    Visitor can use to get weight content, copy/set weights, etc. 
     *   
     * \param field_name the name of field on the layer
     * \param weight the connect weight used in the layer
     * \param grad the gradient of the weight,
     *        it is ensured to be have same shape as weight
     */
    virtual void Visit(const char *field_name,
                       mshadow::Tensor<xpu,1> weight,
                       mshadow::Tensor<xpu,1> grad) = 0;
    virtual void Visit(const char *field_name,
                       mshadow::Tensor<xpu,2> weight,
                       mshadow::Tensor<xpu,2> grad) = 0;
    virtual void Visit(const char *field_name,
                       mshadow::Tensor<xpu,3> weight,
                       mshadow::Tensor<xpu,3> grad) = 0;
  };
 public:
  /*! \brief virtual destructor */
  virtual ~ILayer(void) {}
  /*!
   * \brief initialize the connection, this function takes charge of two shings
   *   (1) setup the shape of output nodes in nodes_out, given the 
   *   (2) allocate necessary temporal state space in p_cstate
   * \param nodes_in vector of input nodes
   * \param nodes_out vector of output nodes
   * \param p_cstate
   */
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) = 0;
  /*!
   * \brief update the p_cstate when batch size(shape[3] of input output nodes) changed
   *        This function is called whenever the batch_size changed, and the Layer can make use
   *        of this to update the p_cstate
   * \param nodes_in vector of input nodes
   * \param nodes_out vector of output nodes
   * \param p_cstate temporal state space that can be used to share information between forward and backprop
   */
  virtual void OnBatchSizeChanged(const std::vector<Node<xpu>*> &nodes_in,
                                  const std::vector<Node<xpu>*> &nodes_out,
                                  ConnectState<xpu> *p_cstate) {}
  /*!
   * \brief Forward propagation from input nodes to output nodes
   * \param is_train the propagation is during training phase
   * \param nodes_in vector of input nodes
   * \param nodes_out vector of output nodes
   * \param p_cstate temporal state space that can be used to share information between forward and backprop
   */
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) = 0;
  /*!
   * \brief Back propagation from output nodes to input nodes
   *    in the beginning of function call, the output nodes is ensured to contain the gradient value
   * \param prop_grad if true, then the layer will propagate gradient back to its input node
   * \param nodes_in vector of input nodes
   * \param nodes_out vector of output nodes
   * \param p_cstate temporal state space that can be used to share information between forward and backprop
   */
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) = 0;
  
 public:
  /*! 
   * \brief apply visitor to the layer,
   *   this is used to visit tha content of the layer
   */
  virtual void ApplyVisitor(IVisitor *pvisitor) {}
  /*!
   * \brief Set param for the layer from string
   * \param name parameter name
   * \param val string for configuration
   */
  virtual void SetParam(const char *name, const char* val) {}
  /*!
   * \brief intialized model parameters, only called when model parameters are not initialzied
   */
  virtual void InitModel(void) {}
  /*!
   * \brief Save model into binary file
   * \param fo output stream
   */
  virtual void SaveModel(utils::IStream &fo) const {}
  /*!
   * \brief Load model from binary file
   * \param fi input stream
   */
  virtual void LoadModel(utils::IStream &fi) {}
};

/*! \brief these are enumeration */
enum LayerType { 
  kFullConnect = 1,
  kSoftmax = 2,
  kRectifiedLinear = 3,
  kSigmoid = 4,
  kTanh = 5,
  kSoftplus = 6,
  kFlatten = 7,
  kDropout = 8,
  kDropConn = 9,
  kConv = 10,
  kMaxPooling = 11,
  kSumPooling = 12,
  kAvgPooling = 13,
  kLRN = 15,
  kBias = 17
};
/*!
 * \brief get the layer type from string
 * \param type indicate the type of a layer
 */
inline LayerType GetLayerType(const char *type) {
  if (!strcmp(type, "fullc")) return kFullConnect;
  if (!strcmp(type, "bias")) return kBias;
  if (!strcmp(type, "softmax")) return kSoftmax;
  if (!strcmp(type, "relu")) return kRectifiedLinear;
  if (!strcmp(type, "sigmoid")) return kSigmoid;
  if (!strcmp(type, "tanh")) return kTanh;
  if (!strcmp(type, "softplus")) return kSoftplus;
  if (!strcmp(type, "flatten")) return kFlatten;
  if (!strcmp(type, "dropout")) return kDropout;
  if (!strcmp(type, "dropconn")) return kDropConn;
  if (!strcmp(type, "conv")) return kConv;
  if (!strcmp(type, "max_pooling")) return kMaxPooling;
  if (!strcmp(type, "sum_pooling")) return kSumPooling;
  if (!strcmp(type, "avg_pooling")) return kAvgPooling;
  if (!strcmp(type, "lrn")) return kLRN;
  utils::Error("unknown layer type: %s", type);
  return kConv;
}
/*!
 * \brief factory: create an upadater algorithm of given type
 * \param type indicate the type of a layer
 * \param p_rnd random number generator
 * \param label_info pointer to the label information field, that will be contain,
 *                   this is similar to node, but contains label information that can be used
 *                   to compute gradient over objetives
 */
template<typename xpu>
ILayer<xpu>* CreateLayer(LayerType type,
                         mshadow::Random<xpu> *p_rnd,
                         const LabelInfo *label_info);
}  // namespace layer
}  // namespace cxxnet
#endif  // CXXNET_LAYER_LAYER_H
