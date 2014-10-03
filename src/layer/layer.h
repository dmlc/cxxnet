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
  /*! \brief matrix view of the node */
  inline mshadow::Tensor<xpu,2> mat(void) {
    mshadow::Shape<4> shape = data.shape;
    return mshadow::Tensor<xpu,2>(data.dptr, mshadow::Shape2(shape[3], shape[0]));
  }
  /*! \brief check whether it holds a matrix data */
  inline bool is_mat( void ) const {
    return data.shape[2] == 1 && data.shape[1] == 1;
  }  
}; // struct Node

/*! 
 * \brief Interface of layer
 *    this is a pure interface and there is not data memember 
 *    in ILayer. However, there can be common pattern of memembers in a layer,
 *    see the following notes
 *
 *  Input and output node: 
 *     Each layer can be associated with set of input or output nodes, 
 *     the layer implementation must hold reference to the input output nodes,
 *     and they are not part of this interface. 
 *     For a common one to one connection layer, see CommonLayerBase
 *      
 *  Connection weights and gradient:
 *     Some layers can have connection weight parameters, and gradient of weights.
 *     These weights are hold in the specific implementation. 
 *     They can be accesed by using a vistor, see also IVisitor
 *     SaveModel and LoadModel are used to serialize and deserialize them
 * 
 *  Parameters:
 *     Parameters related to each layer (e.g. number of hidden nodes), can be set by calling SetParam
 *
 * \tparam xpu the device name it lies in, can be cpu/gpu
 * \sa CommonLayerBase, ILayer::IVisitor
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
   * \brief (1) intialize the temp parameters
   *        (2) adjust output nodes' shape based on input nodes' shape
   *  this function will be called before using a layer
   */
  virtual void InitLayer(void) {}
  /*!
   * \brief Forward propagation from input nodes to output nodes
   * \param is_train the propagation is during training phase
   */
  virtual void Forward(bool is_train) = 0;
  /*!
   * \brief Back propagation from output nodes to input nodes
   *    in the beginning of function call, the output nodes is ensured to contain the gradient value
   * \param prop_grad if true, then the layer will propagate gradient back to its input node
   */
  virtual void Backprop(bool prop_grad) = 0;
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

/*!
 * \brief this is a common layer base that most implementation of layer inheritates
 *    specifically, this implements a common framework one node to one node connection.
 *   
 *    The difference between ILayer, is that Forward, Backprop, InitLayer takes
 *    explicit argument of input/output nodes. This makes the logical explicit and clear,
 *    though less flexible than ILayer
 * \tparam xpu the device name it lies in, can be cpu/gpu
 */
template<typename xpu>
class CommonLayerBase : public ILayer<xpu> {
 public:
  CommonLayerBase(mshadow::Random<xpu> *p_rnd, Node<xpu> *p_in, Node<xpu> *p_out)
      : prnd_(p_rnd), pin_(p_in), pout_(p_out) {
  }
  virtual ~CommonLayerBase(void){}
  virtual void InitLayer(void) {
    this->InitLayer_(*pin_, pout_);
  }
  virtual void Forward(bool is_train) {
    this->Forward_(is_train, pin_, pout_);
  }
  virtual void Backprop(bool prop_grad) {
    this->Backprop_(prop_grad, pin_, pout_);
  }

 protected:
  /*!
   * \brief initialized 
   * \param node_in input node, with the shape already being setted correctly
   * \param pnode_out output node, whose shape should be set by InitLayer function,
   *                  based on the input node shape and the layer configuration
   */  
  virtual void InitLayer_(const Node<xpu> &node_in,
                          Node<xpu> *pnode_out) = 0;
  /*!
   * \brief Forward propagation from input nodes to output nodes
   * \param is_train the propagation is during training phase
   * \param pnode_in pointer to input node, the content is set to be the activation 
   *                 of input node before call of Forward
   * \param pnode_out pointer to output node, the content should be set to
   *                  be activation value of output node
   */
  virtual void Forward_(bool is_train,
                        Node<xpu> *pnode_in,
                        Node<xpu> *pnode_out) = 0;
  /*!
   * \brief Backward propagation from input nodes to output nodes,
   *        and accumulate gradient to the internal gradient variable
   * \param prop_grad if true, then the layer will propagate gradient back to its input node
   * \param pnode_in pointer to input node, the content is set to be the activation 
   *                 of input node before call of Forward
   * \param pnode_out pointer to output node, the content should be set to
   *                  be activation value of output node
   */
  virtual void Backprop_(bool prop_grad,
                         Node<xpu> *pnode_in,
                         Node<xpu> *pnode_out) = 0;
  /*! \brief random number generator, that can be used in child class */
  mshadow::Random<xpu> *prnd_;
  /*! \brief input and output node type */
  Node<xpu> *pin_, *pout_;
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
 * \param nodes_in list of input nodes of the layer
 * \param nodes_out list of output nodes of the layer
 */
template<typename xpu>
ILayer<xpu>* CreateLayer(LayerType type,
                         mshadow::Random<xpu> *p_rnd,
                         const std::vector<Node<xpu>*> &nodes_in,
                         const std::vector<Node<xpu>*> &nodes_out);
}  // namespace layer
}  // namespace cxxnet
#endif  // CXXNET_LAYER_LAYER_H
