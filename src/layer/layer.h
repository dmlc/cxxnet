#ifndef CXXNET_LAYER_LAYER_H_
#define CXXNET_LAYER_LAYER_H_
/*!
 * \file layer.h
 * \brief abstract interface of layer
 * \author Tianqi Chen, Bing Xu
 */
#include <vector>
#include <map>
#include <string>
#include <mshadow/tensor.h>
#include "../global.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#if CXXNET_USE_CUDNN == 1
 #ifdef __CUDACC__
  #include <cudnn.h>
 #endif
#endif

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
  mshadow::Tensor<xpu, 4> data;
  /*! \brief whether the underlying data must be contiguous */
  bool must_contiguous;
  bool inited;
  // constructor
  Node(void) : must_contiguous(false) {
    data.shape_ = mshadow::Shape4(0,0,0,0);
    inited = false;
  }
  /*! \brief matrix view of the node */
  inline mshadow::Tensor<xpu, 2> mat(void) {
    return data.FlatTo2D();
  }
  /*! \brief check whether it holds a matrix data */
  inline bool is_mat(void) const {
    return data.size(1) == 1 && data.size(2) == 1;
  }
  /*! \brief helper rountine to free space */
  inline void FreeSpace(void) {
    if (inited){
      mshadow::FreeSpace(&data);
    }
  }
  /*! \brief helper rountine to allocate space */
  inline void AllocSpace(void) {
    if (must_contiguous) {
      mshadow::AllocSpace(&data, false);
      utils::Assert(data.CheckContiguous(), "contiguous");
    } else {
      mshadow::AllocSpace(&data);
    }
    inited = true;
  }
}; // struct Node

/*!
 * \brief a single label record that can be taken by a loss function
 *    use struct for future extensibility
 */
struct LabelRecord {
  /*! \brief label field */
  mshadow::Tensor<cpu, 2> label;
  /*!
   * \brief slice the label information to take [begin, end)
   * \param begin beginning of index
   * \param end end of index
   */
  inline LabelRecord Slice(index_t begin, index_t end) const {
    LabelRecord r;
    r.label = label.Slice(begin, end);
    return r;
  }
};
/*!
 * \brief data structure to hold additional information about label of instances
 * this information is used by layers that computes the gradient over objectibe functions,
 * this data structure  will be evolving, to meet needs of different kinds of supervision signals
 */
struct LabelInfo {
  /*! \brief fields of each label */
  std::vector<LabelRecord> fields;
  /*!
   * \brief name map that maps field name
   *  to the index of fields
   */
  const std::map<std::string, size_t> *name2findex;
  // constructor
  LabelInfo(void) : name2findex(NULL) {
  }
  /*!
   * \brief slice the label information to take [begin, end)
   * \param begin beginning of index
   * \param end end of index
   */
  inline LabelInfo Slice(index_t begin, index_t end) const {
    LabelInfo ret;
    ret.fields.resize(fields.size());
    for (size_t i = 0; i < fields.size(); ++i) {
      ret.fields[i] = fields[i].Slice(begin, end);
    }
    ret.name2findex = name2findex;
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
                       mshadow::Tensor<xpu, 1> weight,
                       mshadow::Tensor<xpu, 1> grad) = 0;
    virtual void Visit(const char *field_name,
                       mshadow::Tensor<xpu, 2> weight,
                       mshadow::Tensor<xpu, 2> grad) = 0;
    virtual void Visit(const char *field_name,
                       mshadow::Tensor<xpu, 3> weight,
                       mshadow::Tensor<xpu, 3> grad) = 0;
    virtual void Visit(const char *field_name,
                       mshadow::Tensor<xpu, 4> weight,
                       mshadow::Tensor<xpu, 4> grad) = 0;
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
  /*!
   * \brief return whether this layer can be shared across multiple
   * connections, for most layers this should be true
   */
  virtual bool AllowSharing(void) const {
    return true;
  }
  /*!
   * \brief set the stream of internal computation to be stream
   * \param stream the stream to be used
   */
  virtual void SetStream(mshadow::Stream<xpu> *stream) {}
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
// shared layer is a special type indicating that this connection
// is sharing Layer with an existing connection
const int kSharedLayer = 0;
const int kFullConnect = 1;
const int kSoftmax = 2;
const int kRectifiedLinear = 3;
const int kSigmoid = 4;
const int kTanh = 5;
const int kSoftplus = 6;
const int kFlatten = 7;
const int kDropout = 8;
const int kConv = 10;
const int kMaxPooling = 11;
const int kSumPooling = 12;
const int kAvgPooling = 13;
const int kLRN = 15;
const int kBias = 17;
const int kConcat = 18;
const int kXelu = 19;
const int kCaffe = 20;
// first apply relu then maxpooling
const int kReluMaxPooling = 21;
const int kMaxout = 22;
const int kSplit = 23;
const int kInsanity = 24;
const int kInsanityPooling = 25;
const int kL2Loss = 26;
const int kMultiLogistic = 27;
const int kChConcat = 28;
const int kPRelu = 29;
/*! \brief gap used to encode pairtest layer */
const int kPairTestGap = 1024;
/*! \brief use integer to encode layer types */
typedef int LayerType;
/*!
 * \brief get the layer type from string
 * \param type indicate the type of a layer
 */
inline LayerType GetLayerType(const char *type) {
  if (!strncmp(type, "share", 5)) return kSharedLayer;
  if (!strcmp(type, "fullc")) return kFullConnect;
  if (!strcmp(type, "bias")) return kBias;
  if (!strcmp(type, "softmax")) return kSoftmax;
  if (!strcmp(type, "relu")) return kRectifiedLinear;
  if (!strcmp(type, "sigmoid")) return kSigmoid;
  if (!strcmp(type, "tanh")) return kTanh;
  if (!strcmp(type, "softplus")) return kSoftplus;
  if (!strcmp(type, "flatten")) return kFlatten;
  if (!strcmp(type, "dropout")) return kDropout;
  if (!strcmp(type, "conv")) return kConv;
  if (!strcmp(type, "relu_max_pooling")) return kReluMaxPooling;
  if (!strcmp(type, "max_pooling")) return kMaxPooling;
  if (!strcmp(type, "sum_pooling")) return kSumPooling;
  if (!strcmp(type, "avg_pooling")) return kAvgPooling;
  if (!strcmp(type, "lrn")) return kLRN;
  if (!strcmp(type, "concat")) return kConcat;
  if (!strcmp(type, "xelu")) return kXelu;
  if (!strcmp(type, "maxout")) return kMaxout;
  if (!strcmp(type, "split")) return kSplit;
  if (!strcmp(type, "insanity")) return kInsanity;
  if (!strcmp(type, "insanity_max_pooling")) return kInsanityPooling;
  if (!strcmp(type, "l2_loss")) return kL2Loss;
  if (!strcmp(type, "multi_logistic")) return kMultiLogistic;
  if (!strcmp(type, "ch_concat")) return kChConcat;
  if (!strcmp(type, "prelu")) return kPRelu;
  #if CXXNET_USE_CAFFE_ADAPTOR
  if (!strcmp(type, "caffe")) return kCaffe;
  #endif
  if (!strncmp(type, "pairtest-", 9)) {
    char tmaster[256], tslave[256];
    sscanf(type + 9, "%[^-]-%[^:]", tmaster, tslave);
    return kPairTestGap * GetLayerType(tmaster) + GetLayerType(tslave);
  }
  utils::Error("unknown layer type: \"%s\"", type);
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
/*!
 * \brief this data structure specifies a connection
 * this is a node specific data structure, that defines connection between nodes
 * \tparam xpu the device the connection lies in
 */
template<typename xpu>
struct Connection {
  /*! \brief the backend layer of the connection */
  ILayer<xpu> *layer;
  /*! \brief the type of the backend layer */
  LayerType type;
  /*! \brief shared states of the connection */
  ConnectState<xpu> state;
  /*! \brief list of input nodes */
  std::vector<Node<xpu>*> nodes_in;
  /*! \brief list of output nodes */
  std::vector<Node<xpu>*> nodes_out;
  /*!
   * \brief set the internal computation stream
   * \param stream the stream that was used for computation
   */
  inline void SetStream(mshadow::Stream<xpu> *stream) {
    layer->SetStream(stream);
    for (size_t i = 0; i < state.states.size(); ++i) {
      state.states[i].set_stream(stream);
    }
    for (size_t i = 0; i < nodes_in.size(); ++i) {
      nodes_in[i]->data.set_stream(stream);
    }
    for (size_t i = 0; i < nodes_out.size(); ++i) {
      nodes_out[i]->data.set_stream(stream);
    }
  }
};
}  // namespace layer
}  // namespace cxxnet
#endif  // CXXNET_LAYER_LAYER_H
