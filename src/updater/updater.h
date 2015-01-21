#ifndef CXXNET_UPDATER_UPDATER_H_
#define CXXNET_UPDATER_UPDATER_H_

#include <vector>
#include <mshadow/tensor.h>
#include <mshadow-ps/ps.h>
#include "../global.h"
#include "../layer/layer.h"

namespace cxxnet {
/*! \brief namespace of updating algorithms */
namespace updater {
/*!
 * \brief interface of parameter updater,
 *        it defines the updating behavior of parameters
 *        ILayer takes no charge of parameter update,
 *        IUpdater takes the gradient value accumulated by ILayer and the weight
 *        to perform update on the weight
 * \tparam xpu which device the data of the updater lies 
 */
template<typename xpu>
class IUpdater { 
 public:
  /*! \brief reuse layer's visitor type, can be used to access weight in updater */
  typedef typename layer::ILayer<xpu>::IVisitor IVisitor;
  /*!\brief virtual destructor */
  virtual ~IUpdater(void) {}
  /*!
   * \brief set the stream of internal computation to be stream
   * \param stream the stream to be used
   */
  virtual void SetStream(mshadow::Stream<xpu> *stream) = 0;
  /*! \brief intialize, print information about updater if not silent */
  virtual void Init(void) = 0;
  /*! 
   * \brief apply visitor to the updater,
   *   this is used to visit tha content of the updater
   */
  virtual void ApplyVisitor(IVisitor *pvisitor) = 0;
  /*!
   * \brief inform the updater that we are starting
   *        new round of iteration over data
   * \param round round counter
   */
  virtual void StartRound(int round) = 0;
  /*!
   * \brief update parameter
   * \param epoch what current epoch is.
   *        epoch is number of mini-batches passed, 
   *        while round is one pass over training data
   */
  virtual void Update(long epoch) = 0;
  /*!
   * \brief update the parameter, provides the
   *        gradient value from outside 
   * \param epoch what current epoch is
   *        epoch is number of mini-batches passed, 
   *        while round is one pass over training data
   * \param grad the pointer to pass in gradient value
   *        to minimize the interface, FlatTo2D should
   *        be called before passing in the gradient value
   */
  virtual void Update(long epoch, mshadow::Tensor<xpu, 2> grad) = 0;
  /*!\ brief set parameters that could be spefic to this updater */
  virtual void SetParam(const char *name, const char *val) = 0;
};

/*!
 * \brief asynchronize updater,
 * BeforeBackprop and AfterBackprop are asynchronize functions calls
 * and user need to call UpdateWait to wait the update to finish
 */
template<typename xpu>
class IAsyncUpdater : public IUpdater<xpu> {
 public:
  /*!
   * \brief this function is called before calling backprop
   * used by updater in case updater want to recover gradient by itself,
   * instead of calculated by the ILayer
   */
  virtual void BeforeBackprop(const std::vector<layer::Node<xpu>*> &nodes_in,
                              const std::vector<layer::Node<xpu>*> &nodes_out) = 0;
  /*!
   * \brief this function is called after calling backprop
   * \param do_update whether an update is performed in this iteration
   * \param epoch the update epoch if doing update
   */  
  virtual void AfterBackprop(bool do_update, long epoch) = 0;
  /*!
   * \brief this function will be called before
   * the forwardprop of all layers being calls
   */
  virtual void BeforeForward(void) = 0;
  /*!
   * \brief block until update is finished
   * if there were no update or update was already finished
   * this function will directly return
   */
  virtual void UpdateWait(void) = 0;
  // disable update function
  virtual void Update(long epoch) {
    utils::Error("IAsyncUpdater.Update call AfterBackprop instead");
  }
  // disable update function
  virtual void Update(long epoch, mshadow::Tensor<xpu, 2> grad) {
    utils::Error("IAsyncUpdater.Update call AfterBackprop instead");
  }
};
/*!
 * \brief factory: create updaters for a given layer, push_back them to out_updaters
 * \param type indicate the type of updater
 * \param p_rnd pointer to random number generator
 * \param p_layer pointer to the layer object, where the data is going to be pulled from
 * \param out_updaters vector to hold outputs, if there is already elements in out_updaters, 
 *                     the function is going to push new updaters to the back of the vector
 */
template<typename xpu>
void CreateUpdaters(const char *type,
                    mshadow::Random<xpu> *p_rnd,
                    layer::ILayer<xpu> *p_layer,
                    std::vector<IUpdater<xpu>*> *out_updaters);

/*!
 * \brief factory: create updaters for a given layer, push_back them to out_updaters
 * \param data_key_base used to index the updaters, each new updater will
 *                      take data_key_base as their id, and increase it by 1
 * \param device_id the device id where the async updater lies
 * \param param_server parameter server that could be used by async updater
 * \param type indicate the type of updater
 * \param p_rnd pointer to random number generator
 * \param layer_type the type of the layer
 * \param p_layer pointer to the layer object, where the data is going to be pulled from
 * \param out_updaters vector to hold outputs, if there is already elements in out_updaters, 
 *                     the function is going to push new updaters to the back of the vector
 */
template<typename xpu>
void CreateAsyncUpdaters(int data_key_base,
                         int device_id,
                         mshadow::ps::IParamServer<xpu, real_t> *param_server,
                         const char *type,
                         mshadow::Random<xpu> *p_rnd,
                         layer::LayerType layer_type,
                         layer::ILayer<xpu> *p_layer,
                         std::vector<IAsyncUpdater<xpu>*> *out_updaters);
}  // namespace updater
}  // namespace cxxnet
#endif  // UPDATER_UPDATER_H_
