#ifndef CXXNET_UPDATER_UPDATER_H_
#define CXXNET_UPDATER_UPDATER_H_

#include <vector>
#include <mshadow/tensor.h>
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
class IUpdater{  
 public:
  /*! \brief reuse layer's visitor type, can be used to access weight in updater */
  typedef typename layer::ILayer<xpu>::IVisitor IVisitor;
  /*!\brief virtual destructor */
  virtual ~IUpdater(void) {}
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
  /*!\ brief set parameters that could be spefic to this updater */
  virtual void SetParam(const char *name, const char *val) = 0;
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
}  // namespace updater
}  // namespace cxxnet
#endif  // UPDATER_UPDATER_H_
