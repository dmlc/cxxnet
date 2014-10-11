#ifndef CXXNET_SYNC_SYNC_H_
#define CXXNET_SYNC_SYNC_H_
/*!
 * \file sync.h
 * \brief simple synchronization modules defined for multi-threading case
 * \author Tianqi
 */
#include <vector>
#include <mshadow/tensor.h>

namespace cxxnet {
namespace sync {
/*! \brief simple weight synchronizer */
template<typename xpu>
class ISynchronizer {
 public:
  virtual ~ISynchronizer(void) {}
  /*!
   * \brief Set param for the layer from string
   * \param name parameter name
   * \param val string for configuration
   */  
  virtual void SetParam(const char *name, const char *val) {}
  /*! \brief initialize the synchronizer */
  virtual void Init(void) {}
  /*! \brief synchronization actions to be performs before the updater */
  virtual void SyncBeforeUpdate(void) {}
  /*! \brief synchronization actions to be performs before the updater */
  virtual void SyncAfterUpdate(void) {}
};

/*!
 * \brief create a synchronizer of specific type 
 * \param type type of synchronizer
 * \param weights list of weights, all the matrix is flattened to 2D with same content reference as original weightx
 * \param grads list of gradients
 * \param tag the tag of the synchronizer
 */
template<typename xpu>
ISynchronizer<xpu>* CreateSynch(const char *type,
                                const std::vector< mshadow::Tensor<xpu,2> > &weights,
                                const std::vector< mshadow::Tensor<xpu,2> > &grads,
                                const char *tag);
}  // namespace sync
}  // namespace cxxnet
#include "./sync_impl-inl.hpp"
#endif
