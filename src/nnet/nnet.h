#ifndef CXXNET_NNET_NNET_H_
#define CXXNET_NNET_NNET_H_
/*!
 * \file nnet.h
 * \brief trainer abstraction
 * \author Bing Xu, Tianqi Chen
 */

#include <vector>
#include <mshadow/tensor.h>
#include "../global.h"
#include "../utils/io.h"
#include "../io/data.h"

namespace cxxnet {
namespace nnet {
/*! \brief interface for network */
class INetTrainer{
 public:
  virtual ~INetTrainer(void) {}
  /*!
   * \brief Set parametters
   * \param name parameter name
   * \param val string for configuration
   */
  virtual void SetParam(const char *name, const char *val) = 0;
  /*! \brief random initalize model */
  virtual void InitModel(void) = 0;
  /*! \brief save model to stream */
  virtual void SaveModel(utils::IStream &fo) = 0;
  /*! \brief load model from stream */
  virtual void LoadModel(utils::IStream &fi) = 0;
  /*!
   * \brief inform the updater that a new round has been started
   * \param round round counter
   */
  virtual void StartRound(int round) = 0;
  /*!
   * \brief update model parameter
   * \param training data batch
   */
  virtual void Update(const DataBatch& data) = 0;
  /*! 
   * \brief evaluate a test statistics, output results as a string
   * \param iter_eval the iterator containing the evaluation data
   * \param data_name the name of the dataset, used to construct the returing string
   * \return a string containing the evaluation result in format data_name-metric:result
   */
  virtual std::string Evaluate(IIterator<DataBatch> *iter_eval, const char* data_name) = 0;
  /*! \brief  predict labels */
  virtual void Predict(std::vector<float> &preds, const DataBatch& batch) = 0;
  virtual void PredictRaw( std::vector<std::vector<float> > &preds, const DataBatch &batch) = 0;
  virtual void PredictTop( std::vector<std::vector<float> > &preds, const DataBatch &batch) = 0;

};

/*!
 * \brief create a net implementation
 * \param net_type network type, used to select trainer variants
 * \tparam device type the net lies
 */
template<typename xpu>
INetTrainer* CreateNet(int net_type);
}  // namespace nnet
}  // namespace cxxnet
#endif // CXXNET_NNET_NNET_H_
