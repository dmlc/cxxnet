#ifndef CXXNET_WRAPPER_H_
#define CXXNET_WRAPPER_H_
/*!
 * \file cxxnet_wrapper.h
 * \author Tianqi Chen
 * \brief a C style wrapper of cxxnet
 *  can be used to create wrapper of other languages
 */
#include "../src/global.h"
#ifdef _MSC_VER
#define CXXNET_DLL __declspec(dllexport)
#else
#define CXXNET_DLL
#endif
// manually define unsign long
typedef unsigned long cxx_ulong;
typedef unsigned int cxx_uint;
typedef cxxnet::real_t cxx_real_t;

#ifdef __cplusplus
extern "C" {
#endif
  /*!
   * \brief create an cxxnet io iterator from config string
   * \param cfg config string that contains the configuration about
   *        the iterator
   * \return the handle pointer to the iterator
   */
  CXXNET_DLL void *CXNIOCreateFromConfig(const char *cfg);
  /*!
   * \brief move iterator to next position
   * \param handle the handle to iterator
   * \return whether it can be moved
   */
  CXXNET_DLL int CXNIONext(void *handle);
  /*!
   * \brief call iterator.BeforeFirst
   * \param handle the handle to iterator
   */
  CXXNET_DLL void CXNIOBeforeFirst(void *handle);
  /*!
   * \brief call iterator.Value().data
   * \param handle the handle to iterator
   * \param oshape the shape of output
   * \param ostride the stride of the output tensor
   */
  CXXNET_DLL const cxx_real_t *CXNIOGetData(void *handle,
                                            cxx_uint oshape[4],
                                            cxx_uint *ostride);
  /*!
   * \brief call iterator.Value().label
   * \param handle the handle to iterator
   * \param oshape the shape of output
   * \param ostride the stride of the output tensor
   */
  CXXNET_DLL const cxx_real_t *CXNIOGetLabel(void *handle,
                                             cxx_uint oshape[2],
                                             cxx_uint *ostride);
  /*!
   * \brief free the cxxnet io iterator handle
   * \param handle the handle pointer to the data iterator
   */
  CXXNET_DLL void CXNIOFree(void *handle);
  /*!
   * \brief create a cxxnet neural net object
   * \param devcie the device type of the net, corresponds to parameter devices
   *        can be NULL, if it is NULL, device type wil be decided by config
   * \param cfg configuration string of the net
   */
  CXXNET_DLL void *CXNNetCreate(const char *device, const char *cfg);
  /*!
   * \brief free the cxxnet neural net handle
   * \param handle net handle
   */
  CXXNET_DLL void CXNNetFree(void *handle);
  /*!
   * \brief set additional parameter to cxxnet
   * \param handle net handle
   * \param name name of parameter
   * \param val the value of parameter
   */
  CXXNET_DLL void CXNNetSetParam(void *handle, const char *name, const char *val);
  /*!
   * \brief initialize
   * \param handle net handle
   */
  CXXNET_DLL void CXNNetInitModel(void *handle);
  /*!
   * \brief save model into existing file
   * \param handle handle
   * \param fname file name
   */
  CXXNET_DLL void CXNNetSaveModel(void *handle, const char *fname);
  /*!
   * \brief load model from model file
   * \param handle net handle
   * \param fname file name
   */
  CXXNET_DLL void CXNNetLoadModel(void *handle, const char *fname);
  /*!
   * \brief inform the updater that a new round has been started
   * \param handle net handle
   * \param round round counter
   */
  CXXNET_DLL void CXNNetStartRound(void *handle, int round);
  /*!
   * \brief set weight by inputing an flattened array with same layout as original weight
   * \param handle net handle
   * \param p_weight pointer to the weight
   * \param size_weight size of the weight
   * \param layer_name the name of the layer
   * \param wtag the tag of weight, can be bias or wmat
   */
  CXXNET_DLL void CXNNetSetWeight(void *handle,
                                  cxx_real_t *p_weight,
                                  cxx_uint size_weight,
                                  const char *layer_name,
                                  const char *wtag);
  /*!
   * \brief get weight out
   * \param handle net handle
   * \param layer_name the name of the layer
   * \param wtag the tag of weight, can be bias or wmat
   * \param wshape the array holding output shape, weight can be maximumly 4 dim
   * \param out_dim the place holding dimension of output
   * \return the pointer to contiguous space of weight,
   *    can be NULL if weight do not exist
   */
  CXXNET_DLL const cxx_real_t *
  CXNNetGetWeight(void *handle,
                  const char *layer_name,
                  const char *wtag,
                  cxx_uint wshape[4],
                  cxx_uint *out_dim);
                                  
  /*!
   * \brief update the model, using current position on iterator
   * \param handle net handle
   * \param data_handle the data iterator handle
   */
  CXXNET_DLL void CXNNetUpdateIter(void *handle,
                                   void *data_handle);
  /*!
   * \brief update the model using one batch of image
   * \param handle net handle
   * \param p_data pointer to the data tensor, shape=(nbatch, nchannel, height, width)
   * \param dshape shape of input batch
   * \param p_label pointer to the label field, shape=(nbatch, label_width)
   * \param lshape shape of input label
   */
  CXXNET_DLL void CXNNetUpdateBatch(void *handle,
                                    cxx_real_t *p_data,
                                    const cxx_uint dshape[4],
                                    cxx_real_t *p_label,
                                    const cxx_uint lshape[2]);
  /*!
   * \brief make a prediction
   * \param handle net handle
   * \param p_data pointer to the data tensor, shape=(nbatch, nchannel, height, width)
   * \param dshape shape of input batch
   * \param out_size the final size of output label
   *
   * \return the pointer to the result field, the caller must copy the result out
   *         before calling any other cxxnet functions
   */
  CXXNET_DLL const cxx_real_t *
  CXNNetPredictBatch(void *handle,
                     cxx_real_t *p_data,
                     const cxx_uint dshape[4],
                     cxx_uint *out_size);
  /*!
   * \brief make a prediction based on iterator input
   * \param handle net handle
   * \param data_handle
   *
   * \return the pointer to the result field, the caller must copy the result out
   *         before calling any other cxxnet functions
   */  
  CXXNET_DLL const cxx_real_t *CXNNetPredictIter(void *handle,
                                                 void *data_handle,
                                                 cxx_uint *out_size);
  /*!
   * \brief make a feature extraction based on node name
   * \param handle net handle
   * \param p_data pointer to the data tensor, shape=(nbatch, nchannel, height, width)
   * \param dshape shape of input batch
   * \param out_size the final size of output label
   * \param node_name name of the node to be get feature from
   * \param oshape the shape out extracted data
   *
   * \return the pointer to the result field, the caller must copy the result out
   *         before calling any other cxxnet functions
   */
  CXXNET_DLL const cxx_real_t *
  CXNNetExtractBatch(void *handle,
                     cxx_real_t *p_data,
                     const cxx_uint dshape[4],
                     const char *node_name,
                     cxx_uint oshape[4]);
  /*!
   * \brief make a prediction based on iterator input
   * \param handle net handle
   * \param data_handle 
   * \param node_name name of the node to be get feature from
   * \param oshape the shape out extracted data

   * \return the pointer to the result field, the caller must copy the result out
   *         before calling any other cxxnet functions
   */  
  CXXNET_DLL const cxx_real_t *CXNNetExtractIter(void *handle,
                                                 void *data_handle,
                                                 const char *node_name,
                                                 cxx_uint oshape[4]);
  /*!
   * \brief evaluate the net using the data source
   * \param handle net handle
   * \param data_handle the data iterator handle
   * \param data_name the name of data, used to attach to the result
   *
   * \return a string representing the evaluation result, user need to copy the result out
   *         before claling any other cxxnet function
   */
  CXXNET_DLL const char* CXNNetEvaluate(void *handle,
                                        void *data_handle,
                                        const char *data_name);
#ifdef __cplusplus
}
#endif
#endif  // CXXNET_WRAPPER_H_
