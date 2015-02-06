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
   * \brief update the model for one iteration
   * \param handle net handle
   * \param data_handle the data iterator handle
   */
  CXXNET_DLL void CXNNetUpdateOneIter(void *handle,
                                      void *data_handle);
  /*!
   * \brief update the model using one batch of image
   * \param handle net handle
   * \param p_data pointer to the data tensor, shape=(nbatch, nchannel, height, width)
   * \param nbatch number of batch in the image
   * \param nchannel number of channels in the image
   * \param height height of image
   * \param width width of image
   * \param p_label pointer to the label field, shape=(nbatch, label_width)
   * \param label_width number of labels 
   */
  CXXNET_DLL void CXNNetUpdateOneBatch(void *handle,
                                       cxx_real_t *p_data,
                                       cxx_uint nbatch,
                                       cxx_uint nchannel,
                                       cxx_uint height,
                                       cxx_uint width,
                                       cxx_real_t *p_label,
                                       cxx_uint label_width);
  /*!
   * \brief make a prediction
   * \param handle net handle
   * \param p_data pointer to the data tensor, shape=(nbatch, nchannel, height, width)
   * \param nbatch number of batch in the image
   * \param nchannel number of channels in the image
   * \param height height of image
   * \param width width of image
   * \param out_size the final size of output label
   *
   * \return the pointer to the result field, the caller must copy the result out
   *         before calling any other cxxnet functions
   */
  CXXNET_DLL const cxx_real_t *CXNNetPredict(void *handle,
                                             cxx_real_t *p_data,
                                             cxx_uint nbatch,
                                             cxx_uint nchannel,
                                             cxx_uint height,
                                             cxx_uint width,
                                             cxx_uint *out_size);
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
