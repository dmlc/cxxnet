#ifndef CXXNET_LAYER_PARAM_H_
#define CXXNET_LAYER_PARAM_H_
/*!
 * \file param.h
 * \brief commonly used parameters in each layer
 *        the LayerParam is not a must for each layer but can be helpful
 * \author Bing Xu, Tianqi Chen
 */
#include <cstring>
#include <string>

namespace cxxnet {
namespace layer {
/*! \brief potential parameters for each layer */
struct LayerParam {
  /*! \brief number of hidden layers */
  int num_hidden;
  /*! \brief initialization sd for weight */
  float init_sigma;
  /*! \brief intialization value for bias */
  float init_bias;
  /*! \brief initialization random type */
  int random_type;
  /*! \brief number of output channel */
  int num_channel;
  /*! \brief number of parallel group */
  int num_group;
  /*! \brief kernel height */        
  int kernel_height;
  /*! \brief kernel width */        
  int kernel_width;
  /*! \brief stride prameter */
  int stride;
  /*! \brief padding in y dimension */
  int pad_y;
  /*! \brief padding in x dimension */
  int pad_x;
  /*! \brief whether not include bias term */
  int no_bias;
  /*! \brief maximum temp_col_size allowed in each layer, we need at least one temp col */
  int temp_col_max;
  /*! \brief shut up */
  int silent;
  /*! \brief reserved fields, for future compatibility */
  int reserved[64];
  /*! \brief construtor */
  LayerParam(void) {
    init_sigma = 0.01f;
    init_bias  = 0.0f;
    num_hidden = 0;
    random_type = 0;
    num_channel = 0;
    num_group = 1;
    kernel_width = 0;
    kernel_height = 0;
    stride = 1;
    pad_x = pad_y = 0;
    no_bias = 0;
    silent = 0;
    // 64 MB
    temp_col_max = 64<<18;
    memset(reserved, 0, sizeof(reserved));
  }
  /*!
   * \brief Set param for the layer from string
   * \param name parameter name
   * \param val string for configuration
   */
  inline void SetParam(const char *name, const char* val) {
    if (!strcmp(name, "init_sigma")) init_sigma = (float)atof(val);
    if (!strcmp(name, "init_bias")) init_bias  = (float)atof(val);
    if (!strcmp(name, "nhidden")) num_hidden = atoi(val);
    if (!strcmp(name, "random_type") && !strcmp(val, "gaussian"))  random_type = 0;
    if (!strcmp(name, "random_type") && !strcmp(val, "xavier")) random_type = 1;
    if (!strcmp(name, "nchannel")) num_channel = atoi(val);
    if (!strcmp(name, "ngroup")) num_group = atoi(val);
    if (!strcmp(name, "kernel_size")) {
      kernel_width = kernel_height = atoi(val);
    }
    if (!strcmp(name, "kernel_height")) kernel_height = atoi(val);
    if (!strcmp(name, "kernel_width")) kernel_width = atoi(val);
    if (!strcmp(name, "stride")) stride = atoi(val);
    if (!strcmp(name, "pad")) {
      pad_y = pad_x  = atoi(val);
    }
    if (!strcmp(name, "pad_y")) pad_y = atoi(val);
    if (!strcmp(name, "pad_x")) pad_x = atoi(val);
    if (!strcmp(name, "no_bias")) no_bias = atoi(val);
    if (!strcmp(name, "silent")) silent = atoi(val);
    if (!strcmp(name, "temp_col_max")) temp_col_max = atoi(val) << 18;
  }
  
  template<int dim, typename xpu>
  inline void RandInitWeight(mshadow::Random<xpu> *prng,
                             mshadow::Tensor<xpu, dim> mat,
                             index_t in_num, index_t out_num) {
    if (random_type == 0) {
      // gaussian initialization
      prng->SampleGaussian(mat, 0.0f, init_sigma);
    } else {
      // xavier initialization
      real_t a = sqrt(3.0f / (in_num + out_num));
      prng->SampleUniform(mat, -a, a);
    }
  }                             
};
}  // namespace layer
}  // namespace cxxnet
#endif   // CXXNET_LAYER_PARAM_H_
