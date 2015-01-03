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
  /*! \brief initialization sparse weight */
  int init_sparse;
  /*! \brief initialization uniform for weight */
  float init_uniform;
  /*! \brief intialization value for bias */
  float init_bias;
  /*! \brief number of output channel */
  int num_channel;
  /*! \brief type of random number generation */
  int random_type;
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
  /*! \brief number of input channels */
  int num_input_channel;
  /*! \brief number of input hidden nodes, used by fullc */
  int num_input_node;
  /*! \brief reserved fields, for future compatibility */
  int reserved[64];
  /*! \brief construtor */
  LayerParam(void) {
    init_sigma = 0.01f;
    init_uniform = -1.0f;
    init_sparse = 10;
    init_bias  = 0.0f;
    random_type = 0;
    num_hidden = 0;
    num_channel = 0;
    num_group = 1;
    kernel_width = 0;
    kernel_height = 0;
    stride = 1;
    pad_x = pad_y = 0;
    no_bias = 0;
    silent = 0;
    num_input_channel = 0;
    num_input_node = 0;
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
    if (!strcmp(name, "init_uniform")) init_uniform = (float)atof(val);
    if (!strcmp(name, "init_bias")) init_bias  = (float)atof(val);
    if (!strcmp(name, "init_sparse")) init_sparse = atoi(val);
    if (!strcmp(name, "random_type")) {
      if (!strcmp(val, "gaussian")) random_type = 0;
      else if (!strcmp(val, "uniform")) random_type = 1;
      else if (!strcmp(val, "xavier")) random_type = 1;
      else if (!strcmp(val, "sparse")) random_type = 2;
      else utils::Error("invalid random_type %s", val);
      // 3: mshadow binary file
    }    
    if (!strcmp(name, "nhidden")) num_hidden = atoi(val);
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
      prng->SampleGaussian(&mat, 0.0f, init_sigma);
    } else if (random_type == 1) {
      // uniform initialization
      real_t a = sqrt(3.0f / (in_num + out_num));
      if (init_uniform > 0) a = init_uniform;
      prng->SampleUniform(&mat, -a, a);
    } else if (random_type == 2) {
      // sparse initalization
      real_t a = sqrt(3.0f / (in_num + out_num));
      utils::Check(dim == 2, "Sparse init only support 2 dim");
      std::vector<real_t> tmp(mat.MSize(), 0.0f);
      mshadow::Tensor<cpu, dim> cpu_mat(&tmp[0], mat.shape_);
      for (int i = 0; i < init_sparse; ++i) {
        int idx = static_cast<int>(in_num * static_cast<double>(rand())/RAND_MAX);
        int rej = 0;
        int j = i;
        int loc = j * mat.stride_ + idx;
        while (tmp[loc] > 0.0f) {
          rej++;
          idx = static_cast<int>(in_num * static_cast<double>(rand())/RAND_MAX);
          if (rej > 10) j = static_cast<int>(out_num * static_cast<double>(rand())/RAND_MAX);
          if (rej > 20) break;
        }
        tmp[loc] = ((static_cast<float>(rand())/RAND_MAX) > 0.5f ? 1.0f : -1.0f) * \
            a * static_cast<float>(rand())/RAND_MAX;
      }
      mshadow::Copy(mat, cpu_mat);
    } else if (random_type == 3) {
      // mshadow::utils::LoadBinary(fi, mat, false);
    }
  }
};
}  // namespace layer
}  // namespace cxxnet
#endif   // CXXNET_LAYER_PARAM_H_
