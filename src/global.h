#ifndef CXXNET_GLOBAL_H_
/*!
 * \file global.h
 * \brief global configuration of cxxnet, this controls how cxxnet is compiled
 * \author Tianqi Chen
 */
/*! \brief whether to adapt caffe layers */
#ifndef CXXNET_ADAPT_CAFFE
#define CXXNET_ADAPT_CAFFE 0
#endif

/*! \brief whether to reuse xgboost input sparse data structure */
#ifndef CXXNET_ADAPT_XGBOOST
#define CXXNET_ADAPT_XGBOOST 0
#endif

/*! \brief whether to use opencv support, without it, we will not be able to use load image iterarators */
#ifndef CXXNET_USE_OPENCV
#define CXXNET_USE_OPENCV 1
#endif

/*! \brief namespace of cxxnet */
namespace cxxnet {
typedef mshadow::cpu cpu;
typedef mshadow::gpu gpu;
typedef mshadow::index_t index_t;
typedef mshadow::default_real_t real_t;
} // namespace cxxnet

#endif  // CXXNET_GLOBAL_H_
