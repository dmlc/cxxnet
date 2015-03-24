#ifndef CXXNET_UTILS_GLOBAL_RANDOM_H_
#define CXXNET_UTILS_GLOBAL_RANDOM_H_
/*!
 * \file global_random.h
 * \brief global random number utils, used for some preprocessing
 * \author Tianqi Chen
 */
#include <cstdlib>
#include <vector>
#include <cmath>
#include "./utils.h"

#if _MSC_VER
#define rand_r(x) rand()
#endif


namespace cxxnet {
namespace utils {
/*! \brief simple thread dependent random sampler */
class RandomSampler {
 public:
  RandomSampler(void) {
    this->Seed(0);
  }
  /*!
   * \brief seed random number
   * \param seed the random number seed
   */
  inline void Seed(unsigned seed) {
    this->rseed_ = seed;
#if _MSC_VER
    srand(seed);
#endif    
  }
  /*! \brief return a real number uniform in [0,1) */
  inline double NextDouble() {
    return static_cast<double>(rand_r(&rseed_)) /
        (static_cast<double>(RAND_MAX) + 1.0);
  }
  /*! \brief return a random number in n */
  inline uint32_t NextUInt32(uint32_t n) {
    return static_cast<uint32_t>(floor(NextDouble() * n));
  }
  /*! \brief random shuffle data */
  template<typename T>
  inline void Shuffle(T *data, size_t sz) {
    if(sz == 0) return;
    for(uint32_t i = (uint32_t)sz - 1; i > 0; i--) {
      std::swap(data[i], data[NextUInt32(i+1)]);
    }
  }
  /*!\brief random shuffle data in */
  template<typename T>
  inline void Shuffle(std::vector<T> &data) {
    Shuffle(&data[0], data.size());
  }

 private:
  unsigned rseed_;
};
}  // namespace utils
}  // namespace cxxnet
#endif
