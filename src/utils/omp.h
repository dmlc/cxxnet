
#ifndef CXXNET_UTILS_OMP_H_
#define CXXNET_UTILS_OMP_H_
/*!
 * \file omp.h
 * \brief header to handle OpenMP compatibility issues
 * \author Tianqi Chen
 */
#if CXXNET_USE_OPENMP
#include <omp.h>
#else
// use pragma message instead of warning
#pragma message ("Warning: OpenMP is not available, inst iter will be compiled into single-thread code. Use OpenMP-enabled compiler to get benefit of multi-threading")
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline void omp_set_num_threads(int nthread) {}
#endif

// loop variable used in openmp
namespace cxxnet {
#ifdef _MSC_VER
typedef int bst_omp_uint;
#else
typedef unsigned bst_omp_uint;
#endif
} // namespace cxxnet

#endif  // CXXNET_UTILS_OMP_H_
