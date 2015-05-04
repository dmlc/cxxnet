#ifndef CXXNET_UTILS_UTILS_H_
#define CXXNET_UTILS_UTILS_H_
/*!
 * \file utils.h
 * \brief simple utils to support the code, adopted from xgboost
 * \author Tianqi Chen
 */
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstdarg>
#include <dmlc/base.h>
#include <dmlc/logging.h>
#if MSHADOW_RABIT_PS
#include <rabit.h>
#endif

#define CUDA_CHECK(ARGS) CHECK(ARGS==0);

namespace cxxnet {
/*! \brief include dmlc objects in cxxnet interface */
using namespace dmlc;
/*! \brief namespace for helper utils of the project */
namespace utils {
/*! \brief error message buffer length */
const int kPrintBuffer = 1 << 12;

#ifndef CXXNET_CUSTOMIZE_MSG_
/*!
 * \brief handling of Assert error, caused by in-apropriate input
 * \param msg error message
 */
inline void HandleAssertError(const char *msg) {
  fprintf(stderr, "AssertError:%s\n", msg);
  exit(-1);
}
/*!
 * \brief handling of Check error, caused by in-apropriate input
 * \param msg error message
 */
inline void HandleCheckError(const char *msg) {
  fprintf(stderr, "%s\n", msg);
  exit(-1);
}
inline void HandlePrint(const char *msg) {
  printf("%s", msg);
}
#else
// include declarations, some one must implement this
void HandleAssertError(const char *msg);
void HandleCheckError(const char *msg);
void HandlePrint(const char *msg);
#endif

/*! \brief printf, print message to the console */
inline void Printf(const char *fmt, ...) {
  std::string msg(kPrintBuffer, '\0');
  va_list args;
  va_start(args, fmt);
  vsnprintf(&msg[0], kPrintBuffer, fmt, args);
  va_end(args);
  HandlePrint(msg.c_str());
}
/*! \brief portable version of snprintf */
inline int SPrintf(char *buf, size_t size, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int ret = vsnprintf(buf, size, fmt, args);
  va_end(args);
  return ret;
}

/*!\brief same as assert, but this is intended to be used as message for user*/
inline void Check(bool exp, const char *fmt, ...) {
  if (!exp) {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleCheckError(msg.c_str());
  }
}

/*! \brief report error message, same as check */
inline void Error(const char *fmt, ...) {
  {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleCheckError(msg.c_str());
  }
}

/*! \brief replace fopen, report error when the file open fails */
inline std::FILE *FopenCheck(const char *fname, const char *flag) {
  std::FILE *fp = fopen64(fname, flag);
  Check(fp != NULL, "can not open file \"%s\"\n", fname);
  return fp;
}
// print message to tracker
inline void TrackerPrint(const std::string msg, bool root_only = true) {
#if MSHADOW_RABIT_PS
  if (!root_only || rabit::GetRank() == 0) {
    rabit::TrackerPrint(msg);
  }
#else
  fprintf(stderr, "%s", msg.c_str());
  fflush(stderr);
#endif
}

/*! \brief portable version of snprintf */
inline void TrackerPrintf(const char *fmt, ...) {
  const int kPrintBuffer = 1 << 10;
  std::string msg(kPrintBuffer, '\0');
  va_list args;
  va_start(args, fmt);
  vsnprintf(&msg[0], kPrintBuffer, fmt, args);
  va_end(args);
  msg.resize(strlen(msg.c_str()));
  TrackerPrint(msg);
}
}  // namespace utils
}  // namespace cxxnet
#endif  // CXXNET_UTILS_UTILS_H_
