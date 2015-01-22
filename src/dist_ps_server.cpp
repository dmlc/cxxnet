#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include <mshadow-ps/ps.h>
#include "./global.h"
// put it in PS namespace?
PS::App* CreateServer(const std::string& conf) {
  return new mshadow::MShadowServer<real_t>(conf);
}
