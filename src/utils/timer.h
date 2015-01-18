/*!
* \file timer.h
* \brief This file defines the utils for timing
* \author Tianqi Chen, Nacho, Tianyi
*/
#ifndef CXXNET_TIMER_H
#define CXXNET_TIMER_H
#include <time.h>
#include <string>
namespace cxxnet {
namespace utils {
	inline double GetTime(void) {
		timespec ts;
		utils::Check(clock_gettime(CLOCK_REALTIME, &ts) == 0, "failed to get time");
		return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
	}
}
}
#endif