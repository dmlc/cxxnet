#ifndef CXXNET_UPDATER_PARAM_H_
#define CXXNET_UPDATER_PARAM_H_
/*!
 * \file param.h
 * \brief common parameters for updater behavior, supports complex learning rate scheduling
 * \author Tianqi Chen
 */
#include <string>

namespace cxxnet {
namespace updater {
/*! \brief potential parameters for each layer */
struct UpdaterParam {
  /*! \brief tag of current parameter group */
  std::string tag;
  /*! \brief current round */
  int round;
  /*! \brief whether can print messages */
  int silent;
  /*! \brief learning rate */
  float learning_rate;
  /*! \brief weight decay */
  float wd;
  /*! \brief momentum */
  float momentum;
  // scheduling parameters
  /*! \brief type of learning rate schedule */
  int lr_schedule;
  /*! \brief type of momentum schedule */
  int momentum_schedule;
  /*! \brief base learning rate */
  float base_lr_;
  /*! \brief period of lr decay */
  long lr_step;
  /*! \brief decay parameter gamma */
  float lr_gamma;
  /*! \brief decay parameter gamma */
  float lr_alpha;
  /*! \brief decay parameter factor */
  float lr_factor;
  /*! \brief minimum learning rate */
  float lr_minimum;
  /*! \brief start scheduling epoch */
  long start_epoch;
  /*! \brief base momentum */
  float base_momentum_;
  /*! \brief final momentum */
  float final_momentum_;
  /*! \brief constructor that sets default parameters */
  UpdaterParam(void) {
    base_lr_ = 0.01f;
    base_momentum_ = 0.5f;
    final_momentum_ = 0.65f;
    momentum_schedule = 0;
    lr_schedule = 0;
    lr_step = 1;
    lr_alpha = 0.5f;
    lr_gamma = 0.5f;
    lr_factor = 0.1f;
    lr_minimum = 0.00001;
    start_epoch = 0;
    wd = 0.0f;
    momentum = 0.9f;
    silent = 0;
  }
  /*! \brief do learning rate or other parameter schedule at round epoch */
  inline void ScheduleEpoch(long epoch) {
    if (epoch < start_epoch) {
      learning_rate = base_lr_;
      return;
    }
    switch (lr_schedule) {
      case 0: learning_rate = base_lr_; break;
      case 1: learning_rate = base_lr_ * powf(lr_gamma, epoch / lr_step); break;
      case 2: learning_rate = base_lr_ * powf(1.0f + (epoch/lr_step) * lr_gamma, -lr_alpha); break;
      case 3: learning_rate = base_lr_ * powf(lr_factor, epoch / lr_step); break;
      default: utils::Error("unknown schedule type");
    }
    learning_rate = learning_rate < lr_minimum ? lr_minimum : learning_rate;
  }
  /*!
   * \brief Set param for the layer from string
   * \param name parameter name
   * \param val string for configuration
   */
  inline void SetParam(const char *name, const char* val) {
    // if we set "bias:wd = 0.0", and tag == "bias", the it will set wd in current updater param
    // but will not affect settings with other tags
    if (!strncmp(name, tag.c_str(), tag.length())) {
      if (name[tag.length()] == ':') name += tag.length() + 1;
    }
    if (!strcmp(name, "lr")) base_lr_ = (float)atof(val);
    if (!strcmp(name, "eta")) base_lr_ = (float)atof(val);
    if (!strcmp(name, "wd")) wd = (float)atof(val);
    if (!strcmp(name, "momentum")) momentum = (float)atof(val);
    if (!strcmp(name, "silent")) silent = atoi(val);
    
    if (!strncmp(name, "lr:", 3) || !strncmp(name, "eta:",4)) {
      if (!strncmp(name, "lr:", 3)) name += 3;
      else name += 4;
      if (!strcmp(name, "schedule")) {
        if (!strcmp(val, "constant"))  lr_schedule = 0;
        if (!strcmp(val, "expdecay"))  lr_schedule = 1;
        if (!strcmp(val, "polydecay")) lr_schedule = 2;
        if (!strcmp(val, "factor"))     lr_schedule = 3;
      }
      if (!strcmp(name, "gamma")) lr_gamma = (float)atof(val);
      if (!strcmp(name, "alpha")) lr_alpha = (float)atof(val);
      if (!strcmp(name, "step"))  lr_step = atol(val);
      if (!strcmp(name, "factor")) lr_factor = (float)atof(val);
      if (!strcmp(name, "minimum_lr")) lr_minimum = (float)atof(val);
      if (!strcmp(name, "start_epoch")) start_epoch = atol(val);
    }
  }
};
}  // namespace updater
}  // namespace cxxnet
#endif  // CXXNET_UPDATER_PARAM_H_
