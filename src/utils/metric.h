#ifndef CXXNET_UTILS_METRIC_H_
#define CXXNET_UTILS_METRIC_H_
#pragma once
/*!
 * \file cxxnet_metric.h
 * \brief evaluation metrics, to be moved to nnet
 * \author Tianqi Chen
 */
#include <cmath>
#include <vector>
#include <algorithm>
#include <sstream>
#include "./random.h"
#include "../layer/layer.h"

#if MSHADOW_RABIT_PS
#include <rabit.h>
#endif

namespace cxxnet {
namespace utils {

using namespace cxxnet::layer;
/*! \brief evaluator that evaluates the loss metrics */
class IMetric{
 public:
  IMetric(void) {}
  /*!\brief virtual destructor */
  virtual ~IMetric(void) {}
  /*! \brief clear statistics */
  virtual void Clear(void) = 0;
  /*!
   * \brief evaluate a specific metric, add to current statistics
   * \param preds prediction score array
   * \param labels label
   * \param n number of instances
   */
  virtual void AddEval(const mshadow::Tensor<cpu,2> &predscore,
                       const LabelRecord& labels) = 0;
  /*! \brief get current result */
  virtual double Get(void) const = 0;
  /*! \return name of metric */
  virtual const char *Name(void) const= 0;
};

/*! \brief simple metric Base */
struct MetricBase : public IMetric {
 public:
  virtual ~MetricBase(void) {}
  virtual void Clear(void) {
    sum_metric = 0.0; cnt_inst = 0;
  }
  virtual void AddEval(const mshadow::Tensor<cpu,2> &predscore,
                       const LabelRecord& labels) {
    for (index_t i = 0; i < predscore.size(0); ++ i) {
      sum_metric += CalcMetric(predscore[i], labels.label[i]);
      cnt_inst+= 1;
    }
  }
  virtual double Get(void) const {
    double tmp[2];
    tmp[0] = sum_metric;
    tmp[1] = static_cast<double>(cnt_inst);
#if MSHADOW_RABIT_PS    
    rabit::Allreduce<rabit::op::Sum>(tmp, 2);
#endif    
    return tmp[0] / tmp[1];
  }
  virtual const char *Name(void) const {
    return name.c_str();
  }
 protected:
  MetricBase(const char *name) {
    this->name = name;
    this->Clear();
  }
  virtual float CalcMetric(const mshadow::Tensor<cpu,1> &predscore,
                           const mshadow::Tensor<cpu,1> &label) = 0;
 private:
  double sum_metric;
  long   cnt_inst;
  std::string name;
};

/*! \brief RMSE */
struct MetricRMSE : public MetricBase{
 public:
  MetricRMSE(void) : MetricBase("rmse") {
  }
  virtual ~MetricRMSE(void) {}
 protected:
  virtual float CalcMetric(const mshadow::Tensor<cpu,1> &predscore,
                           const mshadow::Tensor<cpu,1> &label) {
    utils::Check(predscore.size(0) == label.size(0),
                 "Metric: In RMSE metric, the size of prediction and label must be same.");
    float diff = 0;
    for (index_t i = 0; i < label.size(0); ++i) {
      diff += (predscore[i] - label[i]) * (predscore[i] - label[i]);
    }
    return diff;
  }
};

/*! \brief Error */
struct MetricError : public MetricBase{
 public:
  MetricError(void) : MetricBase("error") {
  }
  virtual ~MetricError(void) {}
 protected:
  virtual float CalcMetric(const mshadow::Tensor<cpu,1> &pred,
    const mshadow::Tensor<cpu,1> &label) {
    index_t maxidx = 0;
    if (pred.size(0) != 1) {
      for (index_t i = 1; i < pred.size(0); ++ i) {
        if (pred[i] > pred[maxidx]) maxidx = i;
      }
    }else{
      maxidx = pred[0] > 0.0 ? 1 : 0;
    }
    return maxidx !=(index_t)label[0];
  }
};

/*! \brief Logloss */
struct MetricLogloss : public MetricBase{
 public:
  MetricLogloss(void) : MetricBase("logloss") {
  }
  virtual ~MetricLogloss(void) {}
 protected:
  virtual float CalcMetric(const mshadow::Tensor<cpu,1> &pred,
    const mshadow::Tensor<cpu,1> &label) {
    int target = static_cast<int>(label[0]);
    if (pred.size(0) != 1) {
      return - std::log(std::max(std::min(pred[target], 1.0f - 1e-15f), 1e-15f));
    } else {
      const float py = std::max(std::min(pred[0], 1.0f - 1e-15f), 1e-15f);
      const float y = label[0];
      const float res = - (y * std::log(py) + (1.0f - y)*std::log(1 - py));
      utils::Check(res == res, "NaN detected!");
      return res;
    }
  }
};

/*! \brief Recall@n */
struct MetricRecall : public MetricBase {
 public:
  MetricRecall(const char *name) : MetricBase(name) {
    CHECK(sscanf(name, "rec@%d", &topn) == 1) << "must specify n for rec@n";
  }
  virtual ~MetricRecall(void) {}
 protected:
  virtual float CalcMetric(const mshadow::Tensor<cpu,1> &pred,
                           const mshadow::Tensor<cpu,1> &label) {
    utils::Check(pred.size(0) >= (index_t)topn,
                 "it is meaningless to take rec@n for list shorter than n, evaluating rec@%d, list=%u\n",
      topn, pred.size(0));
    vec.resize(pred.size(0));
    for (index_t i = 0; i < pred.size(0); ++ i) {
      vec[i] = std::make_pair(pred[i], i);
    }
    rnd.Shuffle(vec);
    std::sort(vec.begin(), vec.end(), CmpScore);
    int hit = 0;
    for (int i = 0; i < topn; ++ i) {
      for (index_t j = 0; j < label.size(0); ++j){
        if (vec[i].second == static_cast<index_t>(label[j])) {
          ++hit;
          break;
        }
      }
    }
    return (float)hit / label.size(0);
  }
 private:
  inline static bool CmpScore(const std::pair<float,index_t> &a,
                              const std::pair<float,index_t> &b) {
    return a.first > b.first;
  }

  std::vector< std::pair<float,index_t> > vec;
  utils::RandomSampler rnd;
  int topn;
};

/*! \brief a set of evaluators */
struct MetricSet{
 public:
  ~MetricSet(void) {
    for (size_t i = 0; i < evals_.size(); ++ i) {
      delete evals_[i];
    }
  }
  static IMetric* Create(const char *name) {
    if (!strcmp(name, "rmse")) return new MetricRMSE();
    if (!strcmp(name, "error")) return new MetricError();
    if (!strcmp(name, "logloss")) return new MetricLogloss();
    if (!strncmp(name, "rec@",4)) return new MetricRecall(name);
    return NULL;
  }
  void AddMetric(const char *name, const char* field) {
    IMetric *metric = this->Create(name);
    if (metric != NULL){
      evals_.push_back(metric);
      label_fields_.push_back(field);
    } else {
      utils::Error("Metric: Unknown metric name: %s\n", name);
    }
  }
  inline void Clear(void) {
    for (size_t i = 0; i < evals_.size(); ++ i) {
      evals_[i]->Clear();
    }
  }
  inline void AddEval(const std::vector<mshadow::Tensor<cpu, 2> >& predscores,
    const layer::LabelInfo& labels) {
    CHECK(predscores.size() == evals_.size())
        << "Metric: Number of predict scores and number of metrics should be equal.";
    for (size_t i = 0; i < evals_.size(); ++ i) {
      std::map<std::string, size_t>::const_iterator it =
        labels.name2findex->find(label_fields_[i]);
      utils::Check(it != labels.name2findex->end(), "Metric: unknown target = %s",
                 label_fields_[i].c_str());
      evals_[i]->AddEval(predscores[i], labels.fields[it->second]);
    }
  }
  inline std::string Print(const char *evname) {
    std::stringstream ss;
    for (size_t i = 0; i < evals_.size(); ++ i) {
      ss << '\t' << evname << '-' << evals_[i]->Name();
      if (label_fields_[i] != "label") {
        ss << '[' << label_fields_[i] << ']';
      }
      ss << ':' << evals_[i]->Get();
    }
    return ss.str();
  }
 private:
  inline static bool CmpName(const IMetric *a, const IMetric *b) {
    return strcmp(a->Name(), b->Name()) < 0;
  }
  inline static bool EqualName(const IMetric *a, const IMetric *b) {
    return strcmp(a->Name(), b->Name()) == 0;
  }
 private:
  std::vector<IMetric*> evals_;
  std::vector<std::string> label_fields_;
};
}  // namespace utils
}  // namespace cxxnet
#endif
