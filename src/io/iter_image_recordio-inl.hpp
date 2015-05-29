/*!
 * \file iter_image_recordio-inl.hpp
 * \brief recordio data
iterator
 */
#ifndef ITER_IMAGE_RECORDIO_INL_HPP_
#define ITER_IMAGE_RECORDIO_INL_HPP_
#include <cstdlib>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/omp.h>
#include <dmlc/logging.h>
#include <dmlc/recordio.h>
// this code needs c++11
#if DMLC_USE_CXX11
#include <dmlc/threadediter.h>
#include <unordered_map>
#include <vector>
#include "./data.h"
#include "./inst_vector.h"
#include "./image_recordio.h"
#include "./image_augmenter-inl.hpp"
#include "../utils/decoder.h"
#include "../utils/random.h"
namespace cxxnet {

/*! \brief data structure to hold labels for images */
class ImageLabelMap {
 public:
  /*!
   * \brief initialize the label list into memory
   * \param path_imglist path to the image list
   * \param label_width predefined label_width
   */
  explicit ImageLabelMap(const char *path_imglist,
                         mshadow::index_t label_width,
                         bool silent) {
    label_width_ = label_width;
    image_index_.clear();
    label_.clear();
    idx2label_.clear();
    dmlc::InputSplit *fi = dmlc::InputSplit::Create
        (path_imglist, 0, 1, "text");
    dmlc::InputSplit::Blob rec;
    while (fi->NextRecord(&rec)) {
      // quick manual parsing
      char *p = reinterpret_cast<char*>(rec.dptr);
      char *end = p + rec.size;
      // skip space
      while (isspace(*p) && p != end) ++p;
      image_index_.push_back(static_cast<size_t>(atol(p)));
      for (size_t i = 0; i < label_width_; ++i) {
        // skip till space
        while (!isspace(*p) && p != end) ++p;
        // skip space
        while (isspace(*p) && p != end) ++p;
        CHECK(p != end) << "Bad ImageList format";
        label_.push_back(static_cast<real_t>(atof(p)));
      }
    }
    delete fi;
    // be careful not to resize label_ afterwards
    idx2label_.reserve(image_index_.size());
    for (size_t i = 0; i < image_index_.size(); ++i) {
      idx2label_[image_index_[i]] = BeginPtr(label_) + i * label_width_;
    }
    if (!silent) {
      LOG(INFO) << "Loaded ImageList from " << path_imglist << ' '
                << image_index_.size() << " Image records";
    }
  }
  /*! \brief find a label for corresponding index */
  inline mshadow::Tensor<cpu, 1> Find(size_t imid) const {
    std::unordered_map<size_t, real_t*>::const_iterator it
        = idx2label_.find(imid);
    CHECK(it != idx2label_.end()) << "fail to find imagelabel for id " << imid;
    return mshadow::Tensor<cpu, 1>(it->second, mshadow::Shape1(label_width_));
  }

 private:
  // label with_
  mshadow::index_t label_width_;
  // image index of each record
  std::vector<size_t> image_index_;
  // real label content
  std::vector<real_t> label_;
  // map index to label
  std::unordered_map<size_t, real_t*> idx2label_;
};

// parser to parse image recordio
class ImageRecordIOParser {
 public:
  ImageRecordIOParser(int nthread = 4)
      : nthread_(nthread),
        source_(NULL),
        label_map_(NULL) {
    silent_ = 0;
    dist_num_worker_ = 1;
    dist_worker_rank_ = 0;
    label_width_ = 1;
    int maxthread;
    #pragma omp parallel
    {
      maxthread = std::max(omp_get_num_procs() / 2 - 1, 1);
    }
    nthread_ = std::min(maxthread, nthread_);
    #pragma omp parallel num_threads(nthread_)
    {
      nthread = omp_get_num_threads();
    }
    nthread_ = nthread;
    // setup decoders
    for (int i = 0; i < nthread; ++i) {
      augmenters_.push_back(new ImageAugmenter());
      prnds_.push_back(new utils::RandomSampler());
      prnds_[i]->Seed((i + 1) * kRandMagic);
    }
  }
  ~ImageRecordIOParser(void) {
    // can be NULL
    delete label_map_;
    delete source_;
    for (size_t i = 0; i < augmenters_.size(); ++i) {
      delete augmenters_[i];
    }
    for (size_t i = 0; i < prnds_.size(); ++i) {
      delete prnds_[i];
    }
  }
  // initialize the parser
  inline void Init(void);
  // set parameters of the parser
  inline void SetParam(const char *name,
                       const char *val);
  // set record to the head
  inline void BeforeFirst(void) {
    return source_->BeforeFirst();
  }
  // parse next set of records, return an array of
  // instance vector to the user
  inline bool ParseNext(std::vector<InstVector> *out);
 private:
  // magic nyumber to see prng
  static const int kRandMagic = 111;
  /*! \brief whether to remain silent */
  int silent_;
  /*! \brief number of distributed worker */
  int dist_num_worker_, dist_worker_rank_;
  /*! \brief path to image list */
  std::string path_imglist_;
  /*! \brief path to image recordio */
  std::string path_imgrec_;
  /*! \brief number of threads */
  int nthread_;
  /*! \brief augmenters */
  std::vector<ImageAugmenter*> augmenters_;
  /*! \brief random samplers */
  std::vector<utils::RandomSampler*> prnds_;
  /*! \brief label-width */
  int label_width_;
  /*! \brief data source */
  dmlc::InputSplit *source_;
  /*! \brief label information, if any */
  ImageLabelMap *label_map_;
};

inline void ImageRecordIOParser::Init(void) {
  // handling for hadoop
  const char *ps_rank = getenv("PS_RANK");
  if (ps_rank != NULL) {
    this->SetParam("dist_worker_rank", ps_rank);
  }

  if (path_imglist_.length() != 0) {
    label_map_ = new ImageLabelMap(path_imglist_.c_str(),
                                   label_width_, silent_ != 0);
  } else {
    label_width_ = 1;
  }
  CHECK(path_imgrec_.length() != 0)
    << "ImageRecordIOIterator: must specify image_rec";
#if MSHADOW_DIST_PS
    // TODO move to a better place
    dist_num_worker_ = ::ps::RankSize();
    dist_worker_rank_ = ::ps::MyRank();
    LOG(INFO) << "rank " << dist_worker_rank_
              << " in " << dist_num_worker_;
#endif
  source_ = dmlc::InputSplit::Create
      (path_imgrec_.c_str(), dist_worker_rank_,
       dist_num_worker_, "recordio");
  // use 64 MB chunk when possible
  source_->HintChunkSize(8 << 20UL);
}
inline void ImageRecordIOParser::
SetParam(const char *name, const char *val) {
  using namespace std;
  for (int i = 0; i < nthread_; ++i) {
    augmenters_[i]->SetParam(name, val);
  }
  if (!strcmp(name, "image_list")) path_imglist_ = val;
  if (!strcmp(name, "image_rec")) path_imgrec_ = val;
  if (!strcmp(name, "dist_num_worker")) {
    dist_num_worker_ = atoi(val);
  }
  if (!strcmp(name, "dist_worker_rank")) {
    dist_worker_rank_ = atoi(val);
  }
  if (!strcmp(name, "label_width")) {
    label_width_ = atoi(val);
  }
}

inline bool ImageRecordIOParser::
ParseNext(std::vector<InstVector> *out_vec) {
  CHECK(source_ != NULL);
  dmlc::InputSplit::Blob chunk;
  if (!source_->NextChunk(&chunk)) return false;
  out_vec->resize(nthread_);
  #pragma omp parallel num_threads(nthread_)
  {
    CHECK(omp_get_num_threads() == nthread_);
    int tid = omp_get_thread_num();
    dmlc::RecordIOChunkReader reader(chunk, tid, nthread_);
    cxxnet::ImageRecordIO rec;
    dmlc::InputSplit::Blob blob;
    // image data
    InstVector &out = (*out_vec)[tid];
    out.Clear();
    while (reader.NextRecord(&blob)) {
      // result holder
      cv::Mat res;
      rec.Load(blob.dptr, blob.size);
      cv::Mat buf(1, rec.content_size, CV_8U, rec.content);
      res = cv::imdecode(buf, 1);
      res = augmenters_[tid]->Process(res, prnds_[tid]);
      out.Push(static_cast<unsigned>(rec.image_index()),
               mshadow::Shape3(3, res.rows, res.cols),
               mshadow::Shape1(label_width_));
      DataInst inst = out.Back();
      for (int i = 0; i < res.rows; ++i) {
        for (int j = 0; j < res.cols; ++j) {
          cv::Vec3b bgr = res.at<cv::Vec3b>(i, j);
          inst.data[0][i][j] = bgr[2];
          inst.data[1][i][j] = bgr[1];
          inst.data[2][i][j] = bgr[0];
        }
      }
      if (label_map_ != NULL) {
        mshadow::Copy(inst.label, label_map_->Find(rec.image_index()));
      } else {
        inst.label[0] = rec.header.label;
      }
      res.release();
    }
  }
  return true;
}

// iterator on image recordio
class ImageRecordIOIterator : public IIterator<DataInst> {
 public:
  ImageRecordIOIterator()
      : data_(NULL) {
    rnd_.Seed(kRandMagic);
    shuffle_ = 0;
  }
  virtual ~ImageRecordIOIterator(void) {
    iter_.Destroy();
    // data can be NULL
    delete data_;
  }
  virtual void SetParam(const char *name, const char *val) {
    parser_.SetParam(name, val);
    if (!strcmp(name, "seed_data")) {
      rnd_.Seed(atoi(val) + kRandMagic);
    }
    if (!strcmp(name, "shuffle")) {
      shuffle_ = atoi(val);
    }
  }
  virtual void Init(void) {
    parser_.Init();
    iter_.set_max_capacity(4);
    iter_.Init([this](std::vector<InstVector> **dptr) {
        if (*dptr == NULL) {
          *dptr = new std::vector<InstVector>();
        }
        return parser_.ParseNext(*dptr);
      },
      [this]() { parser_.BeforeFirst(); });
    inst_ptr_ = 0;
  }
  virtual void BeforeFirst(void) {
    iter_.BeforeFirst();
    inst_order_.clear();
    inst_ptr_ = 0;
  }
  virtual bool Next(void) {
    while (true) {
      if (inst_ptr_ < inst_order_.size()) {
        std::pair<unsigned, unsigned> p = inst_order_[inst_ptr_];
        out_ = (*data_)[p.first][p.second];
        ++inst_ptr_;
        return true;
      } else {
        if (data_ != NULL) iter_.Recycle(&data_);
        if (!iter_.Next(&data_)) return false;
        inst_order_.clear();
        for (unsigned i = 0; i < data_->size(); ++i) {
          const InstVector &tmp = (*data_)[i];
          for (unsigned j = 0; j < tmp.Size(); ++j) {
            inst_order_.push_back(std::make_pair(i, j));
          }
        }
        // shuffle instance order if needed
        if (shuffle_ != 0) {
          rnd_.Shuffle(inst_order_);
        }
        inst_ptr_ = 0;
      }
    }
    return false;
  }
  virtual const DataInst &Value(void) const {
    return out_;
  }

 private:
  // random magic
  static const int kRandMagic = 111;
  // output instance
  DataInst out_;
  // whether shuffle data
  int shuffle_;
  // data ptr
  size_t inst_ptr_;
  // random sampler
  utils::RandomSampler rnd_;
  // internal instance order
  std::vector<std::pair<unsigned, unsigned> > inst_order_;
  // data
  std::vector<InstVector> *data_;
  // internal parser
  ImageRecordIOParser parser_;
  // backend thread
  dmlc::ThreadedIter<std::vector<InstVector> > iter_;
};
}  // namespace cxxnet
#endif
#endif  // ITER_IMAGE_RECORDIO_INL_HPP_
