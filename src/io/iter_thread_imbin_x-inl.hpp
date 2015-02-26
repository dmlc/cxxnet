#ifndef ITER_THREAD_IMBIN_X_INL_HPP_
#define ITER_THREAD_IMBIN_X_INL_HPP_
/*!
 * \file cxxnet_iter_thread_imbin-inl.hpp
 * \brief threaded version of page iterator
 * \author Tianqi Chen
 */
#include "data.h"
#include <cstdlib>
#include <omp.h>
#include "../utils/thread_buffer.h"
#include "../utils/utils.h"
#include "../utils/decoder.h"
#include "../utils/random.h"

namespace cxxnet {
/*! \brief thread buffer iterator */
class ThreadImagePageIteratorX: public IIterator<DataInst> {
public:
  ThreadImagePageIteratorX(void) {
    silent_ = 0;
    itrpage.SetParam("buffer_size", "2");
    itrimg.SetParam("buffer_size", "256");
    img_conf_prefix_ = "";
    dist_num_worker_ = 0;
    dist_worker_rank_ = 0;
  }
  virtual ~ThreadImagePageIteratorX(void) {
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "image_list")) {
      raw_imglst_ += val;
      raw_imglst_ += ",";
      path_imglst_.push_back(std::string(val));
    }
    if (!strcmp(name, "image_bin")) {
      raw_imgbin_ += val;
      raw_imgbin_ += ",";
      path_imgbin_.push_back(std::string(val));
    }
    if (!strcmp(name, "image_conf_prefix")) {
      img_conf_prefix_ = val;
    }
    if (!strcmp(name, "image_conf_ids")) {
      img_conf_ids_ = val;
    }
    if (!strcmp(name, "dist_num_worker")) {
      dist_num_worker_ = atoi(val);
    }
    if (!strcmp(name, "dist_worker_rank")) {
      dist_worker_rank_ = atoi(val);
    }
    if (!strcmp(name, "silent")) silent_ = atoi(val);
    itrpage.get_factory().SetParam(name, val);
    itrimg.get_factory().SetParam(name, val);
  }
  virtual void Init(void) {
    this->ParseImageConf();
    if (silent_ == 0) {
      if (img_conf_prefix_.length() == 0) {
        printf("ThreadImagePageIterator:image_list=%s, bin=%s\n",
               raw_imglst_.c_str(), raw_imgbin_.c_str());
      } else {
        printf("ThreadImagePageIterator:image_conf=%s, image_ids=%s\n",
               img_conf_prefix_.c_str(), img_conf_ids_.c_str());
      }
    }
    utils::Check(path_imgbin_.size() == path_imglst_.size(),
                 "List/Bin number not consist");
    itrpage.get_factory().path_imgbin = path_imgbin_;
    itrpage.get_factory().path_imglst = path_imglst_;
    itrpage.Init();
    itrimg.get_factory().itrpage = &itrpage;
    this->BeforeFirst();
  }
  virtual void BeforeFirst(void) {
    itrimg.BeforeFirst();
  }
  virtual bool Next(void) {
    if (itrimg.Next(outimg_)) {
      out_.index = outimg_->inst_index;
      out_.label = outimg_->label;
      out_.data = outimg_->img;
      return true;
    } else {
      return false;
    }
  }
  virtual const DataInst &Value(void) const {
    return out_;
  }

 protected:
  /*! \brief number of distributed worker */
  int dist_num_worker_, dist_worker_rank_;
  /*! \brief output data */
  DataInst out_;
  /*! \brief silent */
  int silent_;
  /*! \brief prefix path of image binary, path to input lst */
  // format: imageid label path
  std::vector<std::string> path_imgbin_, path_imglst_;
  /*! \brief configuration bing */
  std::string img_conf_prefix_, img_conf_ids_;
  /*! \brief raw image list */
  std::string raw_imglst_, raw_imgbin_;
  /*! \brief parse configure file */
  inline void ParseImageConf(void) {
    // handling for hadoop
    const char *ps_rank = getenv("PS_RANK");
    if (ps_rank != NULL) {
      this->SetParam("dist_worker_rank", ps_rank);
    }
    if (img_conf_prefix_.length() == 0) return;
    utils::Check(path_imglst_.size() == 0 &&
                 path_imgbin_.size() == 0,
                 "you can either set image_conf_prefix or image_bin/image_list");
    int lb, ub;
    utils::Check(sscanf(img_conf_ids_.c_str(), "%d-%d", &lb, &ub) == 2,
                 "image_conf_ids only support range, like 1-100");
    int n = ub + 1 - lb;
    if (dist_num_worker_ > 1) {
      int step = (n + dist_num_worker_ - 1) / dist_num_worker_;
      int begin = std::min(dist_worker_rank_ * step, n) + lb;
      int end = std::min((dist_worker_rank_ + 1) * step, n) + lb;
      lb = begin; ub = end - 1;
      utils::Check(lb <= ub,
                   "ThreadImagePageIterator: too many workers"\
                   "such that idlist cannot be divided between them");
    }
    for (int i = lb; i <= ub; ++i) {
      std::string tmp;
      tmp.resize(img_conf_prefix_.length() + 30);
      sprintf(&tmp[0], img_conf_prefix_.c_str(), i);
      tmp.resize(strlen(tmp.c_str()));
      path_imglst_.push_back(tmp + ".lst");
      path_imgbin_.push_back(tmp + ".bin");
    }
  }

private:
  // load page data structure
  struct PageEntry {
    utils::BinaryPage page;
    std::vector<float> labels;
    std::vector<unsigned> inst_index;
  };
  // factory to load page
  struct PageFactory {
   public:
    // list of bin path
    std::vector<std::string> path_imgbin;
    // list of img list path
    std::vector<std::string> path_imglst;
    // constructor
    PageFactory(void) {
      label_width = 1;
      list_ptr = 0;
      fplist = NULL;
      shuffle = 0;
      rnd.Seed(kRandMagic); 
    }
    inline void SetParam(const char *name, const char *val) {
      if (!strcmp(name, "label_width")) {
        label_width = atoi(val);
      }
      if (!strcmp(name, "shuffle")) {
        shuffle = atoi(val);
      }
      if (!strcmp(name, "seed_data")) {
        rnd.Seed(atoi(val) + kRandMagic); 
      }
    }
    inline bool Init(void) {
      list_order.resize(path_imgbin.size());
      for (size_t i = 0; i < path_imgbin.size(); ++i) {
        list_order[i] = i;
      }
      if (shuffle != 0) {
        rnd.Shuffle(list_order);
      }
      // load in data
      list_ptr = 0;
      fi.Open(path_imgbin[list_order[0]].c_str(), "rb");
      fplist = utils::FopenCheck(path_imglst[list_order[0]].c_str(), "r");
      return true;
    }
    inline void BeforeFirst(void) {
      list_ptr = 0;
      if (path_imgbin.size() == 1) {
        fi.Seek(0);
        fseek(fplist, 0, SEEK_SET);
      } else {
        if (shuffle != 0) {
          rnd.Shuffle(list_order);
        }
        fi.Close();
        fi.Open(path_imgbin[list_order[0]].c_str(), "rb");
        if (fplist != NULL) fclose(fplist);
        fplist = utils::FopenCheck(path_imglst[list_order[0]].c_str(), "r");
      }
    }
    inline PageEntry *Create(void) {
      return new PageEntry();
    }
    inline bool LoadNext(PageEntry *&a) {
      while (true) {
        if (a->page.Load(fi)) {
          a->labels.resize(a->page.Size() * label_width);
          a->inst_index.resize(a->page.Size());
          for (int i = 0; i < a->page.Size(); ++i) {
            utils::Check(fscanf(fplist, "%u", &(a->inst_index[i])) == 1,
                         "invalid list format");
            for (int j = 0; j < label_width; ++j) {
              utils::Check(fscanf(fplist, "%f", &(a->labels[i * label_width + j])) == 1,
                           "ImageList format:label_width=%u but only have %d labels per line",
                           label_width, j);
              
            }
            utils::Assert(fscanf(fplist, "%*[^\n]\n") == 0, "ignore");
          }          
        } else {
          list_ptr += 1;
          if (list_ptr >= list_order.size()) return false;
          fi.Close();
          fi.Open(path_imgbin[list_order[list_ptr]].c_str(), "rb");
          if (fplist != NULL) fclose(fplist);
          fplist = utils::FopenCheck(path_imglst[list_order[list_ptr]].c_str(), "r");
        }
      }
    }
    inline void FreeSpace(PageEntry *&a) {
      delete a;
    }
    inline void Destroy() {
      fi.Close();
      if (fplist != NULL) fclose(fplist);
    }
   
   private:
    // file stream for binary page
    utils::StdFile fi;
    // seq of list index
    std::vector<size_t> list_order;
    /*! \brief label-width */
    int label_width;
    // pointer for each list
    size_t list_ptr;
    // file ptr for list
    FILE *fplist;
    // shuffle
    int shuffle;
    // random sampler
    utils::RandomSampler rnd;
    // magic seed number for random sampler
    static const int kRandMagic = 121;    
  };  
  // put everything in inst entry
  struct ImageEntry {
    // insance index
    unsigned inst_index;
    // label of each instance
    mshadow::TensorContainer<cpu, 1> label;
    // image data
    mshadow::TensorContainer<cpu, 3> img;
    ImageEntry() : label(false), img(false) {}
  };
  struct ImageFactory {
  public:
    // page iterator
    utils::ThreadBuffer<PageEntry*, PageFactory> *itrpage;
    // constructor
    ImageFactory(void) {
      label_width = 1;
      data_ptr = 0;
      shuffle = 0;
      end_of_data = false;
      page = NULL;
      rnd.Seed(kRandMagic);       
    }
    inline void SetParam(const char *name, const char *val) {
      if (!strcmp(name, "label_width")) {
        label_width = atoi(val);
      }
      if (!strcmp(name, "shuffle")) {
        shuffle = atoi(val);
      }
      if (!strcmp(name, "seed_data")) {
        rnd.Seed(atoi(val) + kRandMagic); 
      }
    }
    inline bool Init(void) {
      return true;
    }
    inline ImageEntry *Create(void) {
      return new ImageEntry();
    }
    inline void FreeSpace(ImageEntry *&a) {
      delete a;
    }
    inline bool LoadNext(ImageEntry *&val) {
      if (end_of_data) return false;
      while (true) {
        if (page == NULL || data_ptr >= page->page.Size()) {
          if (!itrpage->Next(page)) {
            end_of_data = true; return false;
          }
          data_ptr = 0;
          inst_order.resize(page->page.Size());
          for (int i = 0; i < page->page.Size(); ++i) {
            inst_order[i] = i;
          }
          if (shuffle != 0) {
            rnd.Shuffle(inst_order);
          }
        } else {
          const int idx = inst_order[data_ptr];
          utils::BinaryPage::Obj obj = page->page[idx];
          decoder.Decode(static_cast<unsigned char*>(obj.dptr),
                         obj.sz, &img);
          val->img.Resize(mshadow::Shape3(3, img.size(0), img.size(1)));         
          // assign image
          if (img.size(0) == 3) {
            mshadow::Tensor<cpu, 3> dst = val->img;
            for (index_t i = 0; i < img.size(0); ++i) {
              for (index_t j = 0; j < img.size(1); ++j) {
                for (index_t k = 0; k < 3; ++k) {
                  dst[k][i][j] = static_cast<real_t>(img[i][j][k]);                
                }
              }
            }
          } else {
            mshadow::Tensor<cpu, 3> dst = val->img;
            for (index_t i = 0; i < img.size(0); ++i) {
              for (index_t j = 0; j < img.size(1); ++j) {
                real_t s = static_cast<real_t>(img[i][j][0]);
                dst[0][i][j] = s;
                dst[1][i][j] = s;
                dst[2][i][j] = s;
              }
            }
          }
          val->label.Resize(mshadow::Shape1(label_width));
          for (int j = 0; j < label_width; ++j) {
            val->label[j] = page->labels[idx * label_width + j];
          }
          val->inst_index = page->inst_index[idx];
          data_ptr += 1;
          return true;
        }
      }
    }
    inline void Destroy() {}
    inline void BeforeFirst() {
      itrpage->BeforeFirst();
      end_of_data = false;
      page = NULL;
      data_ptr = 0;
    }
   private:
    // mark end of data
    bool end_of_data;
    // current page
    PageEntry *page;    
    // seq of inst index
    std::vector<int> inst_order;
    // jpeg decoder
    utils::JpegDecoder decoder;    
    // id for data
    int data_ptr;
    // shuffle
    int shuffle;
    // label_width
    int label_width;
    // image
    mshadow::TensorContainer<cpu, 3, unsigned char> img;
    // random number generator
    utils::RandomSampler rnd;
    // magic number
    static const int kRandMagic = 111;
  };

protected:
  /*! \brief output data */
  ImageEntry *outimg_;
  utils::ThreadBuffer<PageEntry*, PageFactory> itrpage;
  utils::ThreadBuffer<ImageEntry*, ImageFactory> itrimg;
}; // class ThreadImagePageIterator
}; // namespace cxxnet
#endif
