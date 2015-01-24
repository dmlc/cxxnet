#ifndef ITER_THREAD_IMBIN_INL_HPP
#define ITER_THREAD_IMBIN_INL_HPP
#pragma once
/*!
 * \file cxxnet_iter_thread_imbin-inl.hpp
 * \brief threaded version of page iterator
 * \author Tianqi Chen
 */
#include "data.h"
#include <opencv2/opencv.hpp>
#include "../utils/thread_buffer.h"
#include "../utils/utils.h"

namespace cxxnet {
/*! \brief thread buffer iterator */
class ThreadImagePageIterator: public IIterator< DataInst > {
public:
  ThreadImagePageIterator(void) {
    idx_ = 0;
    img_.set_pad(false);
    fplst_ = NULL;
    silent_ = 0;
    itr.SetParam("buffer_size", "4");
    page_.page = NULL;
    flag_ = true;
  }
  virtual ~ThreadImagePageIterator(void) {
    if (fplst_ != NULL) fclose(fplst_);
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "image_list")) {
      raw_imglst_ += val;
      raw_imglst_ += ",";
      std::string v(val);
      path_imglst_.push_back(v);
    }
    if (!strcmp(name, "image_bin")) {
      raw_imgbin_ += val;
      raw_imgbin_ += ",";
      std::string v(val);
      path_imgbin_.push_back(v);
    }
    if (!strcmp(name, "image_conf_prefix")) {
      std::string v(val);
      img_conf_prefix_.push_back(v);
    }
    if (!strcmp(name, "idlist")) {
      std::string buf;
      char name[1024];
      int lb = 0;
      int ub = 0;
      utils::Check(sscanf(val, "%d-%d", &lb, &ub) == 2, "idlist only support range, like 1-100");
      for (; lb <= ub; ++lb) {
        for (unsigned int i = 0; i < img_conf_prefix_.size(); ++i) {
          sprintf(name, "%s-%04d", img_conf_prefix_[i].c_str(), lb);
          std::string tmp(name);
          buf = tmp;
          std::string v = buf + ".lst";
          path_imglst_.push_back(v);
          v = buf + ".bin";
          path_imgbin_.push_back(v);
          buf.clear();
        }
      }
    }
    if (!strcmp(name, "silent"))      silent_ = atoi(val);
  }
  virtual void Init(void) {
    fplst_  = utils::FopenCheck(path_imglst_[0].c_str(), "r");
    if (silent_ == 0) {
      printf("ThreadImagePageIterator:image_list=%s, bin=%s\n", raw_imglst_.c_str(), raw_imgbin_.c_str());
    }
    utils::Check(path_imgbin_.size() == path_imglst_.size(), "List/Bin number not consist");
    itr.get_factory().path_imgbin = path_imgbin_;
    itr.get_factory().Ready();
    itr.Init();
    this->BeforeFirst();
  }
  virtual void BeforeFirst(void) {
    if (path_imglst_.size() == 1) {
      fseek(fplst_ , 0, SEEK_SET);

    } else {
      if (fplst_) fclose(fplst_);
      idx_ = 0;
      fplst_  = utils::FopenCheck(path_imglst_[0].c_str(), "r");
    }
    itr.BeforeFirst();
    this->LoadNextPage();
    flag_ = true;
  }
  virtual bool Next(void) {
    if (!flag_) return flag_;
    while (fscanf(fplst_, "%u%f%*[^\n]\n", &out_.index, &out_.label) == 2) {
      this->NextBuffer(buf_);
      this->LoadImage(img_, out_, buf_);
      return true;
    }
    idx_ += 1;
    idx_ %= path_imglst_.size();
    if (idx_ == 0 || path_imglst_.size() == 1) {
      flag_ = false;
      return flag_;
    } else {
      if (fplst_) fclose(fplst_);
      fplst_  = utils::FopenCheck(path_imglst_[idx_].c_str(), "r");
      return Next();
    }
  }
  virtual const DataInst &Value(void) const {
    return out_;
  }
protected:
  inline static void LoadImage(mshadow::TensorContainer<cpu, 3> &img,
                               DataInst &out,
                               std::vector<unsigned char> &buf) {
    cv::Mat res = cv::imdecode(buf, 1);
    utils::Assert(res.data != NULL, "decoding fail");

    img.Resize(mshadow::Shape3(3, res.rows, res.cols));
    for (index_t y = 0; y < img.size(1); ++y) {
      for (index_t x = 0; x < img.size(2); ++x) {
        cv::Vec3b bgr = res.at<cv::Vec3b>(y, x);
        // store in RGB order
        img[2][y][x] = bgr[0];
        img[1][y][x] = bgr[1];
        img[0][y][x] = bgr[2];
      }
    }
    out.data = img;
    // free memory
    res.release();
  }
  inline void NextBuffer(std::vector<unsigned char> &buf) {
    while (ptop_ >= page_.page->Size()) {
      this->LoadNextPage();
    }
    utils::BinaryPage::Obj obj = (*page_.page)[ ptop_ ];
    buf.resize(obj.sz);
    memcpy(&buf[0], obj.dptr, obj.sz);
    ++ ptop_;
  }
  inline void LoadNextPage(void) {
    utils::Assert(itr.Next(page_), "can not get first page");
    ptop_ = 0;
  }
protected:
  bool flag_;
  int idx_;
  // output data
  DataInst out_;
  // silent
  int silent_;
  // file pointer to list file, information file
  FILE *fplst_;
  // prefix path of image binary, path to input lst, format: imageid label path
  std::vector<std::string> path_imgbin_, path_imglst_, img_conf_prefix_;
  std::string raw_imglst_, raw_imgbin_;
  // temp storage for image
  mshadow::TensorContainer<cpu, 3> img_;
  // temp memory buffer
  std::vector<unsigned char> buf_;
private:
  struct PagePtr {
    utils::BinaryPage *page;
  };
  struct Factory {
  public:
    utils::StdFile fi;
    std::vector<std::string> path_imgbin;
    int idx;
    bool flag;
  public:
    Factory() : idx(0), flag(true) {}
    inline bool Init() {
      return true;
    }
    inline void SetParam(const char *name, const char *val) {}
    inline void Ready() {
      fi.Open(path_imgbin[idx].c_str(), "rb");
    }
    inline bool LoadNext(PagePtr &val) {
      if (!flag) return flag;
      bool res = val.page->Load(fi);
      if (res) return res;
      else {
        idx += 1;
        idx %= path_imgbin.size();
        if (idx == 0) {
          flag = false;
          return flag;
        } else {
          fi.Close();
          fi.Open(path_imgbin[idx].c_str(), "rb");
          return val.page->Load(fi);
        }
      }
    }
    inline PagePtr Create(void) {
      PagePtr a; a.page = new utils::BinaryPage();
      return a;
    }
    inline void FreeSpace(PagePtr &a) {
      delete a.page;
    }
    inline void Destroy() {
      fi.Close();
    }
    inline void BeforeFirst() {
      if (path_imgbin.size() == 1) {
        fi.Seek(0);
      } else {
        idx = 0;
        fi.Close();
        fi.Open(path_imgbin[idx].c_str(), "rb");
      }
      flag = true;
    }
  };
protected:
  PagePtr page_;
  int     ptop_;
  utils::ThreadBuffer<PagePtr, Factory> itr;
}; // class ThreadImagePageIterator
}; // namespace cxxnet
#endif

