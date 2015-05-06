/*!
 *  Copyright (c) 2015 by Contributors
 * \file im2rec.cc
 * \brief convert images into image recordio format
 *  Image Record Format: zeropad[64bit] imid[64bit] img-binary-content
 *  The 64bit zero pad was reserved for future purposes
 *
 *  Image List Format: unique-image-index label[s] path-to-image
 * \sa dmlc/recordio.h
 */
#include <cctype>
#include <cstring>
#include <vector>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/timer.h>
#include <dmlc/logging.h>
#include <dmlc/recordio.h>
#include <opencv2/opencv.hpp>
#include "../src/io/image_recordio.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "Usage: <image.lst> <image_root_dir> <output_file> [resize=new_size] [label_width=1]\n");
    return 0;
  }
  int label_width = 1;
  int new_size = -1;
  for (int i = 0; i < argc; ++i) {
    char key[64];
    char val[64];
    if (sscanf(argv[i], "%[^=]=%s", key, val) == 2) {
      if (!strcmp(key, "resize")) new_size = atoi(val);
      if (!strcmp(key, "label_width")) label_width = atoi(val);
    }
  }
  if (new_size > 0) {
    LOG(INFO) << "New Image Size: Short Edge " << new_size;
  } else {
    LOG(INFO) << "Keep origin image size";
  }
  if (argc > 5) label_width = atoi(argv[5]);
  using namespace dmlc;
  const static size_t kBufferSize = 1 << 20UL;
  std::string root = argv[2];
  cxxnet::ImageRecordIO rec;
  size_t imcnt = 0;
  double tstart = dmlc::GetTime();
  dmlc::Stream *flist = dmlc::Stream::Create(argv[1], "r");
  dmlc::istream is(flist);
  dmlc::Stream *fo = dmlc::Stream::Create(argv[3], "w");
  LOG(INFO) << "Output: " << argv[3];
  dmlc::RecordIOWriter writer(fo);
  std::string fname, path, blob;
  std::vector<unsigned char> decode_buf;
  std::vector<unsigned char> encode_buf;
  std::vector<int> encode_params;
  encode_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  encode_params.push_back(80);
  while (is >> rec.header.image_id[0] >> rec.header.label) {
    for (int k = 1; k < label_width; ++ k) {
      float tmp;
      CHECK(is >> tmp)
          << "Invalid ImageList, did you provide the correct label_width?";
    }
    CHECK(std::getline(is, fname));
    const char *p = fname.c_str();
    while (isspace(*p)) ++p;
    path = root + p;
    // use "r" is equal to rb in dmlc::Stream
    dmlc::Stream *fi = dmlc::Stream::Create(path.c_str(), "r");
    rec.SaveHeader(&blob);
    decode_buf.clear();
    size_t imsize = 0;
    while (true) {
      decode_buf.resize(imsize + kBufferSize);
      size_t nread = fi->Read(BeginPtr(decode_buf) + imsize, kBufferSize);
      imsize += nread;
      decode_buf.resize(imsize);
      if (nread != kBufferSize) break;
    }
    delete fi;
    if (new_size > 0) {
      cv::Mat img = cv::imdecode(decode_buf, CV_LOAD_IMAGE_COLOR);
      CHECK(img.data != NULL) << "OpenCV decode fail:" << path;
      cv::Mat res;
      if (img.rows > img.cols) {
        cv::resize(img, res, cv::Size(img.rows * new_size / img.cols, 
                  new_size), 0, 0, CV_INTER_LINEAR);
      } else {
        cv::resize(img, res, cv::Size(new_size, new_size * img.cols
          / img.rows), 0, 0, CV_INTER_LINEAR);
      }
      encode_buf.clear();
      CHECK(cv::imencode(".jpg", res, encode_buf, encode_params));
      size_t bsize = blob.size();
      blob.resize(bsize + encode_buf.size());
      memcpy(BeginPtr(blob) + bsize,
             BeginPtr(encode_buf), encode_buf.size());
    } else {
      size_t bsize = blob.size();
      blob.resize(bsize + decode_buf.size());
      memcpy(BeginPtr(blob) + bsize,
             BeginPtr(decode_buf), decode_buf.size());
    }
    writer.WriteRecord(BeginPtr(blob), blob.size());
    // write header
    ++imcnt;
    if (imcnt % 1000 == 0) {
      LOG(INFO) << imcnt << " images processed, " << GetTime() - tstart << " sec elapsed";
    }
  }
  LOG(INFO) << "Total: " << imcnt << " images processed, " << GetTime() - tstart << " sec elapsed";
  delete fo;
  delete flist;
  return 0;
}
