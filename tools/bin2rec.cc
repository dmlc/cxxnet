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
#include <string>
#include <cstring>
#include <vector>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/timer.h>
#include <dmlc/logging.h>
#include <dmlc/recordio.h>
#include <opencv2/opencv.hpp>
#include "../src/io/image_recordio.h"
#include "../src/utils/io.h"


int main(int argc, char **argv) {
  using namespace cxxnet::utils;
  using namespace dmlc;
  if (argc < 4) {
    printf("usage: bin2rec img_list bin_file rec_file [label_width=1]\n");
    exit(-1);
  }
  FILE *fplst = fopen(argv[1], "r");
  CHECK(fplst != NULL);
  dmlc::Stream *fo = dmlc::Stream::Create(argv[3], "w");
  dmlc::RecordIOWriter writer(fo);
  cxxnet::ImageRecordIO rec;
  std::string blob, fname;
  StdFile fi;
  fi.Open(argv[2], "rb");
  int label_width = 1;
  if (argc > 4) {
    label_width = atoi(argv[4]);
  }
  BinaryPage pg;
  size_t imcnt = 0;
  while (pg.Load(fi)) {
    for (int i = 0; i < pg.Size(); ++i) {
      CHECK(fscanf(fplst, "%lu", &rec.header.image_id[0]) == 1);
      CHECK(fscanf(fplst, "%f", &rec.header.label) == 1);
      for (int k = 1; k < label_width; ++k) {
        float tmp;
        CHECK(fscanf(fplst, "%f", &tmp) == 1);
      }
      CHECK(fscanf(fplst, "%*[^\n]\n") == 0) << "ignore";
      rec.SaveHeader(&blob);
      BinaryPage::Obj obj = pg[i];
      size_t bsize = blob.size();
      blob.resize(bsize + obj.sz);
      memcpy(BeginPtr(blob) + bsize, obj.dptr, obj.sz);
      writer.WriteRecord(BeginPtr(blob), blob.size());
      imcnt++;
    }
  }
  LOG(INFO) << "Total: " << imcnt << " images processed";
  delete fo;
  fclose(fplst);
}




