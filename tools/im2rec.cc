/*!
 *  Copyright (c) 2015 by Contributors
 * \file im2rec.cc
 * \brief convert images into image recordio format
 *  Image Record Format: zeropad[64bit] imid[64bit] img-binary-content
 *  The 64bit zero pad was reserved for future purposes
 *
 *  Image List Format: unique-image-index label path-to-image
 * \sa dmlc/recordio.h
 */
#include <cctype>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/timer.h>
#include <dmlc/logging.h>
#include <dmlc/recordio.h>
#include "../src/io/image_recordio.h"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Usage: <image.lst> <image_root_dir> <output_file>\n");
    return 0;
  }
  using namespace dmlc;
  const static size_t kBufferSize = 1 << 20UL;
  std::string root = argv[2];
  cxxnet::ImageRecordIO rec;

  size_t imcnt = 0;
  double tstart = dmlc::GetTime();  
  dmlc::Stream *flist = dmlc::Stream::Create(argv[1], "r");
  dmlc::istream is(flist);
  dmlc::Stream *fo = dmlc::Stream::Create(argv[3], "w");
  dmlc::RecordIOWriter writer(fo);
  std::string fname, path, blob;
  while (is >> rec.header.image_id[0] >> rec.header.label) {
    CHECK(std::getline(is, fname));
    const char *p = fname.c_str();
    while (isspace(*p)) ++p;
    path = root + p;
    dmlc::Stream *fi = dmlc::Stream::Create(path.c_str(), "r");
    rec.SaveHeader(&blob);
    size_t size = blob.length();
    while (true) {
      blob.resize(size + kBufferSize);
      size_t nread = fi->Read(BeginPtr(blob) + size, kBufferSize);
      size += nread;
      if (nread != kBufferSize)  break;
    }
    delete fi;
    writer.WriteRecord(BeginPtr(blob), size);
    // write header
    ++imcnt;
    if (imcnt % 1000 == 0) {
      LOG(INFO) << imcnt << " images processed, " << GetTime() - tstart << " sec elapsed"; 
    }
  }
  delete fo;
  delete flist;
  return 0;
}
