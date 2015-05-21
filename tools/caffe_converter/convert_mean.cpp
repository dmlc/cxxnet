/*!
 * \file convert_mean.cpp
 * \brief convert caffe mean file to cxx bin
 * \author Zehua Huang, Naiyan Wang
 */

#include <mshadow/tensor.h>
#include <utils/io.h>
#include <caffe/blob.hpp>
#include <caffe/util/io.hpp>

void ConvertMean(const char *caffe_file_path, const char *cxx_file_path){
  // read caffe mean file
  caffe::BlobProto blob_proto;
  caffe::ReadProtoFromBinaryFileOrDie(caffe_file_path, &blob_proto);
  caffe::Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);

  mshadow::TensorContainer<mshadow::cpu, 3> img;
  mshadow::Shape<3> shape = mshadow::Shape3(3, mean_blob.height(), mean_blob.width());
  img.Resize(shape);

  for (size_t y = 0; y < img.size(1); ++y) {
    for (size_t x = 0; x < img.size(2); ++x) {
      // store in BGR order
      img[2][y][x] = mean_blob.data_at(0, 0, y, x);
      img[1][y][x] = mean_blob.data_at(0, 1, y, x);
      img[0][y][x] = mean_blob.data_at(0, 2, y, x);
    }
  }

  cxxnet::utils::StdFile fo(cxx_file_path, "wb");
  img.SaveBinary(fo);
}

int main(int argc, char *argv[]){
  if (argc != 3) {
    printf("usage: <caffe_mean_path> <cxx_mean_path>\n");
    return 0;
  }

  ConvertMean(argv[1], argv[2]);
  return 0;
}
