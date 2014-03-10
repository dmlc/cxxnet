#ifndef CXXNET_MNIST_HPP
#define CXXNET_MNIST_HPP
#pragma once
#include "../cxxnet.h"
#include "../cxxnet_data.h"
#include <zlib.h>

namespace cxxnet {
    struct GzFile {
        gzFile fp_;
        GzFile(const char *path, const char *mode) {
            fp_ = gzopen(path, mode);
            Assert(fp != NULL, "Failed to open file\n");
        }
        ~GzFile() { if (fp_) gzclose(fp_); }
        int ReadInt() {
            char buf[4];
            Assert(gzread(fp_, buf, sizeof buf) == sizeof(buf), "Failed to read an int\n");
            return int(buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3]);
        }

        uint8_t ReadByte() {
            uint8_t i;
            Assert(gzread(fp_, &i, sizeof(i)) == sizeof(i), "Failed to read an byte");
            return i;
        }
    };
    template<typename DType>
    class MNIST : public IIterator {
    public:
        MNIST(const char *path_img, const char *path_label, \
              const char *name, const char *val) {
            gzimg_ = GzFile(path_img, "rb");
            gzlabel_ = GzFile(path_label, "rb");
            LoadImage();
            LoadLabel();
            SetParam(name, val);
        }
        void SetParam(const char *name, const char *val) {
            if( !strncmp( name, "batch", strlen(tag) ) ){
                int ltag = strlen("batch");
                if( name[ltag] == ':' ) name += ltag + 1;
            }
            if( !strcmp( name, "batch") )  batch_size_ = (int)atoi(val);
        }
        void Init() {
            loc_ = 0;
        }
        void BeforeFirst() {
            // ?
        }
    // private:
        index_t batch_size_;
        index_t sz_;
        index_t loc_;
        GzFile gzimg_;
        GzFile gzlabel_;
        CTensor4D img_;
        CTensor1D label_;
    // private:
        void LoadImage() {
            int image_magic = gzimg_.read_int();
            int image_count = gzimg_.read_int();
            int image_rows = gzimg_.read_int();
            int image_cols = gzimg_.read_int();
            img_ = mshadow::NewCTensor(mshadow::Shape4(1, image_rows, image_cols, img_count) , 0);
            for (int i = 0; i < image_count; ++i) {
                for (int j = 0; j < image_rows; ++j) {
                    for (int k = 0; k < image_cols; ++k) {
                        img_[0][k][j][i] = gzlabel_.read_byte();
                    }
                }
            }
        }

        void LoadLabel() {
            int label_magic = gzlabel_.read_int();
            int label_count = gzlabel_.read_int();
            label_ = mshadow::NewCTensor(mshadow::Shape1(label_count) , 0);
            for (int i = 0; i < label_count; ++i) {
                label_[i] = gzlabel_.read_byte();
            }
        }

    }; //class MNIST
}; // namespace cxxnet
#endif // CXXNET_MNIST_HPP
