#include <mshadow/tensor.h>
#include <cstdio>
#include <assert.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: bin2txt binary_file txt_output\n");
        exit(-1);
    }
    FILE *fp = fopen(argv[1], "rb");
    FILE *wp = fopen(argv[2], "w");
    assert(wp != NULL);
    assert(fp != NULL);
    long total = 0;
    mshadow::index_t fea = 0;

    fread(&total, sizeof(long), 1, fp);
    fread(&fea, sizeof(mshadow::index_t), 1, fp);

    mshadow::real_t *buf = new mshadow::real_t[total * fea];
    fread(buf, sizeof(mshadow::real_t), total * fea, fp);
    for (long i = 0; i < total; ++i) {
        for (mshadow::index_t j = 0; j < fea; ++j) {
            fprintf(wp, "%f ", buf[i * fea + j]);
        }
        fprintf(wp, "\n");
    }
    delete [] buf;
    fclose(fp);
    fclose(wp);
}
