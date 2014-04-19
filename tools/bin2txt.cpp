#include <mshadow/tensor.h>
#include <cstdio>

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: bin2txt binary_file txt_output\n");
        exit(-1);
    }
    FILE *fp = fopen(argv[1], "rb");
    mshadow::utils::FileStream fs(fp);
    FILE *wp = fopen(argv[2], "w");
    while (!feof(fp)) {
        mshadow::Tensor<mshadow::cpu, 4> t;
        mshadow::LoadBinary(fs, t, false);
        mshadow::Tensor<mshadow::cpu, 2> data = t.FlatTo2D();
        for (mshadow::index_t i = 0; i < data.shape[1]; ++i) {
            for (mshadow::index_t j = 0; j < data.shape[0] - 1; ++j) {
                fprintf(wp, "%f,", data[i][j]);
            }
            fprintf(wp, "%f\n", data[i][data.shape[0] - 1]);
        }
        mshadow::FreeSpace(data);
    }
    fclose(fp);
    fclose(wp);
    // Incorrect assert =,=, but result seems right
}
