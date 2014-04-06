#include "cxxnet/utils/cxxnet_io_utils.h"
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    using namespace cxxnet::utils;
    if (argc != 4) {
        fprintf(stderr, "Usage: imbin image.lst image_root_dir output_file\n");
        exit(-1);
    }
    char fname[ 256 ];
    unsigned int index = 0;
    float label = 0.0f;
    std::string root_path = argv[2];
    BinaryPage pg;

    StdFile writer(argv[3], "wb");
    std::vector<unsigned char> buf( BinaryPage::kPageSize * sizeof(int), 0 );

    FILE *fplst = FopenCheck(argv[1], "r");
    time_t start = time( NULL );
    long imcnt = 0, pgcnt = 0;
    long elapsed;

    printf( "create image binary pack from %s, this will take some time...\n", argv[1] );

    while( fscanf( fplst,"%u%f %[^\n]\n", &index, &label, fname ) == 3 ) {
        std::string path = fname;
        path = root_path + path;
        StdFile reader(path.c_str(), "rb");
        BinaryPage::Obj fobj(&buf[0], reader.Size());

        if( reader.Size() > buf.size() ){
            fprintf( stderr, "image %s is too large to fit into a single page, considering increase kPageSize\n", path.c_str() );
            Error("image size too large");
        }

        reader.Read(fobj.dptr, fobj.sz);
        reader.Close();

        ++ imcnt;
        if (!pg.Push(fobj)) {
            pg.Save(writer);
            pg.Clear();
            if( !pg.Push(fobj) ){
                fprintf( stderr, "image %s is too large to fit into a single page, considering increase kPageSize\n", path.c_str() );
                Error("image size too large");
            }
            pgcnt += 1;
        }
        if( imcnt % 1000 == 0 ){
            elapsed = (long)(time(NULL) - start);
            printf("\r                                                               \r");
            printf("[%8lu] images processed to %lu pages, %ld sec elapsed", imcnt, pgcnt, elapsed );
            fflush( stdout );
        }
    }
    if( pg.Size() != 0 ){
        pg.Save(writer);
        pgcnt += 1;
    }
    elapsed = (long)(time(NULL) - start);
    printf("\nfinished [%8lu] images processed to %lu pages, %ld sec elapsed\n", imcnt, pgcnt, elapsed );
    writer.Close();
    return 0;
}
