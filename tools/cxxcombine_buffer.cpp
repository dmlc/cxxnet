/*!
 * a tool to combine different set of features into binary buffer
 * not well organized code, but does it's job
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>

#include "../cxxnet/io/cxxnet_data.h"
#include "../cxxnet/utils/cxxnet_utils.h"
#include "../cxxnet/utils/cxxnet_io_utils.h"
#include "../cxxnet/io/cxxnet_iter_sparse-inl.hpp"
using namespace cxxnet;

// header in dataset
struct Header{
    FILE *fi;
    int   tmp_num;
    int   base;
    int   num_feat;
    // whether it's dense format
    bool  is_dense;
	bool  warned;
    
	Header( void ){ this->warned = false; this->is_dense = false; }

	inline void CheckBase( unsigned findex ){
		if( findex >= (unsigned)num_feat && ! warned ) {
			fprintf( stderr, "warning:some feature exceed bound, num_feat=%d\n", num_feat );
			warned = true;
		}
	}
};

inline int norm( std::vector<Header> &vec, int base = 0 ){
    int n = base;
    for( size_t i = 0; i < vec.size(); i ++ ){
        if( vec[i].is_dense ) vec[i].num_feat = 1;
        vec[i].base = n; n += vec[i].num_feat;
    }
    return n;        
}

inline void vclose( std::vector<Header> &vec ){
    for( size_t i = 0; i < vec.size(); i ++ ){
        fclose( vec[i].fi );
    }
}

inline int readnum( std::vector<Header> &vec ){
    int n = 0;
    for( size_t i = 0; i < vec.size(); i ++ ){
        if( !vec[i].is_dense ){
            utils::Assert( fscanf( vec[i].fi, "%d", &vec[i].tmp_num ) == 1, "load num" );
            n += vec[i].tmp_num;
        }else{
            n ++;
        }
    }
    return n;        
}

inline void vskip( std::vector<Header> &vec ){
    for( size_t i = 0; i < vec.size(); i ++ ){
        if( !vec[i].is_dense ){
            utils::Assert( fscanf( vec[i].fi, "%*d%*[^\n]\n" ) >= 0 );
        }else{
            utils::Assert( fscanf( vec[i].fi, "%*f\n" ) >= 0 );
        }
    }
}



class DataLoader{
public:
    // whether to do node and edge feature renormalization
    int rescale;
    int linelimit;
public:
    FILE *fp, *fwlist, *fo;
    std::vector<Header> fheader;
    utils::BinaryPage page;
    
    DataLoader( void ){
        rescale = 0; 
        linelimit = -1; 
        fo = NULL;
        fp = NULL; fwlist = NULL; 
        page.Clear();
    }
private:
    inline void Load( std::vector<SparseInst::Entry> &data, std::vector<Header> &vec ){
        unsigned fidx; float fv;
        for( size_t i = 0; i < vec.size(); i ++ ){
            if( !vec[i].is_dense ) { 
                for( int j = 0; j < vec[i].tmp_num; j ++ ){
                    utils::Assert( fscanf ( vec[i].fi, "%u:%f", &fidx, &fv ) == 2, "Error when load feat" );  
                    vec[i].CheckBase( fidx );
                    fidx += vec[i].base;
                    data.push_back( SparseInst::Entry( fidx, fv ) );
                }
            }else{
                utils::Assert( fscanf ( vec[i].fi, "%f", &fv ) == 1, "load feat" );  
                fidx = vec[i].base;
                data.push_back( SparseInst::Entry( fidx, fv ) );
            }
        }
    }
    inline void DoRescale( std::vector<SparseInst::Entry> &vec ){
        double sum = 0.0;
        for( size_t i = 0; i < vec.size(); i ++ ){
            sum += vec[i].fvalue * vec[i].fvalue;
        } 
        sum = sqrt( sum );
        for( size_t i = 0; i < vec.size(); i ++ ){
            vec[i].fvalue /= sum;
        } 
    }    
public:    
    // basically we are loading all the data inside
    inline void Load( void ){
        float label;
        long lcnt = 0;
        while( fscanf( fp, "%f", &label ) == 1 ){            
            int pass = 1;
            if( fwlist != NULL ){
                utils::Assert( fscanf( fwlist, "%u", &pass ) ==1 );
            }
            if( pass == 0 ){
                vskip( fheader ); 
            }else{            
                const int nfeat = readnum( fheader );
                std::vector<SparseInst::Entry> data;
                // pairs 
                this->Load( data, fheader );
                utils::Assert( data.size() == (unsigned)nfeat );
                if( rescale != 0 ) this->DoRescale( data );
                this->AddRow( data, label, lcnt++ );
            }             
            // linelimit
            if( linelimit >= 0 ) {
                if( -- linelimit <= 0 ) break;
            }
        }        
        if( page.Size() != 0 ){
            mshadow::utils::FileStream fs( fo );
            page.Save( fs );            
        }
    }
    inline void AddRow( const std::vector<SparseInst::Entry> &data, float label, long lcnt ){
        SparseInst inst;
        inst.label = label;
        inst.index = (unsigned)lcnt;
        inst.length = data.size();
        inst.data = &data[0];
        SparseInstObj obj(inst);

        if( !page.Push(obj.GetObj()) ){
            mshadow::utils::FileStream fs( fo );
            page.Save( fs );
            page.Clear();
            utils::Assert( page.Push(obj.GetObj()), "too large sparse vector" );
        }
        obj.FreeSpace();
    }
};

const char *folder = "features";

int main( int argc, char *argv[] ){
    if( argc < 3 ){
        printf("Usage:xgcombine_buffer <inname> <outname> [options] -f [features] -fd [densefeatures]\n"\
               "options: -rescale -linelimit -fgroup <groupfilename> -wlist <whitelistinstance>\n");
        return 0; 
    }

    DataLoader loader;
    time_t start = time( NULL );

    int mode = 0;
    for( int i = 3; i < argc; i ++ ){        
        if( !strcmp( argv[i], "-f") ){
            mode = 0; continue;
        }
        if( !strcmp( argv[i], "-fd") ){
            mode = 2; continue;
        }
        if( !strcmp( argv[i], "-rescale") ){
            loader.rescale = 1; continue;
        }
        if( !strcmp( argv[i], "-wlist") ){
            loader.fwlist = utils::FopenCheck( argv[ ++i ], "r" ); continue;
        }
        if( !strcmp( argv[i], "-linelimit") ){
            loader.linelimit = atoi( argv[ ++i ] ); continue;
        }
       
        char name[ 256 ];
        sprintf( name, "%s/%s.%s", folder, argv[1], argv[i] );
        Header h;
        h.fi = utils::FopenCheck( name, "r" );
        if( mode == 2 ){
            h.is_dense = true; h.num_feat = 1;
            loader.fheader.push_back( h );
        }else{
            utils::Assert( fscanf( h.fi, "%d", &h.num_feat ) == 1, "num feat" );
            switch( mode ){
            case 0: loader.fheader.push_back( h ); break;
            default: ;
            }             
        }
    }
    
    loader.fp = utils::FopenCheck( argv[1], "r" );
    loader.fo = utils::FopenCheck( argv[2], "wb" );
    printf("num_features=%d\n", norm( loader.fheader ) ); 
    printf("start creating buffer...\n");
    loader.Load();
    // close files
    fclose( loader.fp ); fclose( loader.fo );
    if( loader.fwlist != NULL ) fclose( loader.fwlist );    
    vclose( loader.fheader );
    printf("all generation end, %lu sec used\n", (unsigned long)(time(NULL) - start) );    
    return 0;
}
