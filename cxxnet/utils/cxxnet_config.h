#ifndef CXXNET_CONFIG_H
#define CXXNET_CONFIG_H
/*!
 * \file cxxnet_config.h
 * \brief helper class to load in configures from file, adapted from xgboost
 * \author Tianqi Chen
 */
#include <cstdio>
#include <cstring>
#include <string>
#include "cxxnet_utils.h"

namespace cxxnet{
    namespace utils{
        /*! 
         * \brief an iterator that iterates over a configure file and gets the configures
         */
        class ConfigIterator{
        public:
            /*! 
             * \brief constructor 
             * \param fname name of configure file
             */
            ConfigIterator( const char *fname ){
                fi = FopenCheck( fname, "r");
                ch_buf = fgetc( fi );
            }
            /*! \brief destructor */
            ~ConfigIterator(){
                fclose( fi );
            }
            /*! 
             * \brief get current name, called after Next returns true
             * \return current parameter name 
             */
            inline const char *name( void )const{
                return s_name;
            }
            /*! 
             * \brief get current value, called after Next returns true
             * \return current parameter value 
             */
            inline const char *val( void ) const{
                return s_val;
            }
            /*! 
             * \brief move iterator to next position
             * \return true if there is value in next position
             */
            inline bool Next( void ){
                while( !feof( fi ) ){
                    GetNextToken( s_name );
                    if( s_name[0] == '=')  return false;               
                    if( GetNextToken( s_buf ) || s_buf[0] != '=' ) return false;
                    if( GetNextToken( s_val ) || s_val[0] == '=' ) return false;
                    return true;
                }
                return false;
            }
        private:
            FILE *fi;        
            char ch_buf;
            char s_name[100000],s_val[100000],s_buf[100000];
            
            inline void SkipLine(){           
                do{
                    ch_buf = fgetc( fi );
                }while( ch_buf != EOF && ch_buf != '\n' && ch_buf != '\r' );
            }
            
            inline void ParseStr( char tok[] ){
                int i = 0; 
                while( (ch_buf = fgetc(fi)) != EOF ){
                    switch( ch_buf ){
                    case '\\': tok[i++] = fgetc( fi ); break;
                    case '\"': tok[i++] = '\0'; return;
                    case '\r':
                    case '\n': Error("unterminated string"); 
                    default: tok[i++] = ch_buf;
                    }
                }
                Error("unterminated string"); 
            }

            inline void ParseStrML( char tok[] ){
                int i = 0; 
                while( (ch_buf = fgetc(fi)) != EOF ){
                    switch( ch_buf ){
                    case '\\': tok[i++] = fgetc( fi ); break;
                    case '\'': tok[i++] = '\0'; return;
                    default: tok[i++] = ch_buf;
                    }
                }
                Error("unterminated string"); 
            }
            // return newline 
            inline bool GetNextToken( char tok[] ){
                int i = 0;
                bool new_line = false; 
                while( ch_buf != EOF ){
                    switch( ch_buf ){
                    case '#' : SkipLine(); new_line = true; break;
                    case '\"':
                        if( i == 0 ){
                            ParseStr( tok );ch_buf = fgetc(fi); return new_line;
                        }else{
                            Error("token followed directly by string"); 
                        }
                    case '\'':
                        if( i == 0 ){
                            ParseStrML( tok );ch_buf = fgetc(fi); return new_line;
                        }else{
                            Error("token followed directly by string"); 
                        }
                    case '=':
                        if( i == 0 ) {
                            ch_buf = fgetc( fi );     
                            tok[0] = '='; 
                            tok[1] = '\0'; 
                        }else{
                            tok[i] = '\0'; 
                        }
                        return new_line;
                case '\r':
                    case '\n':
					if( i == 0 ) new_line = true;
                    case '\t':
                    case ' ' :
                        ch_buf = fgetc( fi );
                        if( i > 0 ){
                            tok[i] = '\0'; 
                            return new_line;
                        }               
                        break;
                    default: 
                        tok[i++] = ch_buf;
                        ch_buf = fgetc( fi );
                        break;                    
                    }
                }
                return true;
            }
        };
    };
};
#endif
