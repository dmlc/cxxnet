#ifndef CXXNET_THREAD_H
#define CXXNET_THREAD_H
#pragma once
/*!
 * \file cxxnet_thread.h
 * \brief  this header include the necessary resource for multi-threading
 * \author Tianqi Chen
 * Acknowledgement: this file is adapted from SVDFeature project, by same author. 
 *  The MAC support part of this code is provided by Artemy Kolchinsky
 */

#ifdef _MSC_VER

#include "cxxnet_utils.h"
#include <windows.h>
#include <process.h>

namespace cxxnet{
    namespace utils{
        class Semaphore{
        public :
            inline void Init( int init_val ){
                sem = CreateSemaphore( NULL, init_val, 10, NULL );
                utils::Assert( sem != NULL, "create Semaphore error");
            }
            inline void Destroy(){
                CloseHandle( sem );
            }        
            inline void Wait(){
                utils::Assert( WaitForSingleObject( sem, INFINITE ) == WAIT_OBJECT_0, "WaitForSingleObject error" );
            }
            inline void Post(){
                utils::Assert( ReleaseSemaphore( sem, 1, NULL )  != 0, "ReleaseSemaphore error");
            }
        private:
            HANDLE sem;
        };
        
        class Thread{
        private:
            HANDLE    thread_handle;                
            unsigned  thread_id;                
        public :
            inline void Start( unsigned int __stdcall entry( void* ), void *param ){
                thread_handle = (HANDLE)_beginthreadex( NULL, 0, entry, param, 0, &thread_id );
            }            
            inline int Join(){
                WaitForSingleObject( thread_handle, INFINITE );                                 
                return 0;
            }        
        };        
        inline void ThreadExit( void *status ){
            _endthreadex(0);
        }
    };
};    
#define CXXNET_THREAD_PREFIX unsigned int __stdcall 

#else

// thread interface using g++     
#include <semaphore.h>
#include <pthread.h>

namespace cxxnet{     
    namespace utils{
        /*!\brief semaphore class */
        class Semaphore{
            #ifdef __APPLE__
        private:
            sem_t* semPtr;
            char sema_name[20];            
        private:
            inline void GenRandomString(char *s, const int len) {
                static const char alphanum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" ;
                for (int i = 0; i < len; ++i) {
                    s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
                }
                s[len] = 0;
            }
        public:
            inline void Init( int init_val ){
                sema_name[0]='/'; 
                sema_name[1]='s'; 
                sema_name[2]='e'; 
                sema_name[3]='/'; 
                GenRandomString(&sema_name[4], 16);
                if((semPtr = sem_open(sema_name, O_CREAT, 0644, init_val)) == SEM_FAILED) {
                    perror("sem_open");
                    exit(1);
                }
                utils::Assert( semPtr != NULL, "create Semaphore error");
            }
            inline void Destroy(){
                if (sem_close(semPtr) == -1) {
                    perror("sem_close");
                    exit(EXIT_FAILURE);
                }
                if (sem_unlink(sema_name) == -1) {
                    perror("sem_unlink");
                    exit(EXIT_FAILURE);
                }
            }
            inline void Wait(){
                sem_wait( semPtr );
            }
            inline void Post(){
                sem_post( semPtr );
            }                   
            #else
        private:
            sem_t sem;
        public:
            inline void Init( int init_val ){
                sem_init( &sem, 0, init_val );
            }
            inline void Destroy(){
                sem_destroy( &sem );
            }
            inline void Wait(){
                sem_wait( &sem );
            }
            inline void Post(){
                sem_post( &sem );
            }       
            #endif
            
        };
        /*!\brief simple thread class */
        class Thread{
        private:
            pthread_t thread;                
        public :
            inline void Start( void * entry( void* ), void *param ){
                pthread_attr_t attr;
                pthread_attr_init( &attr );
                pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );
                pthread_create( &thread, &attr, entry, param );
            }        
            
            inline int Join(){
                void *status;
                return pthread_join( thread, &status );
            }
        };    
        
        inline void ThreadExit( void *status ){
            pthread_exit( status );
        }
    };
};

#define CXXNET_THREAD_PREFIX void *
#endif

#endif
