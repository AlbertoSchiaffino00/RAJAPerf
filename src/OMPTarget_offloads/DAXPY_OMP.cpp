////// HERO_1 includes /////
#ifdef __HERO_1
#include "printf.h"
#include "omp.h"
#include <stdarg.h>

#include <snrt.h>

#include "io.h"
#include "faxpy.hpp"
////// HOST includes /////
#else
extern "C" {
#include <libhero/hero_api.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
}
#include <iostream>
#endif
///// ALL includes /////
extern "C" {
    #include <hero_64.h>
}

#include "DAXPY_OMP.hpp"

///// END includes /////

namespace rajaperf
{
// #pragma omp declare target

//     double x_1[2][DIM_BUFFER/8] __attribute__((section(".l1_0")));
//     double y_1[2][DIM_BUFFER/8] __attribute__((section(".l1_0")));

//     double x_2[2][DIM_BUFFER/8] __attribute__((section(".l1_1")));
//     double y_2[2][DIM_BUFFER/8] __attribute__((section(".l1_1")));


// #pragma omp end declare target



    DAXPY_OMPTarget::DAXPY_OMPTarget(Real_ptr x, Real_ptr y, Real_type a,
                                   Index_type ibegin, Index_type iend,
                                   RepIndex_type run_reps)
        : x((Real_ptr)x),
          y((Real_ptr)y),
          a((Real_type)a),
          ibegin((Index_type)ibegin),
          iend((Index_type)iend),
          run_reps((RepIndex_type)run_reps){}

    void DAXPY_OMPTarget::OMPTarget_initialization(){
        #ifndef __HERO_1
        std::cout << "Initialization" << std::endl;
        #endif
        #pragma omp target device(1)
        {
            asm volatile("nop");
        }

        #ifndef __HERO_1

        x_virt = (double *)hero_dev_l3_malloc(NULL, iend * sizeof(double),  &x_phys);
        y_virt = (double *)hero_dev_l3_malloc(NULL, iend * sizeof(double), &y_phys);

        for(Index_type i = 0; i < iend; i++){  
            x_virt[i] = (double)x[i];
            y_virt[i] = (double)y[i];
        }

        #endif

        #ifndef __HERO_1
        std::cout << "End Initialization" << std::endl;
        #endif
    }

    void DAXPY_OMPTarget::DAXPY_OMP(){

        // TODO: map does not like class variables. 
        uint64_t x_phys_, y_phys_;
        int iend_ = (int)iend;
        x_phys_ = x_phys;
        y_phys_ = y_phys;
        double a_ = (double)a;
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

            #pragma omp target teams device(1) num_teams(2) map (to: x_phys_, y_phys_, a_, iend_) 
            {
                (volatile void) x_phys_;
                (volatile void) y_phys_;
                (volatile void) a_;
                (volatile void) iend_;

                #ifdef __HERO_1
                #pragma omp distribute parallel for 
                {
                    for (int i = 0; i < iend_; i++ ) {
                            double inter = a_ * ((double *)x_phys_)[i];
                            ((double *)y_phys_)[i] = (double) (((double *)y_phys_)[i] + inter);
                    }
                }
                #endif

            }

        }

    }


    void DAXPY_OMPTarget::DAXPY_OMP_opt(){

        // // TODO: map does not like class variables. 
        // uint64_t x_phys_, y_phys_, a_phys_;
        // int iend_ = (int)iend;
        // x_phys_ = x_phys;
        // y_phys_ = y_phys;
        // double a_ = (double)a;

        // for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //     #pragma omp target teams device(1) num_teams(2) map (to: x_phys_, y_phys_, a_, iend_) 
        //     {
        //         (volatile void) x_phys_;
        //         (volatile void) y_phys_;
        //         (volatile void) a_;
        //         (volatile void) iend_;
        //         #ifdef __HERO_1

        //         double *x_locals[2], *y_locals[2],*curr_loc_x[1],*curr_loc_y[1],*dma_loc_x[1], *dma_loc_y[1];//double buffering 
        //         //change to fixed addresses
        //         if(snrt_global_core_idx() == 0){
        //             x_locals[0] = x_1[0];
        //             y_locals[0] = y_1[0];
        //             x_locals[1] = x_1[1];
        //             y_locals[1] = y_1[1];
        //         }else{
        //             x_locals[0] = x_2[0];
        //             y_locals[0] = y_2[0];
        //             x_locals[1] = x_2[1];
        //             y_locals[1] = y_2[1];
        //         }
        //         #pragma omp parallel
        //         {
        //             uint32_t cluster_idx = snrt_cluster_idx();
        //             uint32_t global_core_idx = snrt_global_core_idx();  
        //             uint32_t loc_core_idx = snrt_cluster_core_idx();
        //             //half of the array per cluster
        //             uint32_t tot_size_per_cluster = iend_/2;
        //             //if second cluster has an extra element
        //             if(iend_%2 != 0 && cluster_idx == 1){
        //                 tot_size_per_cluster++;
        //             }
        //             tot_size_per_cluster *= sizeof(double);


        //             //number of iterations needed to fill buffer everytime 
        //             uint32_t iters = (tot_size_per_cluster%DIM_BUFFER == 0) ? tot_size_per_cluster/DIM_BUFFER : tot_size_per_cluster/DIM_BUFFER + 1;

        //             //size of current iteration and size of the next for double buffering
        //             uint32_t curr_size =  (tot_size_per_cluster > DIM_BUFFER) ? DIM_BUFFER : tot_size_per_cluster ;
        //             uint32_t next_curr_size = curr_size;
        //             uint32_t curr_num_elements = curr_size/sizeof(double);
        //             uint32_t offload_cl = (cluster_idx==0) ? 0 : tot_size_per_cluster;
        //             //if array is odd, the second cluster needs to start one element before
        //             if(iend_%2 != 0 && cluster_idx == 1){
        //                 offload_cl-=sizeof(double);
        //             }

        //             if(loc_core_idx ==0){
        //                 snrt_dma_start_1d_wideptr((uint64_t)x_locals[0], (uint64_t) x_phys_+offload_cl,curr_size);
        //                 snrt_dma_start_1d_wideptr((uint64_t)y_locals[0], (uint64_t) y_phys_+offload_cl,curr_size);                    
        //             }

        //             for(uint32_t i=0;i<iters;i++){
        //                 if(i==iters-1){
        //                     //only last iteration has a diff number of elements
        //                     curr_size = next_curr_size;
        //                     curr_num_elements = curr_size/sizeof(double);
        //                 }
        //                 next_curr_size = (i!=iters-2) ? DIM_BUFFER : tot_size_per_cluster - (i+1)*DIM_BUFFER;

        //                 //double buffering pointers curr_* for computaton, dma_* for dma transfers
        //                 curr_loc_x[0]   =   x_locals[i%2];
        //                 curr_loc_y[0]   =   y_locals[i%2];
        //                 dma_loc_x[0]    =   x_locals[(i+1)%2];
        //                 dma_loc_y[0]    =   y_locals[(i+1)%2];

        //                 if(loc_core_idx == 0){
        //                     snrt_dma_wait_all();
        //                     if(i!=iters-1){
        //                         snrt_dma_start_1d_wideptr((uint64_t)dma_loc_x[0], (uint64_t) x_phys_+offload_cl+(i+1)*DIM_BUFFER,next_curr_size);
        //                         snrt_dma_start_1d_wideptr((uint64_t)dma_loc_y[0], (uint64_t) y_phys_+offload_cl+(i+1)*DIM_BUFFER,next_curr_size);  

        //                     }
        //                 }

        //                 snrt_cluster_hw_barrier();

        //                 //divide the work between the two local cores
        //                 uint32_t loc_start = (loc_core_idx == 0) ? 0 : curr_num_elements/2;
        //                 uint32_t loc_end = (loc_core_idx == 0) ?  curr_num_elements/2 : curr_num_elements;  

        //                 // for(uint32_t i=loc_start; i<loc_end; i++){
        //                     // double inter = a_ * ((double *)curr_loc_x[0])[i];
        //                     // ((double *)curr_loc_y[0])[i] = (double) (((double *)curr_loc_y[0])[i] + inter);
        //                 // }
        //                 faxpy_v64b(a_,( __attribute__((address_space(0)))  double*) curr_loc_x[0]+loc_start,( __attribute__((address_space(0))) double*)  curr_loc_y[0]+loc_start, loc_end-loc_start);


        //                 snrt_cluster_hw_barrier();

        //                 //copy back and wait for dma transfers to be completed
        //                 if(loc_core_idx == 0){
        //                     snrt_dma_start_1d_wideptr((uint64_t) (y_phys_ + offload_cl+i*DIM_BUFFER), (uint64_t)curr_loc_y[0], curr_size);
        //                 }
        //             }
        //         }
        //         #endif
        //     }
        // }
    }


    void DAXPY_OMPTarget::OMPTarget_conclusion(){
        #ifndef __HERO_1
        std::cout << "Conclusion" << std::endl;
        for(int i=0; i < iend; i++){
            y[i] = (Real_type)y_virt[i];
        }
        hero_dev_l3_free(NULL,(uintptr_t) x_virt,(uintptr_t) x_phys);
        hero_dev_l3_free(NULL,(uintptr_t) y_virt,(uintptr_t) y_phys);
        std::cout << "End Conclusion" << std::endl;
        #endif
    }

}