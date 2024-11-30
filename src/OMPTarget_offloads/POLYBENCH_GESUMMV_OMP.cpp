///// HERO_1 includes /////
#ifdef __HERO_1
#include "printf.h"
#include "omp.h"
#include <stdarg.h>

#include <snrt.h>

#include "io.h"
#include "dp_gesummv.hpp"
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

#include "POLYBENCH_GESUMMV_OMP.hpp"

///// END includes /////

// #pragma omp declare target

// double A_loc[2][N_BUFFER*K_BUFFER/2] __attribute__((section(".l1")));
// double B_loc[2][N_BUFFER*K_BUFFER/2] __attribute__((section(".l1")));
// double X_loc[2][N_BUFFER] __attribute__((section(".l1")));
// double Y_loc[1][K_BUFFER] __attribute__((section(".l1")))={0};
// #pragma omp end declare target


namespace rajaperf
{

    POLYBENCH_GESUMMV_OMPTarget::POLYBENCH_GESUMMV_OMPTarget(Real_ptr m_A, Real_ptr m_B, Real_ptr m_X, Real_ptr m_Y,
                                    Real_type m_alpha,
                                    Real_type m_beta,
                                    Index_type m_n,
                                    Index_type m_run_reps)
                                    :A((Real_ptr)m_A),
                                    B((Real_ptr)m_B),
                                    X((Real_ptr)m_X),
                                    Y((Real_ptr)m_Y),
                                    alpha((Real_type)m_alpha),
                                    beta((Real_type)m_beta),
                                    N((Index_type)m_n),
                                    run_reps((Index_type)m_run_reps){}
                                    

    void POLYBENCH_GESUMMV_OMPTarget::OMPTarget_initialization(){
        // #ifndef __HERO_1
        // std::cout << "POLYBENCH_GESUMMV_OMPTarget initialization" << std::endl;
        // #endif
        // #pragma omp target device(1)
        // {
        //     asm volatile("nop");
        // }

        // #ifndef __HERO_1

        // A_virt = (double *)hero_dev_l3_malloc(NULL, N * N * sizeof(double), &A_phys);
        // B_virt = (double *)hero_dev_l3_malloc(NULL, N * N * sizeof(double), &B_phys);
        // X_virt = (double *)hero_dev_l3_malloc(NULL, N * sizeof(double), &X_phys);
        // Y_virt = (double *)hero_dev_l3_malloc(NULL, N * sizeof(double), &Y_phys);

        // for(int i=0;i<N;i++){
        //     X_virt[i] = (double)X[i];
        //     Y_virt[i] = (double)Y[i];
        //     for(int j=0;j<N;j++){
        //         A_virt[i*N+j] = (double)A[i*N+j];
        //         B_virt[i*N+j] = (double)B[i*N+j];
        //     }
        // }

        // #endif

        // #ifndef __HERO_1
        // std::cout << "End Initialization" << std::endl;
        // #endif
    }

    void POLYBENCH_GESUMMV_OMPTarget::POLYBENCH_GESUMMV_OMP(){

        // uint64_t A_phys_, B_phys_, X_phys_, Y_phys_;
        // double alpha_, beta_;
        // Index_type N_;

        // A_phys_ = A_phys;
        // B_phys_ = B_phys;
        // X_phys_ = X_phys;
        // Y_phys_ = Y_phys;
        // alpha_ = alpha;
        // beta_ = beta;
        // N_ = N;

        // for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //     #pragma omp target teams device(1) num_teams(2) map (to: A_phys_, B_phys_, X_phys_, Y_phys_, alpha_, beta_, N_)
        //     {
        //         (volatile void) A_phys_;
        //         (volatile void) B_phys_;
        //         (volatile void) X_phys_;
        //         (volatile void) Y_phys_;
        //         (volatile void) alpha_;
        //         (volatile void) beta_;
        //         (volatile void) N_;
                
        //         #ifdef __HERO_1


        //         #pragma omp distribute parallel for
        //         {
        //             for(int i=0; i<N_; i++){
        //                 double partial_1 = 0.0;
        //                 double partial_2 = 0.0;
        //                 for(int j=0; j<N_; j++){
        //                     double par = ((double *)A_phys_)[i*N_ + j]*((double *)X_phys_)[j];
        //                     partial_1 = par + partial_1;
        //                     par = ((double*)B_phys_)[i*N_ + j]*((double*)X_phys_)[j];
        //                     partial_2 = par + partial_2;
        //                 }
        //                 double par = alpha_*partial_1;
        //                 double par2 = beta_*partial_2;
        //                 ((double*)Y_phys_)[i] = par + par2;
        //             }
        //         }
        //         #endif
        //     }
        // }

    }


    void POLYBENCH_GESUMMV_OMPTarget::POLYBENCH_GESUMMV_OMP_opt(){
        // uint64_t A_phys_, B_phys_, X_phys_, Y_phys_;
        // double alpha_, beta_;
        // Index_type N_;

        // A_phys_ = A_phys;
        // B_phys_ = B_phys;
        // X_phys_ = X_phys;
        // Y_phys_ = Y_phys;
        // alpha_ = alpha;
        // beta_ = beta;
        // N_ = N;

        // for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //     #pragma omp target teams device(1) num_teams(2) map (to: A_phys_, B_phys_, X_phys_, Y_phys_, alpha_, beta_, N_)
        //     {
        //         (volatile void) A_phys_;
        //         (volatile void) B_phys_;
        //         (volatile void) X_phys_;
        //         (volatile void) Y_phys_;
        //         (volatile void) alpha_;
        //         (volatile void) beta_;
        //         (volatile void) N_;
                
        //         #ifdef __HERO_1


        //         #pragma omp  parallel 
        //         {
        //             uint32_t cluster_idx = snrt_cluster_idx();
        //             uint32_t global_core_idx = snrt_global_core_idx();
        //             uint32_t cluster_core_idx = snrt_cluster_core_idx();

        //             //tiling over K
        //             uint32_t k_iters = (N_ % 8==0 ) ? N_/8 : N_/8 + 1;
        //             uint32_t dim_k_stride = N_ / k_iters;
        //             uint32_t start_index = dim_k_stride/2 * cluster_idx; // index of each cluster in the I dimension
        //             uint32_t core_size_K = (dim_k_stride/4);
        //             uint32_t start_local_index = (cluster_core_idx==0) ? 0 : core_size_K;

        //             //tiling over N
        //             uint32_t n_iters = (N_ % 512==0 ) ? N_/512 : N_/512 + 1;
        //             uint32_t dim_n_stride = N_ / n_iters;

        //             //double buffering
        //             uint32_t curr_A_B_X =0;
        //             uint32_t curr_Y =0;

        //             //size transfers
        //             const uint32_t size_A_B_transfer = dim_n_stride*sizeof(double);
        //             const uint32_t size_A_B_transfer_last = (N_-dim_n_stride*(n_iters-1))*sizeof(double);

        //             if(cluster_core_idx==0){
        //                     snrt_dma_start_2d_wideptr(  (uint64_t)A_loc[0], //dest
        //                                                 (uint64_t)A_phys_ + start_index*N_*sizeof(double), //src
        //                                                 size_A_B_transfer, //size
        //                                                 size_A_B_transfer, //dst_stride
        //                                                 N_*sizeof(double), //src_stride
        //                                                 dim_k_stride/2 //repeat
        //                     );  
        //                     snrt_dma_start_2d_wideptr(  (uint64_t)B_loc[0], //dest
        //                                                 (uint64_t)B_phys_ + start_index*N_*sizeof(double), //src
        //                                                 size_A_B_transfer, //size
        //                                                 size_A_B_transfer, //dst_stride
        //                                                 N_*sizeof(double), //src_stride
        //                                                 dim_k_stride/2 //repeat
        //                     );
        //                     snrt_dma_start_1d_wideptr((uint64_t)X_loc[0], (uint64_t)X_phys_, size_A_B_transfer);

        //             }
        //             snrt_cluster_hw_barrier();
                    
        //             // uint32_t total_time=0;
        //             for(int k=0;k<k_iters;k++){
        //                 for(int n=0;n<n_iters;n++){
        //                     double *A_curr[1], *A_next[1],*B_curr[1], *B_next[1], *X_curr[1], *X_next[1]; 
        //                     if(curr_A_B_X == 0){
        //                         A_curr[0] = A_loc[0];
        //                         A_next[0] = A_loc[1];
        //                         B_curr[0] = B_loc[0];
        //                         B_next[0] = B_loc[1];
        //                         X_curr[0] = X_loc[0];
        //                         X_next[0] = X_loc[1];
        //                         curr_A_B_X = 1;
        //                     }else{
        //                         A_curr[0] = A_loc[1];
        //                         A_next[0] = A_loc[0];
        //                         B_curr[0] = B_loc[1];
        //                         B_next[0] = B_loc[0];
        //                         X_curr[0] = X_loc[1];
        //                         X_next[0] = X_loc[0];
        //                         curr_A_B_X = 0;
        //                     }   
        //                     if(cluster_core_idx==0){

        //                         snrt_dma_wait_all();

        //                         if(n!=n_iters-1){
        //                             uint32_t curr_transfer = (n==n_iters-2) ? size_A_B_transfer_last : size_A_B_transfer; 
        //                             snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
        //                                                         (uint64_t)A_phys_ + (start_index*N_ + (n+1)*dim_n_stride + k*dim_k_stride*N)*sizeof(double) , //src
        //                                                         curr_transfer, //size
        //                                                         curr_transfer, //dst_stride
        //                                                         N_*sizeof(double), //src_stride
        //                                                         dim_k_stride/2 //repeat
        //                             );  
        //                             snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
        //                                                         (uint64_t)B_phys_ +(start_index*N_ + (n+1)*dim_n_stride + k*dim_k_stride*N)*sizeof(double), //src
        //                                                         curr_transfer, //size
        //                                                         curr_transfer, //dst_stride
        //                                                         N_*sizeof(double), //src_stride
        //                                                         dim_k_stride/2 //repeat
        //                             );
        //                             snrt_dma_start_1d_wideptr((uint64_t)X_next[0], (uint64_t)X_phys_+ (n+1)*dim_n_stride*sizeof(double), curr_transfer);
        //                         }else if(k!=k_iters-1){
        //                             snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
        //                                                         (uint64_t)A_phys_ + (start_index*N_ + (k+1)*dim_k_stride*N_)*sizeof(double) , //src
        //                                                         size_A_B_transfer, //size
        //                                                         size_A_B_transfer, //dst_stride
        //                                                         N_*sizeof(double), //src_stride
        //                                                         dim_k_stride/2 //repeat
        //                             );  
        //                             snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
        //                                                         (uint64_t)B_phys_ + (start_index*N_ + (k+1)*dim_k_stride*N_)*sizeof(double), //src
        //                                                         size_A_B_transfer, //size
        //                                                         size_A_B_transfer, //dst_stride
        //                                                         N_*sizeof(double), //src_stride
        //                                                         dim_k_stride/2 //repeat
        //                             );
        //                             snrt_dma_start_1d_wideptr((uint64_t)X_next[0], (uint64_t)X_phys_, size_A_B_transfer);

        //                         }

        //                     }
        //                     snrt_cluster_hw_barrier();

        //                     bool first_iter = (n==0) ? true : false;
        //                     bool last_iter = (n==n_iters-1) ? true : false;
        //                     if(last_iter) 

        //                         dp_gesummv_2x( (double __attribute__((address_space(1))) *)A_curr[0], (double __attribute__((address_space(1))) *)B_curr[0], 
        //                                         (double __attribute__((address_space(1))) *)X_curr[0], (double __attribute__((address_space(1))) *)Y_loc[0], 
        //                                         start_local_index, start_local_index+2,
        //                                         alpha_, beta_, N_ - n*dim_n_stride, first_iter, true);
        //                     else 
        //                         dp_gesummv_2x( (double __attribute__((address_space(1))) *)A_curr[0], (double __attribute__((address_space(1))) *)B_curr[0], 
        //                                         (double __attribute__((address_space(1))) *)X_curr[0], (double __attribute__((address_space(1))) *)Y_loc[0], 
        //                                         start_local_index, start_local_index+2,
        //                                         alpha_, beta_, dim_n_stride, first_iter, false);                        
        //                     snrt_cluster_hw_barrier();

        //                 }
        //                 //each core produces 2 values, too small for dma. Directly copy it back
        //                 ((double*)Y_phys_)[start_index+start_local_index+k*dim_k_stride] = Y_loc[0][start_local_index];
        //                 ((double*)Y_phys_)[start_index+start_local_index+k*dim_k_stride+1] = Y_loc[0][start_local_index+1];
            
        //             }
        //         }
        //         #endif
        //     }
        // }
        
    }



    void POLYBENCH_GESUMMV_OMPTarget::OMPTarget_conclusion(){
        // #ifndef __HERO_1
        // std::cout << "Conclusion" << std::endl;
        // for(int i=0; i < N; i++){
        //     Y[i] = (Real_type)Y_virt[i];
        // }

        // hero_dev_l3_free(NULL,(uintptr_t) A_virt,(uintptr_t) A_phys);
        // hero_dev_l3_free(NULL,(uintptr_t) B_virt,(uintptr_t) B_phys);
        // hero_dev_l3_free(NULL,(uintptr_t) X_virt,(uintptr_t) X_phys);
        // hero_dev_l3_free(NULL,(uintptr_t) Y_virt,(uintptr_t) Y_phys);


        // std::cout << "End Conclusion" << std::endl;
        // #endif
    }

}