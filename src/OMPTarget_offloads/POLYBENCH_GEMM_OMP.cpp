////// HERO_1 includes /////
#ifdef __HERO_1
#include "printf.h"
#include "omp.h"
#include <stdarg.h>

#include <snrt.h>

#include "io.h"
#include "dp_gemm.hpp"
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

#include "POLYBENCH_GEMM_OMP.hpp"

///// END includes /////
#pragma omp declare target

double C_loc[2][DIM_BUFFER/8] __attribute__((section(".l1")));
double A_loc[2][DIM_BUFFER/8] __attribute__((section(".l1")));
double B_loc[2][DIM_BUFFER/8] __attribute__((section(".l1")));

#pragma omp end declare target


namespace rajaperf
{

    POLYBENCH_GEMM_OMPTarget::POLYBENCH_GEMM_OMPTarget(Real_ptr m_A, Real_ptr m_B, Real_ptr m_C,
                                    Real_type m_alpha,
                                    Real_type m_beta,
                                    Index_type m_ni,
                                    Index_type m_nj,
                                    Index_type m_nk,
                                    Index_type m_run_reps)
                                    :A((Real_ptr)m_A),
                                    B((Real_ptr)m_B),
                                    C((Real_ptr)m_C),
                                    alpha((Real_type)m_alpha),
                                    beta((Real_type)m_beta),
                                    NI((Index_type)m_ni),
                                    NJ((Index_type)m_nj),
                                    NK((Index_type)m_nk),
                                    run_reps((Index_type)m_run_reps){}
                                    

    void POLYBENCH_GEMM_OMPTarget::OMPTarget_initialization(){
        #ifndef __HERO_1
        std::cout << "POLYBENCH_GEMM_OMPTarget initialization" << std::endl;
        #endif
        #pragma omp target device(1)
        {
            asm volatile("nop");
        }

        #ifndef __HERO_1


        A_virt = (double *)hero_dev_l3_malloc(NULL, NI * NK * sizeof(double),  &A_phys);
        B_virt = (double *)hero_dev_l3_malloc(NULL, NK * NJ * sizeof(double), &B_phys);
        C_virt = (double *)hero_dev_l3_malloc(NULL, NI * NJ * sizeof(double), &C_phys);

        for(Index_type i=0;i<NI;i++){
            for(Index_type k=0;k<NK;k++){
                A_virt[i*NK+k] = (double)A[i*NK+k];
            }
        }

        for(Index_type k=0;k<NK;k++){
            for(Index_type j=0;j<NJ;j++){
                B_virt[k*NJ+j] = (double)B[k*NJ+j];
            }
        }

        for(Index_type i=0;i<NI;i++){
            for(Index_type j=0;j<NJ;j++){
                C_virt[i*NJ+j] = (double)C[i*NJ+j];
            }
        }

        #endif

        #ifndef __HERO_1
        std::cout << "End Initialization" << std::endl;
        #endif
    }

    void POLYBENCH_GEMM_OMPTarget::POLYBENCH_GEMM_OMP(){

        // // TODO: map does not like class variables. 
        uint64_t A_phys_, B_phys_, C_phys_;
        double alpha_ = alpha;
        double beta_ = beta;
        A_phys_ = A_phys;
        B_phys_ = B_phys;
        C_phys_ = C_phys;
        Index_type NI_ = NI;
        Index_type NJ_ = NJ;
        Index_type NK_ = NK;

        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

            
            #pragma omp target teams device(1) num_teams(2) map (to: A_phys_, B_phys_, C_phys_, alpha_, beta_, NI_, NJ_, NK_)
            {
                (volatile void) A_phys_;
                (volatile void) B_phys_;
                (volatile void) C_phys_;
                (volatile void) alpha_;
                (volatile void) beta_;
                (volatile void) NI_;
                (volatile void) NJ_;
                (volatile void) NK_;

                #ifdef __HERO_1

                #pragma omp distribute parallel for
                {
                    for(int i=0; i<NI_; i++){
                        for(int j=0; j<NJ_; j++){
                            double par2 = beta_*((double *)C_phys_)[i*NJ_ + j];
                            double partial = 0.0;
                            for(int k=0; k<NK_; k++){
                                double par = ((double *)A_phys_)[i*NK_ + k]*((double*)B_phys_)[k*NJ_ + j];
                                partial = par + partial;
                            }
                            double par = alpha_*partial;
                            ((double *)C_phys_)[i*NJ_ + j] = par + par2;
                        }
                    }
                }
                #endif


            }
                
        }

    }


    void POLYBENCH_GEMM_OMPTarget::POLYBENCH_GEMM_OMP_opt(){

        // TODO: map does not like class variables. 
        uint64_t A_phys_, B_phys_, C_phys_;
        double alpha_ = alpha;
        double beta_ = beta;
        A_phys_ = A_phys;
        B_phys_ = B_phys;
        C_phys_ = C_phys;
        Index_type NI_ = NI;
        Index_type NJ_ = NJ;
        Index_type NK_ = NK;

        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

            
            #pragma omp target teams device(1) num_teams(2) map (to: A_phys_, B_phys_, C_phys_, alpha_, beta_, NI_, NJ_, NK_)
            {
                (volatile void) A_phys_;
                (volatile void) B_phys_;
                (volatile void) C_phys_;
                (volatile void) alpha_;
                (volatile void) beta_;
                (volatile void) NI_;
                (volatile void) NJ_;
                (volatile void) NK_;

                #ifdef __HERO_1
                #pragma omp parallel
                {
                    //TODO:fix tiling with some actual computation
                    uint32_t I_tiling = (NI_%32==0) ? NI_/32 : NI_/32 + 1;
                    uint32_t J_tiling = (NJ_%32==0) ? NJ_/32 : NJ_/32 + 1;
                    uint32_t K_tiling = (NK_%64==0) ? NK_/64 : NK_/64 + 1;

                    //TODO: missing assert
                    uint32_t start_parall = read_csr(mcycle);


                    uint32_t cluster_idx = snrt_cluster_idx();
                    uint32_t global_core_idx = snrt_global_core_idx();
                    uint32_t cluster_core_idx = snrt_cluster_core_idx();
                    
                    //repetitions over k
                    uint32_t k_iters = K_tiling;
                    uint32_t dim_k_stride = NK_ / k_iters;

                    //repetitions over i
                    uint32_t i_iters = I_tiling;
                    uint32_t dim_i_stride = NI_ / i_iters;
                    uint32_t start_index_I = dim_i_stride/2 * cluster_idx; // index of each cluster in the I dimension
                    uint32_t core_size_I = (dim_i_stride/4);
                    uint32_t loc_start_index_i = (cluster_core_idx==0) ? 0 : (core_size_I);

                    //repetitions over j
                    uint32_t j_iters = J_tiling;
                    uint32_t dim_j_stride = NJ_ / j_iters;



                    //double buffering
                    uint32_t curr_A_B =0;
                    uint32_t curr_C =0;

                    //values of C dma copy
                    uint32_t size_C_transfer = dim_j_stride*sizeof(double);

                    //values of A dma copy
                    uint32_t size_A_transfer = NK_*sizeof(double)/k_iters;

                    //values of B dma copy
                    uint32_t size_B_transfer = dim_j_stride*sizeof(double);

                    //first copy of the double buffering
                    if(cluster_core_idx==0){
                        snrt_dma_start_2d_wideptr(  (uint64_t)C_loc[0], //dest
                                                (uint64_t)C_phys_ + start_index_I*NJ_*sizeof(double), //src
                                                size_C_transfer, //size
                                                size_C_transfer, //dst_stride
                                                NJ_*sizeof(double), //src_stride
                                                (dim_i_stride/2) //repeat
                        );

                        snrt_dma_start_2d_wideptr(  (uint64_t)A_loc[0], //dest
                                                    (uint64_t)A_phys_ + start_index_I*NK_*sizeof(double), //src
                                                    size_A_transfer, //size
                                                    size_A_transfer, //dst_stride
                                                    NK_*sizeof(double), //src_stride
                                                    (dim_i_stride/2) //repeat
                        );

                        snrt_dma_start_2d_wideptr(  (uint64_t)B_loc[0], //dest
                                                    (uint64_t)B_phys_, //src
                                                    size_B_transfer, //size
                                                    size_B_transfer, //dst_stride
                                                    NJ_*sizeof(double), //src_stride
                                                    dim_k_stride //repeat
                        );   

                        snrt_dma_wait_all();   
                    
                    }

                    snrt_cluster_hw_barrier();

                    for(int i=0;i<i_iters;i++){
                        for(int j=0;j<j_iters;j++){

                            double *C_curr[1]; 
                            double *C_next[1]; 
                            if(curr_C == 0){
                                C_curr[0] = C_loc[0];
                                C_next[0] = C_loc[1];
                                curr_C = 1;
                            }else{
                                C_curr[0] = C_loc[1];
                                C_next[0] = C_loc[0];
                                curr_C = 0;
                            }

                            
                            if(cluster_core_idx==0){ 
                                // uint32_t start_inner_dma_req = read_csr(mcycle);
                                if(j!=j_iters-1){
                                    snrt_dma_start_2d_wideptr(  (uint64_t)C_next[0], //dest
                                                        (uint64_t)C_phys_ + (start_index_I*NJ_ + i*dim_i_stride*NJ_ + (j+1)*dim_j_stride)*sizeof(double), //src
                                                        size_C_transfer, //size
                                                        size_C_transfer, //dst_stride
                                                        NJ_*sizeof(double), //src_stride
                                                        (dim_i_stride/2) //repeat
                                    );
                                    // total_inner_dma_req++;

                                }else if(i!=i_iters-1){
                                    snrt_dma_start_2d_wideptr(  (uint64_t)C_next[0], //dest
                                                            (uint64_t)C_phys_ + (start_index_I*NJ_ + (i+1)*dim_i_stride*NJ_)*sizeof(double), //src
                                                            size_C_transfer, //size
                                                            size_C_transfer, //dst_stride
                                                            NJ_*sizeof(double), //src_stride
                                                            (dim_i_stride/2) //repeat
                                    );
                                }

                            }

                            snrt_cluster_hw_barrier();  


                            for(int k=0;k<k_iters;k++){
                        
                                double *A_curr[1];  
                                double *A_next[1];
                                double *B_curr[1]; 
                                double *B_next[1];

                                if(curr_A_B == 0){
                                    A_curr[0] = A_loc[0];
                                    A_next[0] = A_loc[1];
                                    B_curr[0] = B_loc[0];
                                    B_next[0] = B_loc[1];
                                    curr_A_B = 1;
                                }else{
                                    A_curr[0] = A_loc[1];
                                    A_next[0] = A_loc[0];
                                    B_curr[0] = B_loc[1];
                                    B_next[0] = B_loc[0];
                                    curr_A_B = 0;
                                }

                                if(cluster_core_idx==0){
                                    if(k!=0 || k_iters==1){
                                        snrt_dma_wait_all();
                                    }

                                    if(k!=k_iters-1){
                                        snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
                                                                    (uint64_t)A_phys_ + (start_index_I*NK_ + (k+1)*dim_k_stride + i*dim_i_stride*NK_)*sizeof(double), //src
                                                                    size_A_transfer, //size
                                                                    size_A_transfer, //dst_stride
                                                                    NK_*sizeof(double), //src_stride
                                                                    (dim_i_stride/2) //repeat
                                        );

                                    }else if(j!=j_iters-1){
                                        snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
                                                                    (uint64_t)A_phys_ + (start_index_I*NK_ + i*dim_i_stride*NK_)*sizeof(double), //src
                                                                    size_A_transfer, //size
                                                                    size_A_transfer, //dst_stride
                                                                    NK_*sizeof(double), //src_stride
                                                                    (dim_i_stride/2) //repeat
                                        );
                                    }else if(i!=i_iters-1){
                                        snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
                                                                    (uint64_t)A_phys_ + (start_index_I*NK_ + (i+1)*dim_i_stride*NK_)*sizeof(double), //src
                                                                    size_A_transfer, //size
                                                                    size_A_transfer, //dst_stride
                                                                    NK_*sizeof(double), //src_stride
                                                                    (dim_i_stride/2) //repeat
                                        );
                                    }

                                    if(k!=k_iters-1){
                                    
                                        snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
                                                                    (uint64_t)B_phys_ + ((k+1)*dim_k_stride*NJ_ + j*dim_j_stride)*sizeof(double), //src
                                                                    size_B_transfer, //size
                                                                    size_B_transfer, //dst_stride
                                                                    NJ_*sizeof(double), //src_stride
                                                                    dim_k_stride //repeat
                                        );    
                                        // total_inner_dma_req++;

                                    }else if(j!=j_iters-1){

                                        snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
                                                                    (uint64_t)B_phys_ +  (j+1)*dim_j_stride*sizeof(double), //src
                                                                    size_B_transfer, //size
                                                                    size_B_transfer, //dst_stride
                                                                    NJ_*sizeof(double), //src_stride
                                                                    dim_k_stride //repeat
                                        );  
                                        // total_inner_dma_req++;

                                    }else if(i!=i_iters-1){
                                        snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
                                                                    (uint64_t)B_phys_, //src
                                                                    size_B_transfer, //size
                                                                    size_B_transfer, //dst_stride
                                                                    NJ_*sizeof(double), //src_stride
                                                                    dim_k_stride //repeat
                                        );  

                                    }
                                    
                                }

                                snrt_cluster_hw_barrier();

                                uint32_t first_iter = (k==0) ? 1 : 0;

                                gemm_4xVL((double __attribute__((address_space(1))) *)C_curr[0], (double __attribute__((address_space(1))) *)A_curr[0], (double __attribute__((address_space(1))) *)B_curr[0], 
                                                    loc_start_index_i, loc_start_index_i+core_size_I, dim_k_stride, dim_j_stride, alpha_, beta_,first_iter);
                                

                                snrt_cluster_hw_barrier();


                            }
                            if(cluster_core_idx==0){
                                    
                                snrt_dma_start_2d_wideptr( (uint64_t)C_phys_ + (start_index_I*NJ_ + i*dim_i_stride*NJ_ + j*dim_j_stride)*sizeof(double), //dest
                                                            (uint64_t)C_curr[0], //src
                                                            size_C_transfer, //size
                                                            NJ_*sizeof(double), //dst_stride
                                                            size_C_transfer, //src_stride
                                                            (dim_i_stride/2) //repeat

                                );
                            }
                        }
                    }

                    if(cluster_core_idx==0){
                        snrt_dma_wait_all();  
                    }

                    uint32_t end_parall = read_csr(mcycle);

                    // if(snrt_global_core_idx()==0) snrt_printf("[%u], Total Time %u\n\r",end_parall-start_parall);
                    snrt_cluster_hw_barrier();

                }
                     
                #endif
            }
        }

    }

    void POLYBENCH_GEMM_OMPTarget::POLYBENCH_GEMM_OMP_opt_one_core(){

        // // TODO: map does not like class variables. 
        uint64_t A_phys_, B_phys_, C_phys_;
        double alpha_ = alpha;
        double beta_ = beta;
        A_phys_ = A_phys;
        B_phys_ = B_phys;
        C_phys_ = C_phys;
        Index_type NI_ = NI;
        Index_type NJ_ = NJ;
        Index_type NK_ = NK;

        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

            
            #pragma omp target teams device(1) num_teams(2) map (to: A_phys_, B_phys_, C_phys_, alpha_, beta_, NI_, NJ_, NK_)
            {
                (volatile void) A_phys_;
                (volatile void) B_phys_;
                (volatile void) C_phys_;
                (volatile void) alpha_;
                (volatile void) beta_;
                (volatile void) NI_;
                (volatile void) NJ_;
                (volatile void) NK_;

                #ifdef __HERO_1

                //TODO:fix tiling with some actual computation
                uint32_t I_tiling = (NI_%32==0) ? NI_/32 : NI_/32 + 1;
                uint32_t J_tiling = (NJ_%32==0) ? NJ_/32 : NJ_/32 + 1;
                uint32_t K_tiling = (NK_%64==0) ? NK_/64 : NK_/64 + 1;

                //TODO: missing assert
                uint32_t start_parall = read_csr(mcycle);


                uint32_t cluster_idx = snrt_cluster_idx();
                uint32_t global_core_idx = snrt_global_core_idx();
                
                //repetitions over k
                uint32_t k_iters = K_tiling;
                uint32_t dim_k_stride = NK_ / k_iters;

                //repetitions over i
                uint32_t i_iters = I_tiling;
                uint32_t dim_i_stride = NI_ / i_iters;
                uint32_t start_index_I = dim_i_stride/2 * cluster_idx; // index of each cluster in the I dimension

                //repetitions over j
                uint32_t j_iters = J_tiling;
                uint32_t dim_j_stride = NJ_ / j_iters;



                //double buffering
                uint32_t curr_A_B =0;
                uint32_t curr_C =0;

                //values of C dma copy
                uint32_t size_C_transfer = dim_j_stride*sizeof(double);

                //values of A dma copy
                uint32_t size_A_transfer = NK_*sizeof(double)/k_iters;

                //values of B dma copy
                uint32_t size_B_transfer = dim_j_stride*sizeof(double);

                //first copy of the double buffering
                snrt_dma_start_2d_wideptr(  (uint64_t)C_loc[0], //dest
                                        (uint64_t)C_phys_ + start_index_I*NJ_*sizeof(double), //src
                                        size_C_transfer, //size
                                        size_C_transfer, //dst_stride
                                        NJ_*sizeof(double), //src_stride
                                        (dim_i_stride/2) //repeat
                );

                snrt_dma_start_2d_wideptr(  (uint64_t)A_loc[0], //dest
                                            (uint64_t)A_phys_ + start_index_I*NK_*sizeof(double), //src
                                            size_A_transfer, //size
                                            size_A_transfer, //dst_stride
                                            NK_*sizeof(double), //src_stride
                                            (dim_i_stride/2) //repeat
                );

                snrt_dma_start_2d_wideptr(  (uint64_t)B_loc[0], //dest
                                            (uint64_t)B_phys_, //src
                                            size_B_transfer, //size
                                            size_B_transfer, //dst_stride
                                            NJ_*sizeof(double), //src_stride
                                            dim_k_stride //repeat
                );   

                snrt_dma_wait_all();   
                
                
                for(int i=0;i<i_iters;i++){
                    for(int j=0;j<j_iters;j++){

                        double *C_curr[1]; 
                        double *C_next[1]; 
                        if(curr_C == 0){
                            C_curr[0] = C_loc[0];
                            C_next[0] = C_loc[1];
                            curr_C = 1;
                        }else{
                            C_curr[0] = C_loc[1];
                            C_next[0] = C_loc[0];
                            curr_C = 0;
                        }

                        
                        // uint32_t start_inner_dma_req = read_csr(mcycle);
                        if(j!=j_iters-1){
                            snrt_dma_start_2d_wideptr(  (uint64_t)C_next[0], //dest
                                                (uint64_t)C_phys_ + (start_index_I*NJ_ + i*dim_i_stride*NJ_ + (j+1)*dim_j_stride)*sizeof(double), //src
                                                size_C_transfer, //size
                                                size_C_transfer, //dst_stride
                                                NJ_*sizeof(double), //src_stride
                                                (dim_i_stride/2) //repeat
                            );
                            // total_inner_dma_req++;

                        }else if(i!=i_iters-1){
                            snrt_dma_start_2d_wideptr(  (uint64_t)C_next[0], //dest
                                                    (uint64_t)C_phys_ + (start_index_I*NJ_ + (i+1)*dim_i_stride*NJ_)*sizeof(double), //src
                                                    size_C_transfer, //size
                                                    size_C_transfer, //dst_stride
                                                    NJ_*sizeof(double), //src_stride
                                                    (dim_i_stride/2) //repeat
                            );
                        }



                        for(int k=0;k<k_iters;k++){
                    
                            double *A_curr[1];  
                            double *A_next[1];
                            double *B_curr[1]; 
                            double *B_next[1];

                            if(curr_A_B == 0){
                                A_curr[0] = A_loc[0];
                                A_next[0] = A_loc[1];
                                B_curr[0] = B_loc[0];
                                B_next[0] = B_loc[1];
                                curr_A_B = 1;
                            }else{
                                A_curr[0] = A_loc[1];
                                A_next[0] = A_loc[0];
                                B_curr[0] = B_loc[1];
                                B_next[0] = B_loc[0];
                                curr_A_B = 0;
                            }

                            if(k!=0 || k_iters==1){
                                snrt_dma_wait_all();
                            }

                            if(k!=k_iters-1){
                                snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
                                                            (uint64_t)A_phys_ + (start_index_I*NK_ + (k+1)*dim_k_stride + i*dim_i_stride*NK_)*sizeof(double), //src
                                                            size_A_transfer, //size
                                                            size_A_transfer, //dst_stride
                                                            NK_*sizeof(double), //src_stride
                                                            (dim_i_stride/2) //repeat
                                );

                            }else if(j!=j_iters-1){
                                snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
                                                            (uint64_t)A_phys_ + (start_index_I*NK_ + i*dim_i_stride*NK_)*sizeof(double), //src
                                                            size_A_transfer, //size
                                                            size_A_transfer, //dst_stride
                                                            NK_*sizeof(double), //src_stride
                                                            (dim_i_stride/2) //repeat
                                );
                            }else if(i!=i_iters-1){
                                snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
                                                            (uint64_t)A_phys_ + (start_index_I*NK_ + (i+1)*dim_i_stride*NK_)*sizeof(double), //src
                                                            size_A_transfer, //size
                                                            size_A_transfer, //dst_stride
                                                            NK_*sizeof(double), //src_stride
                                                            (dim_i_stride/2) //repeat
                                );
                            }

                            if(k!=k_iters-1){
                            
                                snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
                                                            (uint64_t)B_phys_ + ((k+1)*dim_k_stride*NJ_ + j*dim_j_stride)*sizeof(double), //src
                                                            size_B_transfer, //size
                                                            size_B_transfer, //dst_stride
                                                            NJ_*sizeof(double), //src_stride
                                                            dim_k_stride //repeat
                                );    
                                // total_inner_dma_req++;

                            }else if(j!=j_iters-1){

                                snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
                                                            (uint64_t)B_phys_ +  (j+1)*dim_j_stride*sizeof(double), //src
                                                            size_B_transfer, //size
                                                            size_B_transfer, //dst_stride
                                                            NJ_*sizeof(double), //src_stride
                                                            dim_k_stride //repeat
                                );  
                                // total_inner_dma_req++;

                            }else if(i!=i_iters-1){
                                snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
                                                            (uint64_t)B_phys_, //src
                                                            size_B_transfer, //size
                                                            size_B_transfer, //dst_stride
                                                            NJ_*sizeof(double), //src_stride
                                                            dim_k_stride //repeat
                                );  

                            }
                                

                            uint32_t first_iter = (k==0) ? 1 : 0;

                            gemm_4xVL((double __attribute__((address_space(1))) *)C_curr[0], (double __attribute__((address_space(1))) *)A_curr[0], (double __attribute__((address_space(1))) *)B_curr[0], 
                                                0, dim_i_stride/2, dim_k_stride, dim_j_stride, alpha_, beta_,first_iter);
                            

                        }
                                
                        snrt_dma_start_2d_wideptr( (uint64_t)C_phys_ + (start_index_I*NJ_ + i*dim_i_stride*NJ_ + j*dim_j_stride)*sizeof(double), //dest
                                                    (uint64_t)C_curr[0], //src
                                                    size_C_transfer, //size
                                                    NJ_*sizeof(double), //dst_stride
                                                    size_C_transfer, //src_stride
                                                    (dim_i_stride/2) //repeat

                        );
                    }
                }

                snrt_dma_wait_all();  

                uint32_t end_parall = read_csr(mcycle);

                // if(snrt_global_core_idx()==0) snrt_printf("[%u], Total Time %u\n\r",end_parall-start_parall);
                     
                #endif
            }
        }

    }


    void POLYBENCH_GEMM_OMPTarget::POLYBENCH_GEMM_OMP_opt_one_team(){

        // TODO: map does not like class variables. 
        uint64_t A_phys_, B_phys_, C_phys_;
        double alpha_ = alpha;
        double beta_ = beta;
        A_phys_ = A_phys;
        B_phys_ = B_phys;
        C_phys_ = C_phys;
        Index_type NI_ = NI;
        Index_type NJ_ = NJ;
        Index_type NK_ = NK;

        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

            
            #pragma omp target teams device(1) num_teams(1) map (to: A_phys_, B_phys_, C_phys_, alpha_, beta_, NI_, NJ_, NK_)
            {
                (volatile void) A_phys_;
                (volatile void) B_phys_;
                (volatile void) C_phys_;
                (volatile void) alpha_;
                (volatile void) beta_;
                (volatile void) NI_;
                (volatile void) NJ_;
                (volatile void) NK_;

                #ifdef __HERO_1
                #pragma omp parallel
                {
                    //TODO:fix tiling with some actual computation
                    uint32_t I_tiling = (NI_%32==0) ? NI_/32 : NI_/32 + 1;
                    uint32_t J_tiling = (NJ_%32==0) ? NJ_/32 : NJ_/32 + 1;
                    uint32_t K_tiling = (NK_%64==0) ? NK_/64 : NK_/64 + 1;

                    //TODO: missing assert
                    uint32_t start_parall = read_csr(mcycle);


                    uint32_t cluster_idx = snrt_cluster_idx();
                    uint32_t global_core_idx = snrt_global_core_idx();
                    uint32_t cluster_core_idx = snrt_cluster_core_idx();
                    
                    //repetitions over k
                    uint32_t k_iters = K_tiling;
                    uint32_t dim_k_stride = NK_ / k_iters;

                    //repetitions over i
                    uint32_t i_iters = I_tiling;
                    uint32_t dim_i_stride = NI_ / i_iters;
                    // uint32_t start_index_I = dim_i_stride/2 * cluster_idx; // index of each cluster in the I dimension
                    uint32_t core_size_I = (dim_i_stride/2);
                    uint32_t loc_start_index_i = (cluster_core_idx==0) ? 0 : (core_size_I);


                    //repetitions over j
                    uint32_t j_iters = J_tiling;
                    uint32_t dim_j_stride = NJ_ / j_iters;



                    //double buffering
                    uint32_t curr_A_B =0;
                    uint32_t curr_C =0;

                    //values of C dma copy
                    uint32_t size_C_transfer = dim_j_stride*sizeof(double);

                    //values of A dma copy
                    uint32_t size_A_transfer = NK_*sizeof(double)/k_iters;

                    //values of B dma copy
                    uint32_t size_B_transfer = dim_j_stride*sizeof(double);

                    //first copy of the double buffering
                    if(cluster_core_idx==0){
                        snrt_dma_start_2d_wideptr(  (uint64_t)C_loc[0], //dest
                                                (uint64_t)C_phys_, //src
                                                size_C_transfer, //size
                                                size_C_transfer, //dst_stride
                                                NJ_*sizeof(double), //src_stride
                                                (dim_i_stride) //repeat
                        );

                        snrt_dma_start_2d_wideptr(  (uint64_t)A_loc[0], //dest
                                                    (uint64_t)A_phys_ , //src
                                                    size_A_transfer, //size
                                                    size_A_transfer, //dst_stride
                                                    NK_*sizeof(double), //src_stride
                                                    (dim_i_stride) //repeat
                        );

                        snrt_dma_start_2d_wideptr(  (uint64_t)B_loc[0], //dest
                                                    (uint64_t)B_phys_, //src
                                                    size_B_transfer, //size
                                                    size_B_transfer, //dst_stride
                                                    NJ_*sizeof(double), //src_stride
                                                    dim_k_stride //repeat
                        );   

                        snrt_dma_wait_all();   
                    
                    }

                    snrt_cluster_hw_barrier();

                    for(int i=0;i<i_iters;i++){
                        for(int j=0;j<j_iters;j++){

                            double *C_curr[1]; 
                            double *C_next[1]; 
                            if(curr_C == 0){
                                C_curr[0] = C_loc[0];
                                C_next[0] = C_loc[1];
                                curr_C = 1;
                            }else{
                                C_curr[0] = C_loc[1];
                                C_next[0] = C_loc[0];
                                curr_C = 0;
                            }

                            
                            if(cluster_core_idx==0){ 
                                // uint32_t start_inner_dma_req = read_csr(mcycle);
                                if(j!=j_iters-1){
                                    snrt_dma_start_2d_wideptr(  (uint64_t)C_next[0], //dest
                                                        (uint64_t)C_phys_ + (i*dim_i_stride*NJ_ + (j+1)*dim_j_stride)*sizeof(double), //src
                                                        size_C_transfer, //size
                                                        size_C_transfer, //dst_stride
                                                        NJ_*sizeof(double), //src_stride
                                                        (dim_i_stride) //repeat
                                    );
                                    // total_inner_dma_req++;

                                }else if(i!=i_iters-1){
                                    snrt_dma_start_2d_wideptr(  (uint64_t)C_next[0], //dest
                                                            (uint64_t)C_phys_ + ((i+1)*dim_i_stride*NJ_)*sizeof(double), //src
                                                            size_C_transfer, //size
                                                            size_C_transfer, //dst_stride
                                                            NJ_*sizeof(double), //src_stride
                                                            (dim_i_stride) //repeat
                                    );
                                }

                            }

                            snrt_cluster_hw_barrier();  


                            for(int k=0;k<k_iters;k++){
                        
                                double *A_curr[1];  
                                double *A_next[1];
                                double *B_curr[1]; 
                                double *B_next[1];

                                if(curr_A_B == 0){
                                    A_curr[0] = A_loc[0];
                                    A_next[0] = A_loc[1];
                                    B_curr[0] = B_loc[0];
                                    B_next[0] = B_loc[1];
                                    curr_A_B = 1;
                                }else{
                                    A_curr[0] = A_loc[1];
                                    A_next[0] = A_loc[0];
                                    B_curr[0] = B_loc[1];
                                    B_next[0] = B_loc[0];
                                    curr_A_B = 0;
                                }

                                if(cluster_core_idx==0){
                                    if(k!=0 || k_iters==1){
                                        snrt_dma_wait_all();
                                    }

                                    if(k!=k_iters-1){
                                        snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
                                                                    (uint64_t)A_phys_ + ((k+1)*dim_k_stride + i*dim_i_stride*NK_)*sizeof(double), //src
                                                                    size_A_transfer, //size
                                                                    size_A_transfer, //dst_stride
                                                                    NK_*sizeof(double), //src_stride
                                                                    (dim_i_stride) //repeat
                                        );

                                    }else if(j!=j_iters-1){
                                        snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
                                                                    (uint64_t)A_phys_ + (i*dim_i_stride*NK_)*sizeof(double), //src
                                                                    size_A_transfer, //size
                                                                    size_A_transfer, //dst_stride
                                                                    NK_*sizeof(double), //src_stride
                                                                    (dim_i_stride) //repeat
                                        );
                                    }else if(i!=i_iters-1){
                                        snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
                                                                    (uint64_t)A_phys_ + ((i+1)*dim_i_stride*NK_)*sizeof(double), //src
                                                                    size_A_transfer, //size
                                                                    size_A_transfer, //dst_stride
                                                                    NK_*sizeof(double), //src_stride
                                                                    (dim_i_stride) //repeat
                                        );
                                    }

                                    if(k!=k_iters-1){
                                    
                                        snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
                                                                    (uint64_t)B_phys_ + ((k+1)*dim_k_stride*NJ_ + j*dim_j_stride)*sizeof(double), //src
                                                                    size_B_transfer, //size
                                                                    size_B_transfer, //dst_stride
                                                                    NJ_*sizeof(double), //src_stride
                                                                    dim_k_stride //repeat
                                        );    
                                        // total_inner_dma_req++;

                                    }else if(j!=j_iters-1){

                                        snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
                                                                    (uint64_t)B_phys_ +  (j+1)*dim_j_stride*sizeof(double), //src
                                                                    size_B_transfer, //size
                                                                    size_B_transfer, //dst_stride
                                                                    NJ_*sizeof(double), //src_stride
                                                                    dim_k_stride //repeat
                                        );  
                                        // total_inner_dma_req++;

                                    }else if(i!=i_iters-1){
                                        snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
                                                                    (uint64_t)B_phys_, //src
                                                                    size_B_transfer, //size
                                                                    size_B_transfer, //dst_stride
                                                                    NJ_*sizeof(double), //src_stride
                                                                    dim_k_stride //repeat
                                        );  

                                    }
                                    
                                }

                                snrt_cluster_hw_barrier();

                                uint32_t first_iter = (k==0) ? 1 : 0;

                                gemm_4xVL((double __attribute__((address_space(1))) *)C_curr[0], (double __attribute__((address_space(1))) *)A_curr[0], (double __attribute__((address_space(1))) *)B_curr[0], 
                                                    loc_start_index_i, loc_start_index_i + core_size_I, dim_k_stride, dim_j_stride, alpha_, beta_,first_iter);
                                

                                snrt_cluster_hw_barrier();


                            }
                            if(cluster_core_idx==0){
                                    
                                snrt_dma_start_2d_wideptr( (uint64_t)C_phys_ + (i*dim_i_stride*NJ_ + j*dim_j_stride)*sizeof(double), //dest
                                                            (uint64_t)C_curr[0], //src
                                                            size_C_transfer, //size
                                                            NJ_*sizeof(double), //dst_stride
                                                            size_C_transfer, //src_stride
                                                            (dim_i_stride) //repeat

                                );
                            }
                        }
                    }

                    if(cluster_core_idx==0){
                        snrt_dma_wait_all();  
                    }

                    uint32_t end_parall = read_csr(mcycle);

                    // if(snrt_global_core_idx()==0) snrt_printf("[%u], Total Time %u\n\r",end_parall-start_parall);
                    snrt_cluster_hw_barrier();

                }
                     
                #endif
            }
        }

    }

void POLYBENCH_GEMM_OMPTarget::POLYBENCH_GEMM_OMP_opt_one_team_one_core(){

        // TODO: map does not like class variables. 
        uint64_t A_phys_, B_phys_, C_phys_;
        double alpha_ = alpha;
        double beta_ = beta;
        A_phys_ = A_phys;
        B_phys_ = B_phys;
        C_phys_ = C_phys;
        Index_type NI_ = NI;
        Index_type NJ_ = NJ;
        Index_type NK_ = NK;

        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

            
            #pragma omp target teams device(1) num_teams(1) map (to: A_phys_, B_phys_, C_phys_, alpha_, beta_, NI_, NJ_, NK_)
            {
                (volatile void) A_phys_;
                (volatile void) B_phys_;
                (volatile void) C_phys_;
                (volatile void) alpha_;
                (volatile void) beta_;
                (volatile void) NI_;
                (volatile void) NJ_;
                (volatile void) NK_;

                #ifdef __HERO_1

                //TODO:fix tiling with some actual computation
                uint32_t I_tiling = (NI_%32==0) ? NI_/32 : NI_/32 + 1;
                uint32_t J_tiling = (NJ_%32==0) ? NJ_/32 : NJ_/32 + 1;
                uint32_t K_tiling = (NK_%64==0) ? NK_/64 : NK_/64 + 1;

                //TODO: missing assert
                uint32_t start_parall = read_csr(mcycle);

                
                //repetitions over k
                uint32_t k_iters = K_tiling;
                uint32_t dim_k_stride = NK_ / k_iters;

                //repetitions over i
                uint32_t i_iters = I_tiling;
                uint32_t dim_i_stride = NI_ / i_iters;


                //repetitions over j
                uint32_t j_iters = J_tiling;
                uint32_t dim_j_stride = NJ_ / j_iters;



                //double buffering
                uint32_t curr_A_B =0;
                uint32_t curr_C =0;

                //values of C dma copy
                uint32_t size_C_transfer = dim_j_stride*sizeof(double);

                //values of A dma copy
                uint32_t size_A_transfer = NK_*sizeof(double)/k_iters;

                //values of B dma copy
                uint32_t size_B_transfer = dim_j_stride*sizeof(double);

                //first copy of the double buffering
                snrt_dma_start_2d_wideptr(  (uint64_t)C_loc[0], //dest
                                        (uint64_t)C_phys_, //src
                                        size_C_transfer, //size
                                        size_C_transfer, //dst_stride
                                        NJ_*sizeof(double), //src_stride
                                        (dim_i_stride) //repeat
                );

                snrt_dma_start_2d_wideptr(  (uint64_t)A_loc[0], //dest
                                            (uint64_t)A_phys_ , //src
                                            size_A_transfer, //size
                                            size_A_transfer, //dst_stride
                                            NK_*sizeof(double), //src_stride
                                            (dim_i_stride) //repeat
                );

                snrt_dma_start_2d_wideptr(  (uint64_t)B_loc[0], //dest
                                            (uint64_t)B_phys_, //src
                                            size_B_transfer, //size
                                            size_B_transfer, //dst_stride
                                            NJ_*sizeof(double), //src_stride
                                            dim_k_stride //repeat
                );   

                snrt_dma_wait_all();   
            
                for(int i=0;i<i_iters;i++){
                    for(int j=0;j<j_iters;j++){

                        double *C_curr[1]; 
                        double *C_next[1]; 
                        if(curr_C == 0){
                            C_curr[0] = C_loc[0];
                            C_next[0] = C_loc[1];
                            curr_C = 1;
                        }else{
                            C_curr[0] = C_loc[1];
                            C_next[0] = C_loc[0];
                            curr_C = 0;
                        }

                        
                        // uint32_t start_inner_dma_req = read_csr(mcycle);
                        if(j!=j_iters-1){
                            snrt_dma_start_2d_wideptr(  (uint64_t)C_next[0], //dest
                                                (uint64_t)C_phys_ + (i*dim_i_stride*NJ_ + (j+1)*dim_j_stride)*sizeof(double), //src
                                                size_C_transfer, //size
                                                size_C_transfer, //dst_stride
                                                NJ_*sizeof(double), //src_stride
                                                (dim_i_stride) //repeat
                            );
                            // total_inner_dma_req++;

                        }else if(i!=i_iters-1){
                            snrt_dma_start_2d_wideptr(  (uint64_t)C_next[0], //dest
                                                    (uint64_t)C_phys_ + ((i+1)*dim_i_stride*NJ_)*sizeof(double), //src
                                                    size_C_transfer, //size
                                                    size_C_transfer, //dst_stride
                                                    NJ_*sizeof(double), //src_stride
                                                    (dim_i_stride) //repeat
                            );
                        }

                        for(int k=0;k<k_iters;k++){
                    
                            double *A_curr[1];  
                            double *A_next[1];
                            double *B_curr[1]; 
                            double *B_next[1];

                            if(curr_A_B == 0){
                                A_curr[0] = A_loc[0];
                                A_next[0] = A_loc[1];
                                B_curr[0] = B_loc[0];
                                B_next[0] = B_loc[1];
                                curr_A_B = 1;
                            }else{
                                A_curr[0] = A_loc[1];
                                A_next[0] = A_loc[0];
                                B_curr[0] = B_loc[1];
                                B_next[0] = B_loc[0];
                                curr_A_B = 0;
                            }

                            if(k!=0 || k_iters==1){
                                snrt_dma_wait_all();
                            }

                            if(k!=k_iters-1){
                                snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
                                                            (uint64_t)A_phys_ + ((k+1)*dim_k_stride + i*dim_i_stride*NK_)*sizeof(double), //src
                                                            size_A_transfer, //size
                                                            size_A_transfer, //dst_stride
                                                            NK_*sizeof(double), //src_stride
                                                            (dim_i_stride) //repeat
                                );

                            }else if(j!=j_iters-1){
                                snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
                                                            (uint64_t)A_phys_ + (i*dim_i_stride*NK_)*sizeof(double), //src
                                                            size_A_transfer, //size
                                                            size_A_transfer, //dst_stride
                                                            NK_*sizeof(double), //src_stride
                                                            (dim_i_stride) //repeat
                                );
                            }else if(i!=i_iters-1){
                                snrt_dma_start_2d_wideptr(  (uint64_t)A_next[0], //dest
                                                            (uint64_t)A_phys_ + ((i+1)*dim_i_stride*NK_)*sizeof(double), //src
                                                            size_A_transfer, //size
                                                            size_A_transfer, //dst_stride
                                                            NK_*sizeof(double), //src_stride
                                                            (dim_i_stride) //repeat
                                );
                            }

                            if(k!=k_iters-1){
                            
                                snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
                                                            (uint64_t)B_phys_ + ((k+1)*dim_k_stride*NJ_ + j*dim_j_stride)*sizeof(double), //src
                                                            size_B_transfer, //size
                                                            size_B_transfer, //dst_stride
                                                            NJ_*sizeof(double), //src_stride
                                                            dim_k_stride //repeat
                                );    
                                // total_inner_dma_req++;

                            }else if(j!=j_iters-1){

                                snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
                                                            (uint64_t)B_phys_ +  (j+1)*dim_j_stride*sizeof(double), //src
                                                            size_B_transfer, //size
                                                            size_B_transfer, //dst_stride
                                                            NJ_*sizeof(double), //src_stride
                                                            dim_k_stride //repeat
                                );  
                                // total_inner_dma_req++;

                            }else if(i!=i_iters-1){
                                snrt_dma_start_2d_wideptr(  (uint64_t)B_next[0], //dest
                                                            (uint64_t)B_phys_, //src
                                                            size_B_transfer, //size
                                                            size_B_transfer, //dst_stride
                                                            NJ_*sizeof(double), //src_stride
                                                            dim_k_stride //repeat
                                );  

                            }

                            uint32_t first_iter = (k==0) ? 1 : 0;

                            gemm_4xVL((double __attribute__((address_space(1))) *)C_curr[0], (double __attribute__((address_space(1))) *)A_curr[0], (double __attribute__((address_space(1))) *)B_curr[0], 
                                                0, dim_i_stride, dim_k_stride, dim_j_stride, alpha_, beta_,first_iter);
                            
                        }
                            
                        snrt_dma_start_2d_wideptr( (uint64_t)C_phys_ + (i*dim_i_stride*NJ_ + j*dim_j_stride)*sizeof(double), //dest
                                                    (uint64_t)C_curr[0], //src
                                                    size_C_transfer, //size
                                                    NJ_*sizeof(double), //dst_stride
                                                    size_C_transfer, //src_stride
                                                    (dim_i_stride) //repeat

                        );
                    }
                }

                snrt_dma_wait_all();  

                uint32_t end_parall = read_csr(mcycle);

                // if(snrt_global_core_idx()==0) snrt_printf("[%u], Total Time %u\n\r",end_parall-start_parall);
                #endif
            }
        }

    }


    void POLYBENCH_GEMM_OMPTarget::OMPTarget_conclusion(){
        #ifndef __HERO_1
        std::cout << "Conclusion" << std::endl;
        for(int i=0; i < NI*NJ; i++){
            C[i] = (Real_type)C_virt[i];
        }
        hero_dev_l3_free(NULL,(uintptr_t) A_virt,(uintptr_t) A_phys);
        hero_dev_l3_free(NULL,(uintptr_t) B_virt,(uintptr_t) B_phys);
        hero_dev_l3_free(NULL,(uintptr_t) C_virt,(uintptr_t) C_phys);

        std::cout << "End Conclusion" << std::endl;
        #endif
    }

}