#include "common/RPTypes.hpp"
#include "common/KernelBase.hpp"

#define DIM_BUFFER 0x4000

namespace rajaperf
{

class POLYBENCH_GEMM_OMPTarget{
    public:
        POLYBENCH_GEMM_OMPTarget(Real_ptr m_A, Real_ptr m_B, Real_ptr m_C, 
                            Real_type m_alpha, Real_type m_beta,
                            Index_type m_ni,Index_type m_nj,Index_type m_nk,
                            Index_type m_run_reps);
        void POLYBENCH_GEMM_OMP();
        void POLYBENCH_GEMM_OMP_opt();
        void POLYBENCH_GEMM_OMP_opt_one_core();
        void POLYBENCH_GEMM_OMP_opt_one_team();
        void POLYBENCH_GEMM_OMP_opt_one_team_one_core();
        void OMPTarget_initialization();
        void OMPTarget_conclusion();


        Real_ptr A,B,C;
        Real_type alpha, beta;
        Index_type NI, NJ, NK, run_reps;
        uint64_t A_phys, B_phys, C_phys;
        double *A_virt, *B_virt, *C_virt;

};

}