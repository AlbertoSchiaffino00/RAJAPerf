#include "common/RPTypes.hpp"
#include "common/KernelBase.hpp"

#define N_BUFFER 512 //64 N per buffer
#define K_BUFFER 8

namespace rajaperf
{

class POLYBENCH_GESUMMV_OMPTarget{
    public:
        POLYBENCH_GESUMMV_OMPTarget(Real_ptr m_A, Real_ptr m_B, Real_ptr m_X, Real_ptr m_Y,
                                    Real_type m_alpha, Real_type m_beta,
                                    Index_type m_n,Index_type m_run_reps);
        void POLYBENCH_GESUMMV_OMP();
        void POLYBENCH_GESUMMV_OMP_opt();
        void OMPTarget_initialization();
        void OMPTarget_conclusion();


        Real_ptr A,B,X,Y;
        Real_type alpha, beta;
        Index_type N, run_reps;
        uint64_t A_phys, B_phys, X_phys, Y_phys;
        double *A_virt, *B_virt, *X_virt, *Y_virt;

};

}