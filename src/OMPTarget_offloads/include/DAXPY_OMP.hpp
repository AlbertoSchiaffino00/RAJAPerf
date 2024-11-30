#include "common/RPTypes.hpp"
#include "common/KernelBase.hpp"

#define DIM_BUFFER 0x4000

namespace rajaperf
{

class DAXPY_OMPTarget{
    public:
        Real_ptr x;
        Real_ptr y;
        Real_type a;
        Index_type ibegin;
        Index_type iend;
        RepIndex_type run_reps;
        uint64_t x_phys, y_phys, a_phys;
        double *x_virt, *y_virt, *a_virt;

        DAXPY_OMPTarget(Real_ptr x, Real_ptr y, Real_type a,Index_type ibegin, Index_type iend, RepIndex_type run_reps);
        void DAXPY_OMP();
        void DAXPY_OMP_opt();
        void OMPTarget_initialization();
        void OMPTarget_conclusion();

};

}