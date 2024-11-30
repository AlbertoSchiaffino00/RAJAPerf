#ifndef __HERO_1
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GESUMMV.hpp"

#include "RAJA/RAJA.hpp"

#include "POLYBENCH_GESUMMV_OMP.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

void POLYBENCH_GESUMMV::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_GESUMMV_DATA_SETUP;

  POLYBENCH_GESUMMV_OMPTarget gemm( A, B, x, y, alpha, beta, N, run_reps );

  if ( vid == Base_OpenMPTarget ) {

    gemm.OMPTarget_initialization();

    startTimer();

    gemm.POLYBENCH_GESUMMV_OMP();

    stopTimer();

    gemm.OMPTarget_conclusion();


  } else if (vid == RAJA_OpenMPTarget) {

      gemm.OMPTarget_initialization();

      startTimer();

      gemm.POLYBENCH_GESUMMV_OMP_opt();

      stopTimer();

      gemm.OMPTarget_conclusion();

    // POLYBENCH_GESUMMV_VIEWS_RAJA;

    // using EXEC_POL =
    //   RAJA::KernelPolicy<
    //     RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<threads_per_team>,   // i
    //       RAJA::statement::Lambda<0, RAJA::Params<0,1>>,
    //       RAJA::statement::For<1, RAJA::seq_exec,     // j
    //         RAJA::statement::Lambda<1,  RAJA::Segs<0,1>, RAJA::Params<0,1>>
    //       >,
    //       RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0,1>>
    //     >
    //   >;

    //   startTimer();
    //   for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

    //     RAJA::kernel_param<EXEC_POL>(
    //       RAJA::make_tuple( RAJA::RangeSegment{0, N},
    //                         RAJA::RangeSegment{0, N} ),
    //       RAJA::make_tuple(static_cast<Real_type>(0.0),
    //                        static_cast<Real_type>(0.0)),

    //       [=] (Real_type& tmpdot,
    //            Real_type& ydot) {
    //         POLYBENCH_GESUMMV_BODY1_RAJA;
    //       },
    //       [=] (Index_type i, Index_type j, Real_type& tmpdot,
    //                                        Real_type& ydot) {
    //         POLYBENCH_GESUMMV_BODY2_RAJA;
    //       },
    //       [=] (Index_type i, Real_type& tmpdot,
    //                          Real_type& ydot) {
    //         POLYBENCH_GESUMMV_BODY3_RAJA;
    //       }
    //     );

    //   }
      // stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_GESUMMV : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP


#endif
