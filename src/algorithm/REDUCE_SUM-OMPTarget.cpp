#ifndef __HERO_1
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;


void REDUCE_SUM::runOpenMPTargetVariant(VariantID vid, size_t tune_idx)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type sum = m_sum_init;

      #pragma omp target is_device_ptr(x) device( did ) map(tofrom:sum)
      #pragma omp teams distribute parallel for reduction(+:sum) \
              thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        REDUCE_SUM_BODY;
      }

      m_sum = sum;

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    if (tune_idx == 0) {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::omp_target_reduce, Real_type> sum(m_sum_init);

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
            REDUCE_SUM_BODY;
        });

        m_sum = sum.get();

      }
      stopTimer();

    } else if (tune_idx == 1) {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type tsum = m_sum_init;

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend),
          RAJA::expt::Reduce<RAJA::operators::plus>(&tsum),
          [=] (Index_type i, Real_type& sum) {
            REDUCE_SUM_BODY;
          }
        );

        m_sum = static_cast<Real_type>(tsum);

      }
      stopTimer();

    } else {
      getCout() << "\n  REDUCE_SUM : Unknown OMP Target tuning index = " << tune_idx << std::endl;
    }

  } else {
    getCout() << "\n  REDUCE_SUM : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

void REDUCE_SUM::setOpenMPTargetTuningDefinitions(VariantID vid)
{
  addVariantTuningName(vid, "default");
  if (vid == RAJA_OpenMPTarget) {
    addVariantTuningName(vid, "new");
  }
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

#endif
