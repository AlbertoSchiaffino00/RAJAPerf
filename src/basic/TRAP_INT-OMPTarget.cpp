#ifndef __HERO_1
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "TRAP_INT-func.hpp"

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;


void TRAP_INT::runOpenMPTargetVariant(VariantID vid, size_t tune_idx)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRAP_INT_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    #pragma omp target enter data map(to:x0,xp,y,yp,h)

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type sumx = m_sumx_init;

      #pragma omp target teams distribute parallel for \
                         map(tofrom: sumx) reduction(+:sumx) \
                         thread_limit(threads_per_team) schedule(static, 1)

      for (Index_type i = ibegin; i < iend; ++i ) {
        TRAP_INT_BODY;
      }

      m_sumx += sumx * h;

    }
    stopTimer();

    #pragma omp target exit data map(delete: x0,xp,y,yp,h)

  } else if ( vid == RAJA_OpenMPTarget ) {

    if (tune_idx == 0) {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::omp_target_reduce, Real_type> sumx(m_sumx_init);

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          TRAP_INT_BODY;
        });

        m_sumx += static_cast<Real_type>(sumx.get()) * h;

      }
      stopTimer();

    } else if (tune_idx == 1) {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type tsumx = m_sumx_init;

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
          RAJA::RangeSegment(ibegin, iend),
          RAJA::expt::Reduce<RAJA::operators::plus>(&tsumx),
          [=] (Index_type i, Real_type& sumx) {
            TRAP_INT_BODY;
          }
        );

        m_sumx += static_cast<Real_type>(tsumx) * h;

      }
      stopTimer();

    } else {
      getCout() << "\n  TRAP_INT : Unknown OMP Target tuning index = " << tune_idx << std::endl;
    }

  } else {
     getCout() << "\n  TRAP_INT : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

void TRAP_INT::setOpenMPTargetTuningDefinitions(VariantID vid)
{
  addVariantTuningName(vid, "default");
  if (vid == RAJA_OpenMPTarget) {
    addVariantTuningName(vid, "new");
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

#endif
