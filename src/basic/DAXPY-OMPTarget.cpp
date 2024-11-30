//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __HERO_1
#include "DAXPY.hpp"
#include "DAXPY_OMP.hpp"
#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

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


void DAXPY::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();
  DAXPY_DATA_SETUP; 
  DAXPY_OMPTarget daxpy_omp(x, y, a, ibegin, iend, run_reps);

  if ( vid == Base_OpenMPTarget ) {
    daxpy_omp.OMPTarget_initialization();      

    startTimer();

    daxpy_omp.DAXPY_OMP();
    
    stopTimer();
    
    daxpy_omp.OMPTarget_conclusion();
  
  } else if ( vid == RAJA_OpenMPTarget ) {


    daxpy_omp.OMPTarget_initialization();      

    startTimer();
   
    daxpy_omp.DAXPY_OMP_opt();

    stopTimer();
  
    daxpy_omp.OMPTarget_conclusion();


  } else {
     getCout() << "\n  DAXPY : Unknown OMP Target variant id = " << vid << std::endl;
  }
}


} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

#endif  // __HERO_1