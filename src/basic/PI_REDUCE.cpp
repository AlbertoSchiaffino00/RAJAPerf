#ifndef __HERO_1
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


PI_REDUCE::PI_REDUCE(const RunParams& params)
  : KernelBase(rajaperf::Basic_PI_REDUCE, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(50);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesReadPerRep( 1*sizeof(Real_type) );
  setBytesWrittenPerRep( 1*sizeof(Real_type) );
  setBytesAtomicModifyWrittenPerRep( 0 );
  setFLOPsPerRep(6 * getActualProblemSize() + 1);

  setComplexity(Complexity::N);

  setUsesFeature(Forall);
  setUsesFeature(Reduction);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Base_SYCL );
  setVariantDefined( RAJA_SYCL );
}

PI_REDUCE::~PI_REDUCE()
{
}

void PI_REDUCE::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  m_dx = 1.0 / double(getActualProblemSize());
  m_pi_init = 0.0;
  m_pi = 0.0;
}

void PI_REDUCE::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += Checksum_type(m_pi);
}

void PI_REDUCE::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
}

} // end namespace basic
} // end namespace rajaperf

#endif
