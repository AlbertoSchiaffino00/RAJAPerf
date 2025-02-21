#ifndef __HERO_1
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ADD.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace stream
{


ADD::ADD(const RunParams& params)
  : KernelBase(rajaperf::Stream_ADD, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(1000);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesReadPerRep( 2*sizeof(Real_type) * getActualProblemSize() );
  setBytesWrittenPerRep( 1*sizeof(Real_type ) * getActualProblemSize() );
  setBytesAtomicModifyWrittenPerRep( 0 );
  setFLOPsPerRep(1 * getActualProblemSize());

  setComplexity(Complexity::N);

  setUsesFeature(Forall);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( Lambda_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Base_SYCL );
  setVariantDefined( RAJA_SYCL );

  setVariantDefined( Kokkos_Lambda );
}

ADD::~ADD()
{
}

void ADD::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitData(m_a, getActualProblemSize(), vid);
  allocAndInitData(m_b, getActualProblemSize(), vid);
  allocAndInitDataConst(m_c, getActualProblemSize(), 0.0, vid);
}

void ADD::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_c, getActualProblemSize(), vid);
}

void ADD::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_a, vid);
  deallocData(m_b, vid);
  deallocData(m_c, vid);
}

} // end namespace stream
} // end namespace rajaperf

#endif
