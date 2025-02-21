#ifndef __HERO_1
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DOT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace stream
{


DOT::DOT(const RunParams& params)
  : KernelBase(rajaperf::Stream_DOT, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(2000);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesReadPerRep( 1*sizeof(Real_type) +
                      2*sizeof(Real_type) * getActualProblemSize() );
  setBytesWrittenPerRep( 1*sizeof(Real_type) );
  setBytesAtomicModifyWrittenPerRep( 0 );
  setFLOPsPerRep(2 * getActualProblemSize());

  setComplexity(Complexity::N);

  setUsesFeature( Forall );
  setUsesFeature( Reduction );

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

  setVariantDefined( Kokkos_Lambda );
}

DOT::~DOT()
{
}

void DOT::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitData(m_a, getActualProblemSize(), vid);
  allocAndInitData(m_b, getActualProblemSize(), vid);

  m_dot = 0.0;
  m_dot_init = 0.0;
}

void DOT::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += m_dot;
}

void DOT::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_a, vid);
  deallocData(m_b, vid);
}

} // end namespace stream
} // end namespace rajaperf

#endif
