#ifndef __HERO_1
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_MVT.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf
{
namespace polybench
{


POLYBENCH_MVT::POLYBENCH_MVT(const RunParams& params)
  : KernelBase(rajaperf::Polybench_MVT, params)
{
  Index_type N_default = 1000;

  setDefaultProblemSize( N_default * N_default );
  setDefaultReps(100);

  m_N = std::sqrt( getTargetProblemSize() ) + std::sqrt(2)-1;


  setActualProblemSize( m_N * m_N );

  setItsPerRep( 2 * m_N );
  setKernelsPerRep(2);
  setBytesReadPerRep( 2*sizeof(Real_type ) * m_N +
                      1*sizeof(Real_type ) * m_N * m_N +

                      2*sizeof(Real_type ) * m_N +
                      1*sizeof(Real_type ) * m_N * m_N );
  setBytesWrittenPerRep( 1*sizeof(Real_type ) * m_N +

                         1*sizeof(Real_type ) * m_N );
  setBytesAtomicModifyWrittenPerRep( 0 );
  setFLOPsPerRep(2 * m_N*m_N +
                 2 * m_N*m_N );

  checksum_scale_factor = 1.0 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );

  setComplexity(Complexity::N);

  setUsesFeature(Kernel);

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

POLYBENCH_MVT::~POLYBENCH_MVT()
{
}

void POLYBENCH_MVT::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  allocAndInitData(m_y1, m_N, vid);
  allocAndInitData(m_y2, m_N, vid);
  allocAndInitData(m_A, m_N * m_N, vid);
  allocAndInitDataConst(m_x1, m_N, 0.0, vid);
  allocAndInitDataConst(m_x2, m_N, 0.0, vid);
}

void POLYBENCH_MVT::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_x1, m_N, checksum_scale_factor , vid);
  checksum[vid][tune_idx] += calcChecksum(m_x2, m_N, checksum_scale_factor , vid);
}

void POLYBENCH_MVT::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_x1, vid);
  deallocData(m_x2, vid);
  deallocData(m_y1, vid);
  deallocData(m_y2, vid);
  deallocData(m_A, vid);
}

} // end namespace polybench
} // end namespace rajaperf

#endif
