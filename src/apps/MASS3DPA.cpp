#ifndef __HERO_1
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>

namespace rajaperf
{
namespace apps
{


MASS3DPA::MASS3DPA(const RunParams& params)
  : KernelBase(rajaperf::Apps_MASS3DPA, params)
{
  m_NE_default = 8000;

  setDefaultProblemSize(m_NE_default*MPA_Q1D*MPA_Q1D*MPA_Q1D);
  setDefaultReps(50);

  m_NE = std::max((getTargetProblemSize() + (MPA_Q1D*MPA_Q1D*MPA_Q1D)/2) / (MPA_Q1D*MPA_Q1D*MPA_Q1D), Index_type(1));

  setActualProblemSize( m_NE*MPA_Q1D*MPA_Q1D*MPA_Q1D );

  setItsPerRep(getActualProblemSize());
  setKernelsPerRep(1);

  setBytesReadPerRep( 2*sizeof(Real_type) * MPA_Q1D*MPA_D1D + // B, Bt
                      2*sizeof(Real_type) * MPA_D1D*MPA_D1D*MPA_D1D*m_NE + // X, Y
                      1*sizeof(Real_type) * MPA_Q1D*MPA_Q1D*MPA_Q1D*m_NE ); // D
  setBytesWrittenPerRep( 1*sizeof(Real_type) * MPA_D1D*MPA_D1D*MPA_D1D*m_NE ); // Y
  setBytesAtomicModifyWrittenPerRep( 0 );

  setFLOPsPerRep(m_NE * (2 * MPA_D1D * MPA_D1D * MPA_D1D * MPA_Q1D +
                         2 * MPA_D1D * MPA_D1D * MPA_Q1D * MPA_Q1D +
                         2 * MPA_D1D * MPA_Q1D * MPA_Q1D * MPA_Q1D + MPA_Q1D * MPA_Q1D * MPA_Q1D +
                         2 * MPA_Q1D * MPA_Q1D * MPA_Q1D * MPA_D1D +
                         2 * MPA_Q1D * MPA_Q1D * MPA_D1D * MPA_D1D +
                         2 * MPA_Q1D * MPA_D1D * MPA_D1D * MPA_D1D + MPA_D1D * MPA_D1D * MPA_D1D));

  setComplexity(Complexity::N);

  setUsesFeature(Launch);

  setVariantDefined( Base_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Base_SYCL );
  setVariantDefined( RAJA_SYCL );

}

MASS3DPA::~MASS3DPA()
{
}

void MASS3DPA::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{

  allocAndInitDataConst(m_B, int(MPA_Q1D*MPA_D1D), Real_type(1.0), vid);
  allocAndInitDataConst(m_Bt,int(MPA_Q1D*MPA_D1D), Real_type(1.0), vid);
  allocAndInitDataConst(m_D, int(MPA_Q1D*MPA_Q1D*MPA_Q1D*m_NE), Real_type(1.0), vid);
  allocAndInitDataConst(m_X, int(MPA_D1D*MPA_D1D*MPA_D1D*m_NE), Real_type(1.0), vid);
  allocAndInitDataConst(m_Y, int(MPA_D1D*MPA_D1D*MPA_D1D*m_NE), Real_type(0.0), vid);
}

void MASS3DPA::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_Y, MPA_D1D*MPA_D1D*MPA_D1D*m_NE, vid);
}

void MASS3DPA::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;

  deallocData(m_B, vid);
  deallocData(m_Bt, vid);
  deallocData(m_D, vid);
  deallocData(m_X, vid);
  deallocData(m_Y, vid);
}

} // end namespace apps
} // end namespace rajaperf

#endif
