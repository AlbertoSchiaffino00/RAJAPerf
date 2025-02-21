//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_GEMM kernel reference implementation:
///
/// for (Index_type i = 0; i < NI; i++) {
///   for (Index_type j = 0; j < NJ; j++) {
///     C[i][j] *= beta;
///     double dot = 0.0;
///     for (Index_type k = 0; k < NK; k++) {
///       dot += alpha * A[i][k] * B[k][j];
///     }
///     C[i][j] = dot;
///   }
/// }


#ifndef RAJAPerf_POLYBENCH_GEMM_HPP
#define RAJAPerf_POLYBENCH_GEMM_HPP

#define POLYBENCH_GEMM_DATA_SETUP \
  const Index_type ni = m_ni; \
  const Index_type nj = m_nj; \
  const Index_type nk = m_nk; \
\
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  Real_ptr C = m_C;


#define POLYBENCH_GEMM_BODY1 \
  Real_type dot = 0.0;

#define POLYBENCH_GEMM_BODY2 \
  C[j + i*nj] *= beta;

#define POLYBENCH_GEMM_BODY3 \
  dot += alpha * A[k + i*nk] * B[j + k*nj];

#define POLYBENCH_GEMM_BODY4 \
  C[j + i*nj] += dot;


#define POLYBENCH_GEMM_BODY1_RAJA \
  dot = 0.0;

#define POLYBENCH_GEMM_BODY2_RAJA \
  Cview(i, j) *= beta;

#define POLYBENCH_GEMM_BODY3_RAJA \
  dot += alpha * Aview(i, k) * Bview(k, j);

#define POLYBENCH_GEMM_BODY4_RAJA \
  Cview(i, j) = dot;


#define POLYBENCH_GEMM_VIEWS_RAJA \
  using VIEW_TYPE = RAJA::View<Real_type, \
                               RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_TYPE Aview(A, RAJA::Layout<2>(ni, nk)); \
  VIEW_TYPE Bview(B, RAJA::Layout<2>(nk, nj)); \
  VIEW_TYPE Cview(C, RAJA::Layout<2>(ni, nj));


#include "common/KernelBase.hpp"

namespace rajaperf
{

class RunParams;

namespace polybench
{

class POLYBENCH_GEMM : public KernelBase
{
public:

  POLYBENCH_GEMM(const RunParams& params);

  ~POLYBENCH_GEMM();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runSyclVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  void setSyclTuningDefinitions(VariantID vid);

  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size,
                                                         integer::MultipleOf<32>>;

  Index_type m_ni;
  Index_type m_nj;
  Index_type m_nk;

  Real_type m_alpha;
  Real_type m_beta;
  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
