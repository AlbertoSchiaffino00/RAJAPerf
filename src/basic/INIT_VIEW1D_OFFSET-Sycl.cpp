#ifndef __HERO_1
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D_OFFSET.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template <size_t work_group_size >
void INIT_VIEW1D_OFFSET::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getActualProblemSize()+1;

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  INIT_VIEW1D_OFFSET_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
 
      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend-ibegin, work_group_size);

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                                         [=] (sycl::nd_item<1> item ) {

          Index_type i = ibegin + item.get_global_id(0);
          if (i < iend) {
            INIT_VIEW1D_OFFSET_BODY
          }

        });
      });

    }
    stopTimer();
  
  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
        INIT_VIEW1D_OFFSET_BODY;
      });

    }
    stopTimer();

  } else {
     std::cout << "\n  INIT_VIEW1D_OFFSET : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(INIT_VIEW1D_OFFSET, Sycl)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL

#endif
