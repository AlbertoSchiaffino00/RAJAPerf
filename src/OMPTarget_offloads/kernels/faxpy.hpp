// Copyright 2022 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>

namespace rajaperf
{
// 64-bit AXPY: y = a * x + y
void  faxpy_v64b(double a, double  __attribute__((address_space(0))) *x, double  __attribute__((address_space(0))) *y, unsigned int avl) {
  unsigned int vl;

  // Stripmine and accumulate a partial vector
  do {
    asm volatile( "vsetvli %0, %1, e64, m8, ta, ma \n" 
                  "vle64.v v0, (%2) \n"
                  "vle64.v v8, (%3) \n"
                  "vfmacc.vf v8, %4, v0 \n"
                  "slli	  t0, %0, 3 \n"
                  "add    %2, %2, t0 \n"
                  "sub    %1, %1, %0 \n"
                  "vse64.v v8, (%3) \n"
                  "add    %3, %3, t0 \n"
                  : "=r"(vl), "+r"(avl),"+r"(x),"+r"(y) 
                  : "f"(a), "r"(sizeof(double))
                  : "memory", "t0");
  } while (avl > 0);
  
}

// // 32-bit AXPY: y = a * x + y
// void faxpy_v32b(const float a, const float *x, const float *y,
//                 unsigned int avl) {
//   unsigned int vl;

//   // Stripmine and accumulate a partial vector
//   do {
//     // Set the vl
//     asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));

//     // Load vectors
//     asm volatile("vle32.v v0, (%0)" ::"r"(x));
//     asm volatile("vle32.v v8, (%0)" ::"r"(y));

//     // Multiply-accumulate
//     asm volatile("vfmacc.vf v8, %0, v0" ::"f"(a));

//     // Store results
//     asm volatile("vse32.v v8, (%0)" ::"r"(y));

//     // Bump pointers
//     x += vl;
//     y += vl;
//     avl -= vl;
//   } while (avl > 0);
// }

// // 16-bit AXPY: y = a * x + y
// void faxpy_v16b(const _Float16 a, const _Float16 *x, const _Float16 *y,
//                 unsigned int avl) {
//   unsigned int vl;

//   // Stripmine and accumulate a partial vector
//   do {
//     // Set the vl
//     asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

//     // Load vectors
//     asm volatile("vle16.v v0, (%0)" ::"r"(x));
//     asm volatile("vle16.v v8, (%0)" ::"r"(y));

//     // Multiply-accumulate
//     asm volatile("vfmacc.vf v8, %0, v0" ::"f"(a));

//     // Store results
//     asm volatile("vse16.v v8, (%0)" ::"r"(y));

//     // Bump pointers
//     x += vl;
//     y += vl;
//     avl -= vl;
//   } while (avl > 0);
// }
} // namespace rajaperf