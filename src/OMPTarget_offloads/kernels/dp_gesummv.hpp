#ifdef __HERO_1
#define snrt_printf(fmt, ...) printf((__attribute__((address_space(1))) char *)(fmt), ##__VA_ARGS__)

__attribute__((always_inline)) void dp_gesummv_2x(const double *a, const double *b, const double *x, double *y, 
                    unsigned int index_start, unsigned int index_end,
                    double alpha, double beta, unsigned int N_stride, const bool first_iter, const bool last_iter) {

      // Stripmine and accumulate a partial reduced vector
      // int flops = 0;
      // uint32_t begin = read_csr(mcycle);
      const double *a_ = a + index_start*N_stride;
      const double *b_ = b + index_start*N_stride;
      const double *x_ = x;
      unsigned int vl;
      unsigned int avl = N_stride;
      unsigned int N_striding = N_stride*sizeof(double);

      asm volatile("vsetvli %0, %1, e64, m2, ta, ma" : "=r"(vl) : "r"(avl));

      const double *a__ = a_;
      const double *b__ = b_;

      asm volatile( 
                    "vle64.v v0, (%[a__])\n"
                    "vle64.v v4, (%[b__])\n"
                    "add  %[a__], %[a__], %[N_striding]\n"
                    "add  %[b__], %[b__], %[N_striding]\n"
                    "vle64.v v8, (%[x_])\n"
        :[a__] "+r"(a__), [b__] "+r"(b__) 
        :[x_]"r"(x_), [N_striding] "r"(N_striding)
        : "memory"
      );

      bool inner_first_iter = true;
      do{
        if(inner_first_iter && first_iter){
          asm volatile( 
                        "vle64.v v2,  (%[a__])\n"
                        "vle64.v v6,  (%[b__])\n"
                        "sll  t0,  %[vl], 3\n"
                        "vfmul.vv v24, v0, v8\n"
                        "add  %[a_], %[a_], t0\n"
                        "add  %[b_], %[b_], t0\n"
                        "vfmul.vv v28, v4, v8\n"
                        "add  %[x_], %[x_], t0\n"
            :[a_]"+r"(a_), [b_]"+r"(b_), [x_]"+r"(x_)
            :[N_striding]"r"(N_striding), [vl]"r"(vl),[a__]"r"(a__), [b__]"r"(b__)
            :"memory", "t0"
          );
          // flops += 16*2;
        }else{
          asm volatile( 
                        "vle64.v v2,  (%[a__])\n"
                        "vle64.v v6,  (%[b__])\n"
                        "sll  t0,  %[vl], 3\n"
                        "vfmacc.vv v24, v0, v8\n"
                        "add  %[a_], %[a_], t0\n"
                        "add  %[b_], %[b_], t0\n"
                        "vfmacc.vv v28, v4, v8\n"
                        "add  %[x_], %[x_], t0\n"
            :[a_]"+r"(a_), [b_]"+r"(b_), [x_]"+r"(x_)
            : [N_striding]"r"(N_striding), [vl]"r"(vl),[a__]"r"(a__), [b__]"r"(b__)
            :"memory", "t0"
          );
          // flops += 2*2*16;
        }
        avl -= vl;
        if(avl>0){
          a__= a_;
          b__= b_;
          if(inner_first_iter && first_iter){
            asm volatile( 
                          "mv %[inner_first_iter], zero\n"
                          "vle64.v v0,  (%[a__])\n"
                          "vle64.v v4, (%[b__])\n"
                          "vsetvli %[vl], %[avl], e64, m2, ta, ma\n"
                          "add  %[a__], %[a__], %[N_striding]\n"
                          "vfmul.vv v26, v2, v8\n"
                          "add  %[b__], %[b__], %[N_striding]\n"
                          "vfmul.vv v30, v6, v8\n"
                          "vle64.v v8, (%[x_])\n"
              :[vl]"=r"(vl), [a__]"+r"(a__), [b__]"+r"(b__), [x_]"+r"(x_), [inner_first_iter]"=r"(inner_first_iter)
              :[avl]"r"(avl), [N_striding]"r"(N_striding)
              : "memory"
            );
            // flops += 16*2;
          }else{
            asm volatile( 
                          "vle64.v v0,  (%[a__])\n"
                          "vle64.v v4, (%[b__])\n"
                          "vsetvli %[vl], %[avl], e64, m2, ta, ma\n"
                          "add  %[a__], %[a__], %[N_striding]\n"
                          "vfmacc.vv v26, v2, v8\n"
                          "add  %[b__], %[b__], %[N_striding]\n"
                          "vfmacc.vv v30, v6, v8\n"
                          "vle64.v v8, (%[x_])\n"
              :[vl]"=r"(vl), [a__]"+r"(a__), [b__]"+r"(b__), [x_]"+r"(x_)
              :[avl]"r"(avl), [N_striding]"r"(N_striding)
              : "memory"
            );  
            // flops += 2*2*16;           
          };
        }else{
          if (inner_first_iter && first_iter) {
            asm volatile("vfmul.vv v26, v2, v8");
            asm volatile("vfmul.vv v30, v6, v8");
            // flops += 16*2;
          } else {
            asm volatile("vfmacc.vv v26, v2, v8");
            asm volatile("vfmacc.vv v30, v6, v8");
            // flops += 2*2*16;
          }
        }
      }while(avl>0);

      if(last_iter){
        asm volatile( "vmv.v.x v12, zero\n"
                      "vfredusum.vs v12, v24, v12\n"
                      "vmv.v.x v14, zero\n"
                      "vfredusum.vs v14, v26, v14\n"
                      "vmv.v.x v16, zero\n"
                      "vfredusum.vs v16, v28, v16\n"
                      "vmv.v.x v18, zero\n"
                      "vfredusum.vs v18, v30, v18\n"

                      "vfmv.f.s ft0, v12\n"
                      "vfmv.f.s ft2, v14\n"
                      "vfmv.f.s ft4, v16\n"
                      "vfmv.f.s ft6, v18\n"

                      "fmul.d ft0, ft0, %[alpha]\n"
                      "fmul.d ft2, ft2, %[alpha]\n"
                      "fmul.d ft4, ft4, %[beta]\n"
                      "fmul.d ft6, ft6, %[beta]\n"
                      
                      "slli   t0,  %[i], 3\n" //x8
                      "add    t0,  t0, %[y]\n"

                      "fadd.d ft0, ft0, ft4\n"
                      "fadd.d ft2, ft2, ft6\n"

                      "fsd ft0, 0(t0)\n"
                      "fsd ft2, 8(t0)\n"
                      :
                      : [alpha] "f"(alpha), [beta] "f"(beta), [y] "r"(y), [i] "r"(index_start), [N] "r"(N_stride)
                      : "memory", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6","ft7","t0");
      // flops += 15*4+4+2;
      }


    // uint32_t end = read_csr(mcycle);
    // if(snrt_global_core_idx()==0) snrt_printf("\n[%u], Time: %u\n\r", end-begin);
    // if(snrt_global_core_idx()==0) snrt_printf("\n[%u], FLOPS: %u\n\r", flops);

}

#endif