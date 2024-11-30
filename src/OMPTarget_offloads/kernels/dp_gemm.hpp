
#ifdef __HERO_1
#define snrt_printf(fmt, ...) printf((__attribute__((address_space(1))) char *)(fmt), ##__VA_ARGS__)

void gemm_4xVL(double *c, const double *a, const double *b,
                 const unsigned int m_start, const unsigned int m_end,
                 const unsigned int N, const unsigned int P,
                 const double alpha, const double beta, const uint32_t first_iter) {
                    

    unsigned int p = 0;
    const unsigned int P_striding = P*sizeof(double);
    const unsigned int N_striding = N*sizeof(double);
    while (p < P) {
        // Calculate the vl
        size_t gvl;
        asm volatile("vsetvli %[gvl], %[vl], e64, m4, ta, ma"
                    : [gvl] "=r"(gvl)
                    : [vl] "r"(P - p));

        const double *b_ = b + p;
        double *c_ = c + p;

        for (unsigned int m = m_start; m < m_end; m += 4) {
            const double *a_ = a + m * N;
            const double *a__ = a_;

            asm volatile("vle64.v v16, (%0);" ::"r"(b_));
            const double *b__;
            // b__ = b_ + P;
            asm volatile("add %0, %1, %2" : "+r"( b__) :"r"( b_), "r"(P_striding));

            double *c__ = c_ + m * P;


            double t0, t1, t2, t3;


            asm volatile(   "fld   %[t0], (%[a__])              \n"         // t0 = *a__;
                            "add   %[a__], %[a__], %[N_striding]\n"         // a__ += N;
                            "fld    %[t1], (%[a__])             \n"
                            "add   %[a__], %[a__], %[N_striding]\n"
                            "fld    %[t2], (%[a__])             \n"
                            "add   %[a__], %[a__], %[N_striding]\n"
                            "fld    %[t3], (%[a__])             \n"
                    :   [t0] "+f"(t0), [t1] "+f"(t1),[t2] "+f"(t2), [t3] "+f"(t3), [a__] "+r"(a__)
                    : [N_striding]"r"(N_striding)
            );


            unsigned int n = 0;
            // uint32_t start = read_csr(mcycle);
            while (n < N_striding) {
                // a__ = a_ + ++n;
                asm volatile("addi %[n], %[n], %[incr]\n"
                            "add %[a__], %[a_], %[n]\n"
                    : [n] "+r"(n), [a__] "+r"(a__)
                    : [incr]"i"(sizeof(double)), [a_]"r"(a_)
                );


                asm volatile("vle64.v v20, (%0);" ::"r"(b__));
                // b__ += P;
                asm volatile("add %0, %0, %1" : "+r"( b__) : "r"(P_striding));


                if (n == sizeof(double)) {

                    asm volatile(   "vfmul.vf v0, v16, %[t0]            \n"
                                    "fld    %[t0], (%[a__])             \n"
                                    "add   %[a__], %[a__], %[N_striding]\n"
                                    "vfmul.vf v4, v16, %[t1]            \n"
                                    "fld    %[t1], (%[a__])             \n"
                                    "add   %[a__], %[a__], %[N_striding]\n"
                                    "vfmul.vf v8, v16, %[t2]            \n"
                                    "fld    %[t2], (%[a__])             \n"
                                    "add   %[a__], %[a__], %[N_striding]\n"
                                    "vfmul.vf v12, v16, %[t3]           \n"
                                    "fld    %[t3], (%[a__])             \n"

                        : [t0] "+f"(t0), [t1] "+f"(t1),[t2] "+f"(t2), [t3] "+f"(t3), [a__] "+r"(a__)
                        : [N_striding] "r"(N_striding)
                    );
                


                } else {

                    asm volatile(   "vfmacc.vf v0, %[t0], v16           \n"
                                    "fld    %[t0], (%[a__])             \n"
                                    "add   %[a__], %[a__], %[N_striding]\n"
                                    "vfmacc.vf v4, %[t1], v16           \n"
                                    "fld    %[t1], (%[a__])             \n"
                                    "add   %[a__], %[a__], %[N_striding]\n"
                                    "vfmacc.vf v8, %[t2], v16           \n"
                                    "fld    %[t2], (%[a__])             \n"
                                    "add   %[a__], %[a__], %[N_striding]\n"
                                    "vfmacc.vf v12, %[t3], v16          \n"
                                    "fld    %[t3], (%[a__])             \n"

                        : [t0] "+f"(t0), [t1] "+f"(t1),[t2] "+f"(t2), [t3] "+f"(t3), [a__] "+r"(a__)
                        : [N_striding] "r"(N_striding)
                    );
                }

                // a__ = a_ + ++n;
                asm volatile("addi %[n], %[n], %[incr]\n"
                            "add %[a__], %[a_], %[n]\n"
                    : [n] "+r"(n), [a__] "+r"(a__)
                    : [incr]"i"(sizeof(double)), [a_]"r"(a_)
                );

                if (n == N_striding)
                    break;

                asm volatile("vle64.v v16, (%0);" ::"r"(b__));
                // b__ += P;
                asm volatile("add %0, %0, %1" : "+r"( b__) : "r"(P_striding));

                asm volatile(   "vfmacc.vf v0, %[t0], v20           \n"
                                "fld    %[t0], (%[a__])             \n"
                                "add   %[a__], %[a__], %[N_striding]\n"
                                "vfmacc.vf v4, %[t1], v20           \n"
                                "fld    %[t1], (%[a__])             \n"
                                "add   %[a__], %[a__], %[N_striding]\n"
                                "vfmacc.vf v8, %[t2], v20           \n"
                                "fld    %[t2], (%[a__])             \n"
                                "add   %[a__], %[a__], %[N_striding]\n"
                                "vfmacc.vf v12, %[t3], v20          \n"
                                "fld    %[t3], (%[a__])             \n"

                    : [t0] "+f"(t0), [t1] "+f"(t1),[t2] "+f"(t2), [t3] "+f"(t3), [a__] "+r"(a__)
                    : [N_striding] "r"(N_striding)
                );
            }
            // uint32_t end = read_csr(mcycle);
            // if(snrt_global_core_idx()==0){
            //     snrt_printf("\n%x,N %u, Inner cycle: %u\n\r",N,  end-start);
            // }

            //last accumulation
            asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
            asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t1));
            asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t2));
            asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t3));

            double *c___ = c__;
            
            //multiply by alpha and hide load of c vectors inside
            asm volatile(   "vle64.v v16, (%[c])            \n" 
                            "add %[c], %[c], %[P]           \n"
                            "vfmul.vf v0, v0, %[alpha]      \n"
                            "vle64.v v20, (%[c])            \n"
                            "add %[c], %[c], %[P]           \n"
                            "vfmul.vf v4, v4, %[alpha]      \n"
                            "vle64.v v24, (%[c])            \n"
                            "add %[c], %[c], %[P]           \n"
                            "vfmul.vf v8, v8, %[alpha]      \n"
                            "vle64.v v28, (%[c])            \n"
                            "vfmul.vf v12, v12, %[alpha]    \n"

            :[c]"+r"(c___)
            :[alpha]"f"(alpha), [P]"r"(P_striding) );


            if(first_iter==1){

                asm volatile(   "vfmul.vf v16, v16, %[beta] \n"
                                "vfmul.vf v20, v20, %[beta] \n"
                                "vfmul.vf v24, v24, %[beta] \n"
                                "vfmul.vf v28, v28, %[beta] \n"

                                "vfadd.vv v0, v0, v16       \n"
                                "vse64.v v0, (%[c])         \n"
                                "add %[c], %[c], %[P]       \n"
                                
                                "vfadd.vv v4, v4, v20       \n"
                                "vse64.v v4, (%[c])         \n"
                                "add %[c], %[c], %[P]       \n"
                                
                                "vfadd.vv v8, v8, v24       \n"
                                "vse64.v v8, (%[c])         \n"
                                "add %[c], %[c], %[P]       \n"

                                "vfadd.vv v12, v12, v28     \n"
                                "vse64.v v12, (%[c])        \n"
                : [c]"+r"(c__)
                :[beta] "f"(beta), [P]"r"(P_striding)
                );

            }else{
                //add to c and store result
                asm volatile(   
                                "vfadd.vv v0, v0, v16       \n"
                                "vse64.v v0, (%[c])         \n"
                                "add %[c], %[c], %[P]       \n"
                                
                                "vfadd.vv v4, v4, v20       \n"
                                "vse64.v v4, (%[c])         \n"
                                "add %[c], %[c], %[P]       \n"
                                
                                "vfadd.vv v8, v8, v24       \n"
                                "vse64.v v8, (%[c])         \n"
                                "add %[c], %[c], %[P]       \n"

                                "vfadd.vv v12, v12, v28     \n"
                                "vse64.v v12, (%[c])        \n"
                : [c]"+r"(c__)
                : [P]"r"(P_striding)
                );                
            }
        
        }

        p += gvl;
    }
}

#endif // __HERO_1
