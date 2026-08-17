#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * apple silicon AMX hardware acceleration for supported compilers.
 * otherwise, the optimised yet naive approach for matmul.
 */
#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

/* out(M,N) = a(M,K) @ b(K,N) : (X * w) */
static void mm(const float *restrict a, const float *restrict b, float *restrict out,
                     int M, int K, int N) {
    #ifdef USE_ACCELERATE
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, a, K, b, N, 0.0f, out, N);
    #else
        memset(out, 0, M*N * sizeof(float));
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                float amk = a[m*K + k];
                for (int n = 0; n < N; n++) {
                    out[m*N + n] += amk * b[k*N + n];
                }
            }
        }
    #endif
}

/* out(K,N) = a(M,K).T @ b(M,N) : (X.T * dout) */
static void mm_at(const float *restrict a, const float *restrict b, float *restrict out,
                        int M, int K, int N) {
    #ifdef USE_ACCELERATE
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    K, N, M, 1.0f, a, K, b, N, 0.0f, out, N);
    #else
        memset(out, 0, K*N * sizeof(float));
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                float akm = a[m*K + k];
                for (int n = 0; n < N; n++) {
                    out[k*N + n] += akm * b[m*N + n];
                }
            }
        }
    #endif
}

/* out(M,K) = a(M,N) @ b(K,N).T : (dout * w.T) */
static void mm_bt(const float *restrict a, const float *restrict b, float *restrict out,
                        int M, int K, int N) {
    #ifdef USE_ACCELERATE
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, K, N, 1.0f, a, N, b, N, 0.0f, out, K);
    #else
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                float acc = 0.0f;
                for (int n = 0; n < N; n++) {
                    acc += a[m*N + n] * b[k*N + n];
                }
                out[m*K + k] = acc;
            }
        }
    #endif
}