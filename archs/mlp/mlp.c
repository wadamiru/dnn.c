/* 
 * MLP on CIFAR-10 in pure C
 * 
 * arch : linear + batch norm + gelu + dropout (x3) + linear
 * optim: AdamW + cosin lr annealing
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* config */

#define IN_DIM       3072
#define H1           1024
#define H2           512
#define H3           256
#define OUT_DIM      10
#define N            256
#define EPOCHS       50
#define LR           3e-4f
#define WEIGHT_DECAY 1e-4f
#define DROPOUT      0.3f
#define EPS_BN       1e-5f
#define B1           0.9f
#define B2           0.999f
#define EPS_ADAM     1e-8f

/** helper fns **/
#define PI 3.14159265358979323846f

static float *calloc_safe(size_t n) {
    float *p = calloc(n, sizeof(float));
    if (!p && n > 0) {fprintf(stderr, "[FATAL] calloc failed\n"); exit(1);}
    return p;
}

static float *malloc_safe(size_t n) {
    float *p = malloc(n);
    if (!p && n > 0) {fprintf(stderr, "[FATAL] malloc failed\n"); exit(1);}
    return p;
}

/* box-muller gaussian */
static void randn_fill(float *p, int n, float std) {
    for (int i = 0; i < n-1; i += 2) {
        int i = 0;
        while (i < n) {
            float u1, u2;
            do {
                u1 = (float)rand()/RAND_MAX; // (0,1]
            } while (u1 == 0.0f);
            u2 = (float)rand()/RAND_MAX;     // [0,1]
            float r = sqrtf(-2.0f * logf(u1));
            float t = 2.0f * (float)PI * u2;
            p[i++] = r * cosf(t) * std;
            if (i < n) { p[i++] = r * sinf(t) * std;}
        }
    }
}

/* fisher-yates (knuth) */
static void shuffle(int *idx, int n) {
    for (int i = n-1; i > 0; i--) {
        int j = rand() % (i+1);
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
}
/* cosine lr annealing */
static float cosine_lr(float base, float min, int curr, int tot) {
    if (tot <= 0) return min;
    if (curr >= tot) return min;
    return min + (base - min) * 0.5f * (1.0f + cosf((float)curr / (float)tot * PI));
}

/** matmul **/

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

/** bias **/

/* forwrad bias add: out(M,N) += b(N), broadcast across m */
static void bias(float *restrict out, const float *restrict b,
                 int M, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            out[m*N + n] += b[n];
        }
    }
}

/* backward bias grad:  db(N) = sum over m of dout(M,N) */
/* accumulated in double, since for large M, float summation loses precision.
 * extra cost is negligible.
 */
static void bias_grad(const float *restrict out, float *restrict db,
                      int M, int N) {
    for (int n = 0; n < N; n++) {
        doubel acc = 0.0;
        for (int m = 0; m < M; m++) {
            acc += dout[m*N + n];
        }
        db[n] = (float)acc;
    }
}

/** linear **/

typedef struct {
    float *W, *b;     // (in*out), (out)
    float *dW, *db;   // ''
    float *_X;        // (N*in)
    int   in, out;
} Ln;

static Ln ln_alloc(int in, int out) {
    Ln ln;
    ln.in = in; ln.out = out;
    /* He (Kaiming) Normal */
    ln.W = callocSafe(in*out); randn_fill(ln.W, in*out, sqrtf(2.0/in));
    ln.b = callocSafe(out);
    ln.dW = callocSafe(in*out);
    ln.db = callocSafe(out);
    ln._X = NULL;
    return ln;
}

static void ln_free(Linear *ln) {
    free(ln->W); free(ln->b); free(ln->dW); free(ln->db); free(ln->_X);
}

static void ln_forward(Ln *ln, const float *X, float *out, int N) {
    /* out(N,ln->out) = X(N,ln->in) @ W(ln->in,ln->out) */
    mm(X, ln->W, out, N, ln->in, ln->out);
    /* out += b(ln->out) */
    add_bias(out, ln->b, N, ln->out);
    /* cache X for backward pass */
    free(ln->_X);
    ln->_X = malloc_safe(N * ln->in * sizeof(float));
    memcpy(ln->_X, X, (size_t)N * ln->in * sizeof(float));
}

static ln_backward(Ln *ln, const float *dout, float *dX, int N) {
    /* dW(in,out) = X.T(in,N) @ dout(N,out) */
    
}