#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define EPS_BN 1e-5f

typedef struct {
    float *gamma;     // scale parameter (D)
    float *beta;      // shift parameter (D)
    float *dgamma;    // scale gradient  (D)
    float *dbeta;     // shift gradient  (D)
    float *_x_hat;    // normalized cash intermediate (max_N * D)
    float *_inv_std;  // inverse standard deviation cache (D)
    float *mu;        // transient mean workspace (D)
    float *var;       // transient variance workspace (D)
    int    D;         // feature dimension
    int    max_N;     // maximum pre-allocated batch size capacity
} BN;

/* helper allocation wrapper */
static float *callocf_safe(size_t count) {
    float *ptr = (float *)calloc(count, sizeof(float));
    if (!ptr) {
        fprintf(stderr, "[FATAL] Calloc failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

/*
 * Allocates BN state and static scratchpads once.
 * Avoids system calls in hot training loops.
 */
static BN bn_alloc(int D, int maxN) {
    BN bn;
    bn.D        = D;
    bn.max_N    = maxN;
    
    bn.gamma    = callocf_safe(D);
    bn.beta     = callocf_safe(D);
    bn.dgamma   = callocf_safe(D);
    bn.dbeta    = callocf_safe(D);
    bn._x_hat   = callocf_safe(maxN * D);
    bn._inv_std = callocf_safe(D);
    bn.mu       = callocf_safe(D);
    bn.var      = callocf_safe(D);

    // initialize scale parameters to 1.0
    for (int i = 0; i < D; i++) bn.gamma[i] = 1.0f;
    return bn;
}

/* Free all static buffers */
static void bn_free(BN *bn) {
    free(bn->gamma);    free(bn->beta);
    free(bn->dgamma);   free(bn->dbeta);
    free(bn->_x_hat);   free(bn->_inv_std);
    free(bn->mu);       free(bn->var);
    
    bn->gamma = bn->beta = bn->dgamma = bn->dbeta = NULL;
    bn->_x_hat = bn->_inv_std = bn->mu = bn->var = NULL;
}

/* 
 * Batch Normalization Forward Pass
 * Zero runtime allocation. Keeps contiguous inner strides across features.
 */
static void bn_forward(BN *bn, const float *X, float *out, int N) {
    int D = bn->D;
    
    // bounds check to guarantee pre-allocated workspace safety
    if (N > bn->maxN) {
        fprintf(stderr, "[FATAL] Batch size %d exceeds max capacity %d\n", N, bn->maxN);
        exit(EXIT_FAILURE);
    }

    // reset workspace buffers
    memset(bn->mu, 0, D * sizeof(float));
    memset(bn->var, 0, D * sizeof(float));

    /* 1. compute feature mean across the batch */
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            bn->mu[d] += X[n * D + d];
        }
    }
    for (int d = 0; d < D; d++) {
        bn->mu[d] /= N;
    }

    /* 2. compute feature variance across the batch */
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            float dif = X[n * D + d] - bn->mu[d];
            bn->var[d] += dif * dif;
        }
    }
    
    /* 3. compute inverse standard deviation */
    for (int d = 0; d < D; d++) {
        bn->var[d] /= N;
        bn->_inv_std[d] = 1.0f / sqrtf(bn->var[d] + EPS_BN);
    }

    /* 4. center, scale, and shift to output */
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            float x_hat = (X[n * D + d] - bn->mu[d]) * bn->_inv_std[d];
            bn->_x_hat[n * D + d] = x_hat; // cached for backprop
            out[n * D + d] = bn->gamma[d] * x_hat + bn->beta[d];
        }
    }
}

/* 
 * Batch Normalization Backward Pass (Closed Form formulation)
 * Zero runtime allocation. Evaluates gradients efficiently.
 */
static void bn_backward(BN *bn, const float *dout, float *dX, int N) {
    int D = bn->D;
    const float *restrict x_hat   = bn->_x_hat;
    const float *restrict inv_std = bn->_inv_std;

    /* 1. accumulate scale and shift parameter gradients */
    memset(bn->dgamma, 0, D * sizeof(float));
    memset(bn->dbeta, 0, D * sizeof(float));
    
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            bn->dgamma[d] += dout[n * D + d] * x_hat[n * D + d];
            bn->dbeta[d]  += dout[n * D + d];
        }
    }

    /* 2. evaluate input gradient dL/dX via simplified closed-form derivative */
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            float t = N * dout[n * D + d] - bn->dbeta[d] - x_hat[n * D + d] * bn->dgamma[d];
            dX[n * D + d] = bn->gamma[d] * inv_std[d] * t / N;
        }
    }
}
