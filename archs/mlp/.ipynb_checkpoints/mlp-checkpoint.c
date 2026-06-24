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
#define B            256
#define EPOCHS       50
#define LR           3e-4f
#define WEIGHT_DECAY 1e-4f
#define DROPOUT      0.3f
#define EPS_BN       1e-5f
#define B1           0.9f
#define B2           0.999f
#define EPS_ADAM     1e-8f

/* helper fns */

static float* alloc_zero(int n) {
    float* p = calloc(n, sizeof(float));
    if (!p) {fprintf(stderr, "out of memory\n"); exit(1);}
    return p;
}

/* box-muller gaussian */
#define PI 3.14159265358979323846f
static void randn_fill(float* p, int n, float std) {
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

