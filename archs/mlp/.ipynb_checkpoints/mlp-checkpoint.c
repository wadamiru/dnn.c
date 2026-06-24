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

static float* alloc_zero(int n)
{
    float* p = calloc(n, sizeof(float));
    if (!p) {fprintf(stderr, "out of memory\n"); exit(1);}
    return p;
}

