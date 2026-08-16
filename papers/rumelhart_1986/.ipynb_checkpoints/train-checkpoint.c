/* The minimal, efficient implementation of the generalised
 * delta rule (error back-propagation), following:
 * 
 * Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).
 * "Learning representations by back-propagating errors."
 * Nature, 323(6088), 533-536.
 *
 * Build:   gcc -O2 -Wall -Wextra -o train train.c -lm
 * Run:     ./train            (runs all three demos below)
 *          ./train xor        (just one)
 *          ./train encoder
 *          ./train symmetry
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* the lib */

typedef struct {
    int nin;
    int nout;

    double *W;     // (nin, nout), row-major
    double *b;     // (nout,)

    double *dW;    // grad acc., same shape
    double *db;

    double *vW;    // momentum buffers, Eq. 8
    doubel *vb;

    double *net;   // (nout,) pre-act
    double *out;   // (nout,) act
    double *delta; // (nout,) error
} Layer;

typedef struct {
    int nlayers;
    int *sizes;    // (nlayers+1), sizes[0] = input dim
    Layer *layers;
} MLP;

static double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// uniform random double in [-r, +r]
static double frand(double r) {
    return ((double)rand() / (double)RAND_MAX * 2.0 - 1.0) * r;
}

// safe calloc, (n, type-size)
static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p && n > 0 && size > 0) {
        fprintf(stderr, "[ERROR] malloc failed\n");
        exit(EXIT_FAILURE);
    }
    return p;
}

static MLP *mlp_init(const int *sizes, int nlayers, double init_r, unsigned int seed) {
    srand(seed);

    MLP *net = xcalloc(1, sizeof(MLP));
    net->nlayers = nlayers;
    net->sizes = xcalloc((size_t)(nlayers+1), sizeof(int));
    memcpy(net->sizes, sizes, sizeof(int) * (size_t)(nlayers+1));
    net->layers = xcalloc((size_t)nlayers, sizeof(Layer));

    for (int l = 0; l < nlayers; l++) {
        int nin = sizes[l];
        int nout = sizes[l+1];
        Layer *L = &net->layers[l];

        L->nin = nin;
        L->nout = nout;

        L->W = xcalloc((size_t)(nin*nout), sizeof(double));
        L->b = xcalloc((size_t)nout, sizeof(double));
        L->dW = xcalloc((size_t)(nin*nout), sizeof(double));
        L->db = xcalloc((size_t)nout, sizeof(double));
        L->vW = xcalloc((size_t)(nin*nout), sizeof(double));
        L->vb = xcalloc((size_t)nout, sizeof(double));
        L->net = xcalloc((size_t)nout, sizeof(double));
        L->out = xcalloc((size_t)nout, sizeof(double));
        L->delta = xcalloc((size_t)nout, sizeof(double));

        for (int i = 0; i < nin*nout; i++) L->W[i] = frand(init_r);
        for (int i = 0; i < nout; i++) L->b[i] = frand(init_r);
    }
    return net;
}