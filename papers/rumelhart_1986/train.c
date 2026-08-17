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

    double *W;     // (nout, nin)
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

/* uniform random double in [-r, +r] */
static double frand(double r) {
    return ((double)rand() / (double)RAND_MAX * 2.0 - 1.0) * r;
}

/* safe calloc, (n, type-size) */
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

static void mlp_free(MLP *net) {
    if (!net) return;
    for (int l = 0; l < net->nlayers; l++) {
        Layer *L = &net->layers[l];
        free(L->W); free(L->b);
        free(L->dW); free(L->db);
        free(L->vW); free(L->vb);
        free(L->net); free(L->out); free(L->delta);
    }
    free(net->layers);
    free(net->sizes);
    free(net);
}

/* forward-prop one input vector (length sizes[0])
 * returns a ptr to the output layer's activations,
 * owned by the net. */
static const double *mlp_fwd(MLP *net, const double *in) {
    const double *curin = in;            // rumelhart col vec. (nin,)
    
    for (int l = 0; l < net->nlayers; l++) {
        Layer *L = &net->layers[l];
        for (int j = 0; j < L->nout; j++) {
            double netj = L->b[j];
            // W is stored row-major with shape (nout, nin)
            // extract row j containing in-weights for output unit j
            const double *Wrow = &L->W[j * L->nin];
            // continuous memory stride across cols maximizes cache locality
            for (int i = 0; i < L->nin; i++) {
                netj += Wrow[i] * curin[i];
            }
            L->net[j] = netj;
            L->out[j] = sigmoid(netj);
        }
        // cascade outs as inputs to the next layer
        curin = L->out;
    }
    // last layer's activation
    return net->layers[net->nlayers-1].out;
}


/* Back-propagate the error for the pattern just forward-propagated,
 * given its target vector. Accumulates into each layer's dW/db.
 * Returns the pattern's error E */
static double mlp_bwd(MLP *net, const double *in, const double *target) {
    int nlayers = net->nlayers;
    Layer *Lout = &net->layers[net->nlayers-1];

    // output layer: delta_j = (t_j - o_j) o_j (1 - o_j)
    double err = 0.0;
    for (int j = 0; j < Lout->nout; j++) {
        double o = Lout->out[j];
        double diff = target[j] - o;
        err += 0.5 * diff * diff;
        Lout->delta[j] = diff * o * (1.0 - o);
    }

    // Hidden layers, propagated backward:
    // delta_j = o_j (1 - o_j) * sum_k delta_k w_kj
    for (int l = nlayers-2; l >= 0; l--) {
        Layer *L = &net->layers[l];
        Layer *next = &net->layers[l+1];
        for (int j = 0; j < L->nout; j++) {
            double sum = 0.0;
            for (k = 0; k < next->nout; k++) {
                sum += next->delta[k] * next->W[k*next->nin + j];
            }
            double o = L->out[j];
            L->delta[j] = sum * o * (1.0 - o);
        }
    }

    // Accumulate gradients: dE/dw_ji ~ delta_j * o_i, where o_i is
    // L's input (previous layer's output, or the network
    // input for layer 0)
    for (int l = 0; l < nlayers; l++) {
        Layer *L = &net->layers[l];
        const double *layer_in = (l == 0) ? input : net->layers[l - 1].out;
        for (int j = 0; j < L->nout; j++) {
            double dj = L->delta[j];
            L->db[j] += dj;
            double *dWrow = &L->dW[j * L->nin];
            for (int i = 0; i < L->nin; i++) {
                dWrow[i] += dj * layer_in[i];
            }
        }
    }

    return err;
}