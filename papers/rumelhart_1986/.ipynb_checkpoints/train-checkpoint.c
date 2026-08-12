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