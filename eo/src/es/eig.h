#ifndef EIG_H__
#define EIG_H__

#include <matrices.h>
#include <valarray>

namespace eo {
/* ========================================================= */
/*
   Calculating eigenvalues and vectors.
   Input:
     N: dimension.
     C: lower_triangular NxN-matrix.
     niter: number of maximal iterations for QL-Algorithm.
   Output:
     diag: N eigenvalues.
     Q: Columns are normalized eigenvectors.
     return: number of iterations in QL-Algorithm.
 */
extern int eig( int N,  const lower_triangular_matrix& C, std::valarray<double>& diag, square_matrix& Q,
       int niter = 0);

} // namespace eo

#endif
