/*
 * C++ification of Nikolaus Hansen's original C-source code for the
 * CMA-ES. These are the eigenvector calculations
 *
 * C++-ificiation performed by Maarten Keijzer (C) 2005. Licensed under
 * the LGPL. Original copyright of Nikolaus Hansen can be found below
 *
 * This algorithm is held almost completely intact. Some other datatypes are used,
 * but hardly any code has changed
 *
 */

/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
/* --- File: cmaes.c  -------- Author: Nikolaus Hansen   --- */
/* --------------------------------------------------------- */
/*
 *      CMA-ES for non-linear function minimization.
 *
 *           Copyright (C) 1996, 2003  Nikolaus Hansen.
 *           e-mail: hansen@bionik.tu-berlin.de
 *
 *           This library is free software; you can redistribute it and/or
 *           modify it under the terms of the GNU Lesser General Public
 *           License as published by the Free Software Foundation; either
 *           version 2.1 of the License, or (at your option) any later
 *           version (see http://www.gnu.org/copyleft/lesser.html).
 *
 *           This library is distributed in the hope that it will be useful,
 *           but WITHOUT ANY WARRANTY; without even the implied warranty of
 *           MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *           Lesser General Public License for more details.
 *
 *                                                             */
/* --- Changes : ---
 *   03/03/21: argument const double *rgFunVal of
 *   cmaes_ReestimateDistribution() was treated incorrectly.
 *   03/03/29: restart via cmaes_resume_distribution() implemented.
 *   03/03/30: Always max std dev / largest axis is printed first.
 *   03/08/30: Damping is adjusted for large mueff.
 *   03/10/30: Damping is adjusted for large mueff always.
 *   04/04/22: Cumulation time and damping for step size adjusted.
 *   No iniphase but conditional update of pc.
 *   Version 2.23.
 *                               */
#include "eig.h"

using namespace std;

/* ========================================================= */
/*
   Householder Transformation einer symmetrischen Matrix
   auf tridiagonale Form.
   -> n             : Dimension
   -> ma            : symmetrische nxn-Matrix
   <- ma            : Transformationsmatrix (ist orthogonal):
                      Tridiag.-Matrix == <-ma * ->ma * (<-ma)^t
   <- diag          : Diagonale der resultierenden Tridiagonalmatrix
   <- neben[0..n-1] : Nebendiagonale (==1..n-1) der res. Tridiagonalmatrix

   */
static void
Householder( int N, square_matrix& ma, valarray<double>& diag, double* neben)
{
  double epsilon;
  int i, j, k;
  double h, sum, tmp, tmp2;

  for (i = N-1; i > 0; --i)
  {
    h = 0.0;
    if (i == 1)
      neben[i] = ma[i][i-1];
    else
    {
      for (k = i-1, epsilon = 0.0; k >= 0; --k)
        epsilon += fabs(ma[i][k]);

      if (epsilon == 0.0)
        neben[i] = ma[i][i-1];
      else
      {
        for(k = i-1, sum = 0.0; k >= 0; --k)
        { /* i-te Zeile von i-1 bis links normieren */
          ma[i][k] /= epsilon;
          sum += ma[i][k]*ma[i][k];
        }
        tmp = (ma[i][i-1] > 0) ? -sqrt(sum) : sqrt(sum);
        neben[i] = epsilon*tmp;
        h = sum - ma[i][i-1]*tmp;
        ma[i][i-1] -= tmp;
        for (j = 0, sum = 0.0; j < i; ++j)
        {
          ma[j][i] = ma[i][j]/h;
          tmp = 0.0;
          for (k = j; k >= 0; --k)
            tmp += ma[j][k]*ma[i][k];
          for (k = j+1; k < i; ++k)
            tmp += ma[k][j]*ma[i][k];
          neben[j] = tmp/h;
          sum += neben[j] * ma[i][j];
        } /* for j */
        sum /= 2.*h;
        for (j = 0; j < i; ++j)
        {
          neben[j] -= ma[i][j]*sum;
          tmp = ma[i][j];
          tmp2 = neben[j];
          for (k = j; k >= 0; --k)
            ma[j][k] -= (tmp*neben[k] + tmp2*ma[i][k]);
        } /* for j */
      } /* else epsilon */
    } /* else i == 1 */
    diag[i] = h;
  } /* for i */

  diag[0] = 0.0;
  neben[0] = 0.0;

  for (i = 0; i < N; ++i)
  {
    if(diag[i] != 0.0)
      for (j = 0; j < i; ++j)
      {
        for (k = i-1, tmp = 0.0; k >= 0; --k)
          tmp += ma[i][k] * ma[k][j];
        for (k = i-1; k >= 0; --k)
          ma[k][j] -= tmp*ma[k][i];
      } /* for j   */
    diag[i] = ma[i][i];
    ma[i][i] = 1.0;
    for (k = i-1; k >= 0; --k)
      ma[k][i] = ma[i][k] = 0.0;
  } /* for i */
}

/*
  QL-Algorithmus mit implizitem Shift, zur Berechnung von Eigenwerten
  und -vektoren einer symmetrischen Tridiagonalmatrix.
  -> n     : Dimension.
  -> diag        : Diagonale der Tridiagonalmatrix.
  -> neben[0..n-1] : Nebendiagonale (==0..n-2), n-1. Eintrag beliebig
  -> mq    : Matrix output von Householder()
  -> maxIt : maximale Zahl der Iterationen
  <- diag  : Eigenwerte
  <- neben : Garbage
  <- mq    : k-te Spalte ist normalisierter Eigenvektor zu diag[k]

  */

static int
QLalgo( int N, valarray<double>& diag, square_matrix& mq,
        int maxIter, double* neben)
{
  int i, j, k, kp1, l;
  double tmp, diff, cneben, c1, c2, p;
  int iter;

  neben[N-1] = 0.0;
  for (i = 0, iter = 0; i < N && iter < maxIter; ++i)
    do /* while j != i */
    {
      for (j = i; j < N-1; ++j)
      {
        tmp = fabs(diag[j]) + fabs(diag[j+1]);
        if (fabs(neben[j]) + tmp == tmp)
          break;
      }
      if (j != i)
      {
        if (++iter > maxIter) return maxIter-1;
        diff = (diag[i+1]-diag[i])/neben[i]/2.0;
        if (diff >= 0)
          diff = diag[j] - diag[i] +
            neben[i] / (diff + sqrt(diff * diff + 1.0));
        else
          diff = diag[j] - diag[i] +
            neben[i] / (diff - sqrt(diff * diff + 1.0));
        c2 = c1 = 1.0;
        p = 0.0;
        for (k = j-1; k >= i; --k)
        {
          kp1 = k + 1;
          tmp = c2 * neben[k];
          cneben = c1 * neben[k];
          if (fabs(tmp) >= fabs(diff))
          {
            c1 = diff / tmp;
            c2 = 1. / sqrt(c1*c1 + 1.0);
            neben[kp1] = tmp / c2;
            c1 *= c2;
          }
          else
          {
            c2 = tmp / diff;
            c1 = 1. / sqrt(c2*c2 + 1.0);
            neben[kp1] = diff / c1;
            c2 *= c1;
          } /* else */
          tmp = (diag[k] - diag[kp1] + p) * c2 + 2.0 * c1 * cneben;
          diag[kp1] += tmp * c2 - p;
          p = tmp * c2;
          diff = tmp * c1 - cneben;
          for (l = N-1; l >= 0; --l) /* TF-Matrix Q */
          {
            tmp = mq[l][kp1];
            mq[l][kp1] = c2 * mq[l][k] + c1 * tmp;
            mq[l][k] = c1 * mq[l][k] - c2 * tmp;
          } /* for l */
        } /* for k */
        diag[i] -= p;
        neben[i] = diff;
        neben[j] = 0.0;
      } /* if */
    } while (j != i);
  return iter;
} /* QLalgo() */

/* ========================================================= */
/*
   Calculating eigenvalues and vectors.
   Input:
     N: dimension.
     C: lower_triangular NxN-matrix.
     niter: number of maximal iterations for QL-Algorithm.
     rgtmp: N+1-dimensional vector for temporal use.
   Output:
     diag: N eigenvalues.
     Q: Columns are normalized eigenvectors.
     return: number of iterations in QL-Algorithm.
 */

namespace eo {
int
eig( int N,  const lower_triangular_matrix& C, valarray<double>& diag, square_matrix& Q,
       int niter)
{
  int ret;
  int i, j;

  if (niter == 0) niter = 30*N;

  for (i=0; i < N; ++i)
  {
      vector<double>::const_iterator row = C[i];
      for (j = 0; j <= i; ++j)
          Q[i][j] = Q[j][i] = row[j];
  }

  double* rgtmp = new double[N+1];
  Householder( N, Q, diag, rgtmp);
  ret = QLalgo( N, diag, Q, niter, rgtmp+1);
  delete [] rgtmp;

  return ret;
}

} // namespace eo
