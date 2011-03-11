/***************************************************************************
 *   Copyright (C) 2005 by Waldo Cancino                                   *
 *   wcancino@icmc.usp.br                                                  *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


#include <matrixutils.h>

using namespace std;


void luinverse(double **inmat, double **imtrx, int size)
{
	double eps = 1.0e-20; /* ! */
	int i, j, k, l, maxi=0, idx, ix, jx;
	double sum, tmp, maxb, aw;
	int index[NUM_AA];
	double *wk;
	double omtrx[NUM_AA][NUM_AA];

	/* copy inmat to omtrx */
	for (i = 0; i < NUM_AA; i++)
		for (j = 0; j < NUM_AA; j++)
			omtrx[i][j] = inmat[i][j];

	wk = new double[size]; //(double *) malloc((unsigned)size * sizeof(double));
	aw = 1.0;
	for (i = 0; i < size; i++)
	{
		maxb = 0.0;
		for (j = 0; j < size; j++)
		{
			if (fabs(omtrx[i][j]) > maxb)
				maxb = fabs(omtrx[i][j]);
		}
		if (maxb == 0.0)
		{
			/* Singular matrix */
			cout << "Error finding LU decomposition.  Singular matrix.";
			exit(1);
		}
		wk[i] = 1.0 / maxb;
	}
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < j; i++)
		{
			sum = omtrx[i][j];
			for (k = 0; k < i; k++)
				sum -= omtrx[i][k] * omtrx[k][j];
			omtrx[i][j] = sum;
		}
		maxb = 0.0;
		for (i = j; i < size; i++)
		{
			sum = omtrx[i][j];
			for (k = 0; k < j; k++)
				sum -= omtrx[i][k] * omtrx[k][j];
			omtrx[i][j] = sum;
			tmp = wk[i] * fabs(sum);
			if (tmp >= maxb)
			{
				maxb = tmp;
				maxi = i;
			}
		}
		if (j != maxi)
		{
			for (k = 0; k < size; k++)
			{
				tmp = omtrx[maxi][k];
				omtrx[maxi][k] = omtrx[j][k];
				omtrx[j][k] = tmp;
			}
			aw = -aw;
			wk[maxi] = wk[j];
		}
		index[j] = maxi;
		if (omtrx[j][j] == 0.0)
			omtrx[j][j] = eps;
		if (j != size - 1)
		{
			tmp = 1.0 / omtrx[j][j];
			for (i = j + 1; i < size; i++)
				omtrx[i][j] *= tmp;
		}
	}
	for (jx = 0; jx < size; jx++)
	{
		for (ix = 0; ix < size; ix++)
			wk[ix] = 0.0;
		wk[jx] = 1.0;
		l = -1;
		for (i = 0; i < size; i++)
		{
			idx = index[i];
			sum = wk[idx];
			wk[idx] = wk[i];
			if (l != -1)
			{
				for (j = l; j < i; j++)
					sum -= omtrx[i][j] * wk[j];
			}
			else if (sum != 0.0)
				l = i;
			wk[i] = sum;
		}
		for (i = size - 1; i >= 0; i--)
		{
			sum = wk[i];
			for (j = i + 1; j < size; j++)
				sum -= omtrx[i][j] * wk[j];
			wk[i] = sum / omtrx[i][i];
		}
		for (ix = 0; ix < size; ix++)
			imtrx[ix][jx] = wk[ix];
	}
	delete [] wk;
	//free(wk);

} /*_ luinverse */

void elmhes(double **a, int ordr[], int n)
{
	int m, j, i;
	double y, x;


	for (i = 0; i < n; i++)
		ordr[i] = 0;
	for (m = 2; m < n; m++)
	{
		x = 0.0;
		i = m;
		for (j = m; j <= n; j++)
		{
			if (fabs(a[j - 1][m - 2]) > fabs(x))
			{
				x = a[j - 1][m - 2];
				i = j;
			}
		}
		ordr[m - 1] = i;      /* vector */
		if (i != m)
		{
			for (j = m - 2; j < n; j++)
			{
				y = a[i - 1][j];
				a[i - 1][j] = a[m - 1][j];
				a[m - 1][j] = y;
			}
			for (j = 0; j < n; j++)
			{
				y = a[j][i - 1];
				a[j][i - 1] = a[j][m - 1];
				a[j][m - 1] = y;
			}
		}
		if (x != 0.0)
		{
			for (i = m; i < n; i++)
			{
				y = a[i][m - 2];
				if (y != 0.0)
				{
					y /= x;
					a[i][m - 2] = y;
					for (j = m - 1; j < n; j++)
						a[i][j] -= y * a[m - 1][j];
					for (j = 0; j < n; j++)
						a[j][m - 1] += y * a[j][i];
				}
			}
		}
	}
} /*_ elmhes */

void eltran(double **a, double **zz, int ordr[NUM_AA], int n)
{
	int i, j, m;


	for (i = 0; i < n; i++)
	{
		for (j = i + 1; j < n; j++)
		{
			zz[i][j] = 0.0;
			zz[j][i] = 0.0;
		}
		zz[i][i] = 1.0;
	}
	if (n <= 2)
		return;
	for (m = n - 1; m >= 2; m--)
	{
		for (i = m; i < n; i++)
			zz[i][m - 1] = a[i][m - 2];
		i = ordr[m - 1];
		if (i != m)
		{
			for (j = m - 1; j < n; j++)
			{
				zz[m - 1][j] = zz[i - 1][j];
				zz[i - 1][j] = 0.0;
			}
			zz[i - 1][m - 1] = 1.0;
		}
	}
} /*_ eltran */

void hqr2(int n, int low, int hgh, double **h,
                      double **zz, double wr[NUM_AA], double wi[NUM_AA])
{
	int i, j, k, l=0, m, en, na, itn, its;
	double p=0, q=0, r=0, s=0, t, w, x=0, y, ra, sa, vi, vr, z=0, norm, tst1, tst2;
	int notlas; /* boolean */

	norm = 0.0;
	k = 1;
	/* store isolated roots and compute matrix norm */
	for (i = 0; i < n; i++)
	{
		for (j = k - 1; j < n; j++)
			norm += fabs(h[i][j]);
		k = i + 1;
		if (i + 1 < low || i + 1 > hgh)
		{
			wr[i] = h[i][i];
			wi[i] = 0.0;
		}
	}
	en = hgh;
	t = 0.0;
	itn = n * 30;
	while (en >= low)
	{     /* search for next eigenvalues */
		its = 0;
		na = en - 1;
		while (en >= 1)
		{
			/* look for single small sub-diagonal element */
			for (l = en; l > low; l--)
			{
				s = fabs(h[l - 2][l - 2]) + fabs(h[l - 1][l - 1]);
				if (s == 0.0)
					s = norm;
				tst1 = s;
				tst2 = tst1 + fabs(h[l - 1][l - 2]);
				if (tst2 == tst1)
					goto L100;
			}
			l = low;
		L100:
			x = h[en - 1][en - 1];  /* form shift */
			if (l == en || l == na)
				break;
			if (itn == 0)
			{
				/* all eigenvalues have not converged */
				cout << "Eigenvalues have not converged!\n";
				exit(1);
			}
			y = h[na - 1][na - 1];
			w = h[en - 1][na - 1] * h[na - 1][en - 1];
			/* form exceptional shift */
			if (its == 10 || its == 20)
			{
				t += x;
				for (i = low - 1; i < en; i++)
					h[i][i] -= x;
				s = fabs(h[en - 1][na - 1]) + fabs(h[na - 1][en - 3]);
				x = 0.75 * s;
				y = x;
				w = -0.4375 * s * s;
			}
			its++;
			itn--;
			/* look for two consecutive small sub-diagonal elements */
			for (m = en - 2; m >= l; m--)
			{
				z = h[m - 1][m - 1];
				r = x - z;
				s = y - z;
				p = (r * s - w) / h[m][m - 1] + h[m - 1][m];
				q = h[m][m] - z - r - s;
				r = h[m + 1][m];
				s = fabs(p) + fabs(q) + fabs(r);
				p /= s;
				q /= s;
				r /= s;
				if (m == l)
					break;
				tst1 = fabs(p) *
				       (fabs(h[m - 2][m - 2]) + fabs(z) + fabs(h[m][m]));
				tst2 = tst1 + fabs(h[m - 1][m - 2]) * (fabs(q) + fabs(r));
				if (tst2 == tst1)
					break;
			}
			for (i = m + 2; i <= en; i++)
			{
				h[i - 1][i - 3] = 0.0;
				if (i != m + 2)
					h[i - 1][i - 4] = 0.0;
			}
			for (k = m; k <= na; k++)
			{
				notlas = (k != na);
				if (k != m)
				{
					p = h[k - 1][k - 2];
					q = h[k][k - 2];
					r = 0.0;
					if (notlas)
						r = h[k + 1][k - 2];
					x = fabs(p) + fabs(q) + fabs(r);
					if (x != 0.0)
					{
						p /= x;
						q /= x;
						r /= x;
					}
				}
				if (x != 0.0)
				{
					if (p < 0.0) /* sign */
						s = - sqrt(p * p + q * q + r * r);
					else
						s = sqrt(p * p + q * q + r * r);
					if (k != m)
						h[k - 1][k - 2] = -s * x;
					else
					{
						if (l != m)
							h[k - 1][k - 2] = -h[k - 1][k - 2];
					}
					p += s;
					x = p / s;
					y = q / s;
					z = r / s;
					q /= p;
					r /= p;
					if (!notlas)
					{
						for (j = k - 1; j < n; j++)
						{   /* row modification */
							p = h[k - 1][j] + q * h[k][j];
							h[k - 1][j] -= p * x;
							h[k][j] -= p * y;
						}
						j = (en < (k + 3)) ? en : (k + 3); /* min */
						for (i = 0; i < j; i++)
						{       /* column modification */
							p = x * h[i][k - 1] + y * h[i][k];
							h[i][k - 1] -= p;
							h[i][k] -= p * q;
						}
						/* accumulate transformations */
						for (i = low - 1; i < hgh; i++)
						{
							p = x * zz[i][k - 1] + y * zz[i][k];
							zz[i][k - 1] -= p;
							zz[i][k] -= p * q;
						}
					}
					else
					{
						for (j = k - 1; j < n; j++)
						{   /* row modification */
							p = h[k - 1][j] + q * h[k][j] + r * h[k + 1][j];
							h[k - 1][j] -= p * x;
							h[k][j] -= p * y;
							h[k + 1][j] -= p * z;
						}
						j = (en < (k + 3)) ? en : (k + 3); /* min */
						for (i = 0; i < j; i++)
						{       /* column modification */
							p = x * h[i][k - 1] + y * h[i][k] + z * h[i][k + 1];
							h[i][k - 1] -= p;
							h[i][k] -= p * q;
							h[i][k + 1] -= p * r;
						}
						/* accumulate transformations */
						for (i = low - 1; i < hgh; i++)
						{
							p = x * zz[i][k - 1] + y * zz[i][k] +
							    z * zz[i][k + 1];
							zz[i][k - 1] -= p;
							zz[i][k] -= p * q;
							zz[i][k + 1] -= p * r;
						}
					}
				}
			}              /* for k */
		}                      /* while infinite loop */
		if (l == en)
		{         /* one root found */
			h[en - 1][en - 1] = x + t;
			wr[en - 1] = h[en - 1][en - 1];
			wi[en - 1] = 0.0;
			en = na;
			continue;
		}
		y = h[na - 1][na - 1];
		w = h[en - 1][na - 1] * h[na - 1][en - 1];
		p = (y - x) / 2.0;
		q = p * p + w;
		z = sqrt(fabs(q));
		h[en - 1][en - 1] = x + t;
		x = h[en - 1][en - 1];
		h[na - 1][na - 1] = y + t;
		if (q >= 0.0)
		{        /* real pair */
			if (p < 0.0) /* sign */
				z = p - fabs(z);
			else
				z = p + fabs(z);
			wr[na - 1] = x + z;
			wr[en - 1] = wr[na - 1];
			if (z != 0.0)
				wr[en - 1] = x - w / z;
			wi[na - 1] = 0.0;
			wi[en - 1] = 0.0;
			x = h[en - 1][na - 1];
			s = fabs(x) + fabs(z);
			p = x / s;
			q = z / s;
			r = sqrt(p * p + q * q);
			p /= r;
			q /= r;
			for (j = na - 1; j < n; j++)
			{  /* row modification */
				z = h[na - 1][j];
				h[na - 1][j] = q * z + p * h[en - 1][j];
				h[en - 1][j] = q * h[en - 1][j] - p * z;
			}
			for (i = 0; i < en; i++)
			{      /* column modification */
				z = h[i][na - 1];
				h[i][na - 1] = q * z + p * h[i][en - 1];
				h[i][en - 1] = q * h[i][en - 1] - p * z;
			}
			/* accumulate transformations */
			for (i = low - 1; i < hgh; i++)
			{
				z = zz[i][na - 1];
				zz[i][na - 1] = q * z + p * zz[i][en - 1];
				zz[i][en - 1] = q * zz[i][en - 1] - p * z;
			}
		}
		else
		{               /* complex pair */
			wr[na - 1] = x + p;
			wr[en - 1] = x + p;
			wi[na - 1] = z;
			wi[en - 1] = -z;
		}
		en -= 2;
	}                              /* while en >= low */
	/* backsubstitute to find vectors of upper triangular form */
	if (norm != 0.0)
	{
		for (en = n; en >= 1; en--)
		{
			p = wr[en - 1];
			q = wi[en - 1];
			na = en - 1;
			if (q == 0.0)
			{/* real vector */
				m = en;
				h[en - 1][en - 1] = 1.0;
				if (na != 0)
				{
					for (i = en - 2; i >= 0; i--)
					{
						w = h[i][i] - p;
						r = 0.0;
						for (j = m - 1; j < en; j++)
							r += h[i][j] * h[j][en - 1];
						if (wi[i] < 0.0)
						{
							z = w;
							s = r;
						}
						else
						{
							m = i + 1;
							if (wi[i] == 0.0)
							{
								t = w;
								if (t == 0.0)
								{
									tst1 = norm;
									t = tst1;
									do
									{
										t = 0.01 * t;
										tst2 = norm + t;
									}
									while (tst2 > tst1);
								}
								h[i][en - 1] = -(r / t);
							}
							else
							{        /* solve real equations */
								x = h[i][i + 1];
								y = h[i + 1][i];
								q = (wr[i] - p) * (wr[i] - p) + wi[i] * wi[i];
								t = (x * s - z * r) / q;
								h[i][en - 1] = t;
								if (fabs(x) > fabs(z))
									h[i + 1][en - 1] = (-r - w * t) / x;
								else
									h[i + 1][en - 1] = (-s - y * t) / z;
							}
							/* overflow control */
							t = fabs(h[i][en - 1]);
							if (t != 0.0)
							{
								tst1 = t;
								tst2 = tst1 + 1.0 / tst1;
								if (tst2 <= tst1)
								{
									for (j = i; j < en; j++)
										h[j][en - 1] /= t;
								}
							}
						}
					}
				}
			}
			else if (q > 0.0)
			{
				m = na;
				if (fabs(h[en - 1][na - 1]) > fabs(h[na - 1][en - 1]))
				{
					h[na - 1][na - 1] = q / h[en - 1][na - 1];
					h[na - 1][en - 1] = (p - h[en - 1][en - 1]) /
					                    h[en - 1][na - 1];
				}
				else
					mcdiv(0.0, -h[na - 1][en - 1], h[na - 1][na - 1] - p, q,
					      &h[na - 1][na - 1], &h[na - 1][en - 1]);
				h[en - 1][na - 1] = 0.0;
				h[en - 1][en - 1] = 1.0;
				if (en != 2)
				{
					for (i = en - 3; i >= 0; i--)
					{
						w = h[i][i] - p;
						ra = 0.0;
						sa = 0.0;
						for (j = m - 1; j < en; j++)
						{
							ra += h[i][j] * h[j][na - 1];
							sa += h[i][j] * h[j][en - 1];
						}
						if (wi[i] < 0.0)
						{
							z = w;
							r = ra;
							s = sa;
						}
						else
						{
							m = i + 1;
							if (wi[i] == 0.0)
								mcdiv(-ra, -sa, w, q, &h[i][na - 1],
								      &h[i][en - 1]);
							else
							{  /* solve complex equations */
								x = h[i][i + 1];
								y = h[i + 1][i];
								vr = (wr[i] - p) * (wr[i] - p);
								vr = vr + wi[i] * wi[i] - q * q;
								vi = (wr[i] - p) * 2.0 * q;
								if (vr == 0.0 && vi == 0.0)
								{
									tst1 = norm * (fabs(w) + fabs(q) + fabs(x) +
									               fabs(y) + fabs(z));
									vr = tst1;
									do
									{
										vr = 0.01 * vr;
										tst2 = tst1 + vr;
									}
									while (tst2 > tst1);
								}
								mcdiv(x * r - z * ra + q * sa,
								      x * s - z * sa - q * ra, vr, vi,
								      &h[i][na - 1], &h[i][en - 1]);
								if (fabs(x) > fabs(z) + fabs(q))
								{
									h[i + 1]
									[na - 1] = (q * h[i][en - 1] -
									            w * h[i][na - 1] - ra) / x;
									h[i + 1][en - 1] = (-sa - w * h[i][en - 1] -
									                    q * h[i][na - 1]) / x;
								}
								else
									mcdiv(-r - y * h[i][na - 1],
									      -s - y * h[i][en - 1], z, q,
									      &h[i + 1][na - 1], &h[i + 1][en - 1]);
							}
							/* overflow control */
							t = (fabs(h[i][na - 1]) > fabs(h[i][en - 1])) ?
							    fabs(h[i][na - 1]) : fabs(h[i][en - 1]);
							if (t != 0.0)
							{
								tst1 = t;
								tst2 = tst1 + 1.0 / tst1;
								if (tst2 <= tst1)
								{
									for (j = i; j < en; j++)
									{
										h[j][na - 1] /= t;
										h[j][en - 1] /= t;
									}
								}
							}
						}
					}
				}
			}
		}
		/* end back substitution. vectors of isolated roots */
		for (i = 0; i < n; i++)
		{
			if (i + 1 < low || i + 1 > hgh)
			{
				for (j = i; j < n; j++)
					zz[i][j] = h[i][j];
			}
		}
		/* multiply by transformation matrix to give vectors of
		 * original full matrix. */
		for (j = n - 1; j >= low - 1; j--)
		{
			m = ((j + 1) < hgh) ? (j + 1) : hgh; /* min */
			for (i = low - 1; i < hgh; i++)
			{
				z = 0.0;
				for (k = low - 1; k < m; k++)
					z += zz[i][k] * h[k][j];
				zz[i][j] = z;
			}
		}
	}
	return;

} /*_ hqr2 */

void mcdiv(double ar, double ai, double br, double bi, double *cr, double *ci)
{
	double s, ars, ais, brs, bis;

	s = fabs(br) + fabs(bi);
	ars = ar / s;
	ais = ai / s;
	brs = br / s;
	bis = bi / s;
	s = brs * brs + bis * bis;
	*cr = (ars * brs + ais * bis) / s;
	*ci = (ais * brs - ars * bis) / s;
}


double gammln(double xx)
{
   double x,tmp,ser;
   static double cof[6]={76.18009173,-86.50532033,24.01409822,
      -1.231739516,0.120858003e-2,-0.536382e-5};
   int j;

   x=xx-1.0;
   tmp=x+5.5;
   tmp -= (x+0.5)*log(tmp);
   ser=1.0;
   for (j=0;j<=5;j++) {
      x += 1.0;
      ser += cof[j]/x;
   }
   return -tmp+log(2.50662827465*ser);
}


double LnGamma (double alpha)
{
/* returns ln(gamma(alpha)) for alpha>0, accurate to 10 decimal places.  
   Stirling's formula is used for the central polynomial part of the procedure.
   Pike MC & Hill ID (1966) Algorithm 291: Logarithm of the gamma function.
   Communications of the Association for Computing Machinery, 9:684
*/
   double x=alpha, f=0, z;

   if (x<7) {
      f=1;  z=x-1;
      while (++z<7)  f*=z;
      x=z;   f=-log(f);
   }
   z = 1/(x*x);
   return  f + (x-0.5)*log(x) - x + .918938533204673 
	  + (((-.000595238095238*z+.000793650793651)*z-.002777777777778)*z
	       +.083333333333333)/x;  
}

/*********************************************************/

double IncompleteGamma (double x, double alpha, double ln_gamma_alpha)
{
/* returns the incomplete gamma ratio I(x,alpha) where x is the upper 
	   limit of the integration and alpha is the shape parameter.
   returns (-1) if in error
   ln_gamma_alpha = ln(Gamma(alpha)), is almost redundant.
   (1) series expansion     if (alpha>x || x<=1)
   (2) continued fraction   otherwise
   RATNEST FORTRAN by
   Bhattacharjee GP (1970) The incomplete gamma integral.  Applied Statistics,
   19: 285-287 (AS32)
*/
   int i;
   double p=alpha, g=ln_gamma_alpha;
   double accurate=1e-8, overflow=1e30;
   double factor, gin=0, rn=0, a=0,b=0,an=0,dif=0, term=0, pn[6];

   if (x==0) return (0);
   if (x<0 || p<=0) return (-1);

   factor=exp(p*log(x)-x-g);   
   if (x>1 && x>=p) goto l30;
   /* (1) series expansion */
   gin=1;  term=1;  rn=p;
 l20:
   rn++;
   term*=x/rn;   gin+=term;

   if (term > accurate) goto l20;
   gin*=factor/p;
   goto l50;
 l30:
   /* (2) continued fraction */
   a=1-p;   b=a+x+1;  term=0;
   pn[0]=1;  pn[1]=x;  pn[2]=x+1;  pn[3]=x*b;
   gin=pn[2]/pn[3];
 l32:
   a++;  b+=2;  term++;   an=a*term;
   for (i=0; i<2; i++) pn[i+4]=b*pn[i+2]-an*pn[i];
   if (pn[5] == 0) goto l35;
   rn=pn[4]/pn[5];   dif=fabs(gin-rn);
   if (dif>accurate) goto l34;
   if (dif<=accurate*rn) goto l42;
 l34:
   gin=rn;
 l35:
   for (i=0; i<4; i++) pn[i]=pn[i+2];
   if (fabs(pn[4]) < overflow) goto l32;
   for (i=0; i<4; i++) pn[i]/=overflow;
   goto l32;
 l42:
   gin=1-factor*gin;

 l50:
   return (gin);
}



double PointNormal (double prob)
{
/* returns z so that Prob{x<z}=prob where x ~ N(0,1) and (1e-12)<prob<1-(1e-12)
   returns (-9999) if in error
   Odeh RE & Evans JO (1974) The percentage points of the normal distribution.
   Applied Statistics 22: 96-97 (AS70)

   Newer methods:
     Wichura MJ (1988) Algorithm AS 241: the percentage points of the
       normal distribution.  37: 477-484.
     Beasley JD & Springer SG  (1977).  Algorithm AS 111: the percentage 
       points of the normal distribution.  26: 118-121.

*/
   double a0=-.322232431088, a1=-1, a2=-.342242088547, a3=-.0204231210245;
   double a4=-.453642210148e-4, b0=.0993484626060, b1=.588581570495;
   double b2=.531103462366, b3=.103537752850, b4=.0038560700634;
   double y, z=0, p=prob, p1;

   p1 = (p<0.5 ? p : 1-p);
   if (p1<1e-20) return (-9999);

   y = sqrt (log(1/(p1*p1)));   
   z = y + ((((y*a4+a3)*y+a2)*y+a1)*y+a0) / ((((y*b4+b3)*y+b2)*y+b1)*y+b0);
   return (p<0.5 ? -z : z);
}

/*********************************************************/

double PointChi2 (double prob, double v)
{
/* returns z so that Prob{x<z}=prob where x is Chi2 distributed with df=v
   returns -1 if in error.   0.000002<prob<0.999998
   RATNEST FORTRAN by
       Best DJ & Roberts DE (1975) The percentage points of the 
       Chi2 distribution.  Applied Statistics 24: 385-388.  (AS91)
   Converted into C by Ziheng Yang, Oct. 1993.
*/
   double e=.5e-6, aa=.6931471805, p=prob, g;
   double xx, c, ch, a=0,q=0,p1=0,p2=0,t=0,x=0,b=0,s1,s2,s3,s4,s5,s6;

   if (p<.000002 || p>.999998 || v<=0) return (-1);

   g = LnGamma (v/2);
   xx=v/2;   c=xx-1;
   if (v >= -1.24*log(p)) goto l1;

   ch=pow((p*xx*exp(g+xx*aa)), 1/xx);
   if (ch-e<0) return (ch);
   goto l4;
l1:
   if (v>.32) goto l3;
   ch=0.4;   a=log(1-p);
l2:
   q=ch;  p1=1+ch*(4.67+ch);  p2=ch*(6.73+ch*(6.66+ch));
   t=-0.5+(4.67+2*ch)/p1 - (6.73+ch*(13.32+3*ch))/p2;
   ch-=(1-exp(a+g+.5*ch+c*aa)*p2/p1)/t;
   if (fabs(q/ch-1)-.01 <= 0) goto l4;
   else                       goto l2;
  
l3: 
   x=PointNormal (p);
   p1=0.222222/v;   ch=v*pow((x*sqrt(p1)+1-p1), 3.0);
   if (ch>2.2*v+6)  ch=-2*(log(1-p)-c*log(.5*ch)+g);
l4:
   q=ch;   p1=.5*ch;
   if ((t=IncompleteGamma (p1, xx, g))<0) {
      printf ("\nerr IncompleteGamma");
      return (-1);
   }
   p2=p-t;
   t=p2*exp(xx*aa+g+p1-c*log(ch));   
   b=t/ch;  a=0.5*t-b*c;

   s1=(210+a*(140+a*(105+a*(84+a*(70+60*a))))) / 420;
   s2=(420+a*(735+a*(966+a*(1141+1278*a))))/2520;
   s3=(210+a*(462+a*(707+932*a)))/2520;
   s4=(252+a*(672+1182*a)+c*(294+a*(889+1740*a)))/5040;
   s5=(84+264*a+c*(175+606*a))/2520;
   s6=(120+c*(346+127*c))/5040;
   ch+=t*(1+0.5*t*s1-b*c*(s1-b*(s2-b*(s3-b*(s4-b*(s5-b*s6))))));
   if (fabs(q/ch-1) > e) goto l4;

   return (ch);
}

/*********************************************************/

/*********************************************************/

int DiscreteGamma (double *freqK, double *rK, 
    double alfa, double beta, int K, int median)
{
/* discretization of gamma distribution with equal proportions in each 
   category
*/
   int i;
   double gap05=1.0/(2.0*K), t, factor=alfa/beta*K, lnga1;

   //printf("Discrete gamma called with median: %d \n", median);

   if(K==1) 
     {
       rK[0] = 1.0;
       return 0;
     }

   if (median) {
      for (i=0; i<K; i++) rK[i]=PointGamma((i*2.0+1)*gap05, alfa, beta);
      for (i=0,t=0; i<K; i++) t+=rK[i];
      for (i=0; i<K; i++)     rK[i]*=factor/t;
   }
   else {
      lnga1=LnGamma(alfa+1);
      for (i=0; i<K-1; i++)
	 freqK[i]=PointGamma((i+1.0)/K, alfa, beta);
      for (i=0; i<K-1; i++)
	 freqK[i]=IncompleteGamma(freqK[i]*beta, alfa+1, lnga1);
      rK[0] = freqK[0]*factor;
      rK[K-1] = (1-freqK[K-2])*factor;
      for (i=1; i<K-1; i++)  rK[i] = (freqK[i]-freqK[i-1])*factor;
   }
   for (i=0; i<K; i++) freqK[i]=1.0/K;
   return (0);
}

