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

#ifndef MATRIXUTILS_H
#define MATRIXUTILS_H
#define NUM_AA 4
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#define PointGamma(prob,alpha,beta)  PointChi2(prob,2.0*(alpha))/(2.0*(beta))

void mcdiv(double ar, double ai, double br, double bi, double *cr, double *ci);
void luinverse(double **inmat, double **imtrx, int size);
void elmhes(double **a, int ordr[], int n);
void eltran(double **a, double **zz, int ordr[NUM_AA], int n);
void hqr2(int n, int low, int hgh, double **h, double **zz, double wr[NUM_AA], double wi[NUM_AA]);
//void hqr21(int n, int low, int hgh, double **h, double **zz, double wr[NUM_AA], double wi [NUM_AA]);
void mcdiv(double ar, double ai, double br, double bi, double *cr, double *ci);

double gammln(double xx);
double LnGamma (double alpha);
double IncompleteGamma (double x, double alpha, double ln_gamma_alpha);
double PointNormal (double prob);
double PointChi2 (double prob, double v);
int DiscreteGamma (double *freqK, double *rK, double alfa, double beta, int K, int median);
#endif
