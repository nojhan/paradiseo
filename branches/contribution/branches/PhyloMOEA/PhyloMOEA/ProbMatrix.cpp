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

#include "ProbMatrix.h"
#include <cmath>
#include <matrixutils.h>

gsl_cheb_series ** ProbMatrix::cs = NULL;

double f_pij(double x, void *p)
{
	struct params *ij = (struct params *)p;
	ij->pmatrix->set_branch_length(x);
	return ij->pmatrix->p_ij_t(ij->i,ij->j);
}



ProbMatrix::ProbMatrix(SubstModel *m, double bl)
{
	model = m;
	branchlenght = bl;
	calculate_derivate=false;
	p = new double[4*4];
	p1d = new double[4*4];
	p2d = new double[4*4];
	tmpexp = new double[4*4];
}

/*
void ProbMatrix::init()
{
	struct params par;
	double tmpbl = branchlenght;
	int k=0;
	// first call
	gsl_function F;
	F.function = f_pij;
	F.params = &par;
	if(cs == NULL)
	{
		std::cout << "initializaing polynomials...\n";
		cs = new gsl_cheb_series*[16];
		calculatepmatrix();
		// calculate chevchevy polynomials
		for(int i=0; i<4; i++)
			for(int j=0;j<4;j++)
			{
				cs[k] = gsl_cheb_alloc(8);
				par.pmatrix = this;
				par.i = i; par.j = j;
				gsl_cheb_init (cs[k], &F, 0.0, 1.0);
				k++;
			}
	}
	//restore the original branch lenght
	branchlenght = tmpbl;
	calculatepmatrixchev();
}*/



void ProbMatrix::init_derivate()
{
	if(calculate_derivate)return;
	calculate_p1d_matrix();
	calculate_p2d_matrix();
	calculate_derivate=true;
}

// set a new branch lenght and recalculate the probability matrix
void ProbMatrix::set_branch_length(double t)
{
	// save time if t = older branch lenght
	if ( branchlenght != t)
	{	
		branchlenght = t;	
		calculatepmatrix();
		calculate_p1d_matrix();
		calculate_p2d_matrix();
	}
	calculate_derivate=false;
}

// calculate the matrix P = evec*diag( e^eigenvalues )*ievec
void ProbMatrix::calculatepmatrix()
{
	double temp;

	double **xevec = model->eigensystem->eigenvectors();
	double *xeval = model->eigensystem->eigenvalues();
	double **xievec = (double **)model->ievec;

	//diag( e^eigenvalues )*ievec
	for (int i = 0; i < 4; i++)
	{
		temp = exp( branchlenght * xeval[i] );
		for (int j = 0; j < 4; j++) 
			tmpexp[i*4+j] = xievec[i][j]*temp;
	}
	
	// P matrix
	for (int i = 0; i < 4; i++)	
	{
		for (int j = 0; j < 4; j++)	
		{
			temp = 0.0;
			for (int k = 0; k < 4; k++)	
				temp += xevec[i][k] * tmpexp[k*4+j]; 
			p[i*4+j] = fabs(temp);
		}

	}
}

void ProbMatrix::calculatepmatrixchev()
{
	int k=0;
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++) 
		{
			p[i*4+j] = gsl_cheb_eval(cs[i*4+j], branchlenght);
			k++;
		}
}

// matrix Q*P
void ProbMatrix::calculate_p1d_matrix()
{
	double temp;
	double *q=model->rate;
	for (int i = 0; i < 4; i++)	
	{
		for (int j = 0; j < 4; j++)	
		{
			temp = 0.0;
			for (int k = 0; k < 4; k++)	
				temp += q[i*4+k] * p[k*4+j]; 
			p1d[i*4+j] = temp;
		}
	}
}

// calculate Q^2*P
void ProbMatrix::calculate_p2d_matrix()
{
	double temp;
	double *q=model->rate;
	for (int i = 0; i < 4; i++)	
	{
		for (int j = 0; j < 4; j++)	
		{
			temp = 0.0;
			for (int k = 0; k < 4; k++)	
				temp += q[i*4+k]*p1d[i*4+k]; 
			p2d[i*4+j] = temp;
		}
	}
}


