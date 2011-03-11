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


#include "eigensolver.h"

EigenSolver::EigenSolver(double *matrix, int dim)
{
	// allocate copy matrix
	copymatrix = new double*[dim];
	evec = new double*[dim];
	eval = new double[dim];
	dimension = dim;
	for(int i=0; i<dim; i++)
	{
		copymatrix[i] = new double[dim];
		evec[i] = new double[dim];
		for(int j=0; j<dim; j++)
			copymatrix[i][j] = matrix[i*dim+j];
	}
}

EigenSolver::EigenSolver(double **matrix, int dim)
{
	// allocate copy matrix
	copymatrix = new double*[dim];
	evec = new double*[dim];
	eval = new double[dim];
	dimension = dim;
	for(int i=0; i<dim; i++)
	{
		copymatrix[i] = new double[dim];
		evec[i] = new double[dim];
		memcpy( copymatrix, matrix, dim*sizeof(double) );
	}
}
			
EigenSolver::~EigenSolver()
{
	// deallocate matrices
	for(int i=0; i<dimension; i++)
	{
		 delete [] evec[i];
		 delete [] copymatrix[i];
	}
	delete [] eval;
	delete [] evec;
	delete [] copymatrix;
}

void EigenSolver::solve()
{
	int ordr[dimension];
	double evali[dimension];
	// resolve eigensystem
	elmhes( copymatrix, ordr, dimension);
	eltran( copymatrix, evec, ordr, dimension);
	hqr2( dimension, 1, dimension, copymatrix, evec, eval, evali);
}
		
