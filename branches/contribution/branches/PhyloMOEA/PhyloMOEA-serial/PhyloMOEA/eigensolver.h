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

#ifndef EIGENSOLVER_H
#define EIGENSOLVER_H

#include <matrixutils.h>
#include <string.h>

// calculate eigenvalues and eigenvectors of a matrix

class EigenSolver
{
	private:
		double **copymatrix;
		double *eval;
		double **evec;
		int dimension;
	public:
		EigenSolver( double *matrix, int dim);
		EigenSolver( double **matrix, int dim);
		~EigenSolver();
		inline double **eigenvectors() { return evec; }
		inline double *eigenvectors(int i) { return evec[ (i>dimension ? 0 : i) ]; }
		inline double *eigenvalues() { return eval; }
		void solve();
};
#endif