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

#ifndef PROBMATRIX_H
#define PROBMATRIX_H
#include "SubsModel.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_chebyshev.h>
#include <map>
// store a set of the probability matric according some brach-lenght

class ProbMatrix
{
	private:
		
		SubstModel *model;

		double branchlenght;

		double *tmpexp;

		static gsl_cheb_series **cs;

		bool calculate_derivate;
		void calculatepmatrix();
		void calculatepmatrixchev();
		void calculate_p1d_matrix();
		void calculate_p2d_matrix();
		
	public:
		double *p, *p1d, *p2d;
		ProbMatrix( SubstModel *m, double bl );
		~ProbMatrix() { delete [] tmpexp; delete [] p; delete [] p1d; delete [] p2d; }
		inline void init() { calculatepmatrix(); }
		//void init();
		void init_derivate();
		inline double p_ij_t(int i, int j) { return p[i*4+j]; }
		inline double p1d_ij_t(int i, int j) { return p1d[i*4+j]; }
		inline double p2d_ij_t(int i, int j) { return p2d[i*4+j]; }
		void set_branch_length(double l);
};

struct params { ProbMatrix *pmatrix; int i; int j; };

#endif

