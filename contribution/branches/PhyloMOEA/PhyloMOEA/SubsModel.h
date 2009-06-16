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

#ifndef SUBSMODEL_H
#define SUBSMODEL_H
// substitution model class
// only nucleotides
#include "eigensolver.h"
#include "Patterns.h"
#include <iostream>

class SubstModel
{
	private:
			
		// instantaneus rate matrix 4x4
		double rate[16]; // rate matrix
		double **ievec; // inverse eigenvectors
		double *frequences;
		double a,b,c,d,e; // model parameterss
		double kappa;
		int model;
		Sequences *patterns;
		EigenSolver *eigensystem;
		// define the substituion model
		void construct_rate_matrix();
		void setsubstmodel( int );
		void calculatepmatrix(double);
		void mult_frequences();
		void set_diagonal();
		void normalize();
		void set_equal_frequences();
		void init_rate_matrix();
		void f81();
		void k2p();
		void jc69();
		void hky85();
		void gtr();
		inline double get_rate(int i, int j) { return rate[4*i+j]; }
		inline void set_rate( int i, int j, double val) { rate[4*i +j] = val; }
	public:
		enum models { JC69, F81, K2P, HKY85, GTR };
		friend class ProbMatrix;
		~SubstModel() { delete eigensystem; for(int i=0; i<4; i++) delete[] ievec[i]; delete [] ievec; }
		SubstModel(Sequences &p, int m = 0);
		void init();
		inline void  set_kappa(double t) { kappa = t; init(); }
		inline void  set_param(double *p) { a = p[0]; b = p[1]; c = p[2]; d = p[3];  e = p[4]; init(); }
		inline void  set_a(double p) { a = p; }
		inline void  set_b(double p) { b = p; }
		inline void  set_c(double p) { c = p; }
		inline void  set_d(double p) { d = p; }
		inline void  set_e(double p) { e = p; }	      
		inline double get_a() { return a; }
		inline double get_b() { return b; }
		inline double get_c() { return c; }
		inline double get_d() { return d; }
		inline double get_e() { return e; }
		inline double get_kappa() { return kappa; }
		void print_rate_matrix();
};
#endif






