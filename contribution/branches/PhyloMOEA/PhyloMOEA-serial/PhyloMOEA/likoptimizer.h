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

#ifndef LIK_OPTIMIZER_H
#define LIK_OPTIMIZER_H
#include "likelihoodcalculator.h"
#include <gsl/gsl_math.h>


// base class for all kind of Likelihood Optimizers

class Lik_Optimizer
{
	protected:
		LikelihoodCalculator *Lik_calc; // point to the calculator
		
	public:
		Lik_Optimizer( LikelihoodCalculator &Lc) { Lik_calc = &Lc; }
		virtual ~Lik_Optimizer() {  };
		// call ot the optimize function
		virtual void optimize() = 0;
};

// branch by branch optimizer

class Step_LikOptimizer:public Lik_Optimizer
{
	protected:
		edge current_edge;
		node a, b, oldroot;
		node invalid_node;
		void update_partials();
		void recalculate_cj(node root, node father);

		Sequences *SeqData;
		phylotreeIND *tree_ptr;
		int nrates;

		ProbMatrixContainer *probmatrixs; // container of probmatrixs
		node_map< double *> *Partials, *Factors; //points to partials

		ProbMatrix **p_current;
	public:
		virtual ~Step_LikOptimizer() { if(p_current!=NULL) delete [] p_current; };

		Step_LikOptimizer( LikelihoodCalculator &Lc) : Lik_Optimizer(Lc) 
		{
			tree_ptr = Lc.tree_ptr;
			probmatrixs = Lc.probmatrixs;
			
			Partials = &Lc.Partials;
			Factors = &Lc.Factors;
			SeqData = Lc.SeqData;
			nrates = Lc.number_rates();
			p_current = new ProbMatrix*[nrates]; // points to the matrixs for each rate
		};
		void optimize();
		virtual void optimize_branch() = 0;
		inline double 	recalculate_likelihood(double tnext)
		{
			for(int i=0; i< nrates; i++)
			{
				double len = tnext*Lik_calc->rate_probabilities(i);
				//len = len < BL_MIN ? BL_MIN : len;
				p_current[i]->set_branch_length(len);
			}
			return Lik_calc->sum_site_liks(); 
		}
		inline double sum_partials(int pos)
		{ 
			for(int i=0; i< nrates; i++)
				Lik_calc->prob[i] = (p_current[i])->p;
			return Lik_calc->sum_partials(pos); 
		}
		inline double sum_partials1d( int pos )	
		{
			for(int i=0; i< nrates; i++)
				Lik_calc->prob[i] = (p_current[i])->p1d;
			return  Lik_calc->sum_partials( pos);
		}
		inline double sum_partials2d( int pos )	
		{
			for(int i=0; i< nrates; i++)
				Lik_calc->prob[i] = (p_current[i])->p2d;
			return  Lik_calc->sum_partials( pos);
		}
};


// Newton Step by Step optimizer:
class Newton_LikOptimizer:public Step_LikOptimizer
{
	private:
		double fnext; //likelihoods
		double f1d, f2d; // derivates
		double dk;	// search direction		
		double linear_decreasing(); 
	public:
		virtual ~Newton_LikOptimizer() {};
		Newton_LikOptimizer(LikelihoodCalculator &Lc): Step_LikOptimizer(Lc) {};
		void calculate_1d2d_loglikelihood();
		void optimize_branch();
};
#endif
