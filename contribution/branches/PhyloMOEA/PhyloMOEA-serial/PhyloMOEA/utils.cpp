/***************************************************************************
 *   Copyright (C) 2008 by Waldo Cancino   *
 *   wcancino@icmc.usp.br   *
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

#include <eo>
#include <moeo>
#include <iostream>
#include <fstream>
#include <likoptimizer.h>
#include <utils.h>

extern gsl_rng *rn2;
extern RandomNr *rn;
//Sequences *seq;
extern long seed;
//vector<phylotreeIND> arbores;
extern string datafile,usertree, expid, path;
extern double pcrossover, pmutation, kappa, alpha;
extern unsigned int ngenerations,  popsize, ncats;

int   timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y)
{
/* Perform the carry for the later subtraction by updating y. */
	if (x->tv_usec < y->tv_usec) {
		int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
		y->tv_usec -= 1000000 * nsec;
		y->tv_sec += nsec;
	}

	if (x->tv_usec - y->tv_usec > 1000000) {
		int nsec = (x->tv_usec - y->tv_usec) / 1000000;
		y->tv_usec += 1000000 * nsec;
		y->tv_sec -= nsec;
	}

	/* Compute the time remaining to wait.
	tv_usec is certainly positive. */
	result->tv_sec = x->tv_sec - y->tv_sec;
	result->tv_usec = x->tv_usec - y->tv_usec;
	
	/* Return 1 if result is negative. */
	return x->tv_sec < y->tv_sec;
}


void print_elapsed_time(struct timeval *x, struct timeval *y)
{
	struct timeval result;
	timeval_subtract(&result,y,x);	
	long remainder = result.tv_sec % 3600;
	long hours = (result.tv_sec - remainder)/3600;
	long seconds = remainder % 60;
	long minutes = (remainder - seconds) / 60;
	cout << "Execution time :  ";
	cout.width(3);
	cout.fill(' ');
	cout << hours << ":";
	cout.width(2);
	cout.fill('0');
	cout << minutes << ":";
	cout.width(2);
	cout.fill('0');
	cout << seconds << "." << result.tv_usec << "(" << result.tv_sec << ")" << endl;
}

void print_elapsed_time_short(struct timeval *x, struct timeval *y, ostream &os)
{
	struct timeval result;
	timeval_subtract(&result,y,x);	
	os << "  " << result.tv_sec << "." << result.tv_usec;
}


void print_cpu_time(clock_t start, clock_t end)
{
	double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	cout << "   " << cpu_time_used << "   ";
}


void welcome_message()
{
	cout << "\nPhyloMOEA, a program for multi-criteria phylogenetic inference\n";
	cout << "using maximum parsimony and maximum likelihood\n";
	cout << "Version 0.2 (ParadisEO backend) (c) 2009, Waldo Cancino";
	cout << "\n";
}

void save_exp_params(ostream &of=cout)
{
	of << "PhyloMOEA Experiment Parameters" << endl;
	of << "--------------------------------------------" << endl;
	of << "Sequence Datafile    : " << datafile << endl;
	of << "Initial Trees File   : " << usertree << endl;
	of << "N. Generations       : " << ngenerations << endl;
	of << "Population Size      : " << popsize << endl;
	of << "Crossover Rate       : " << pcrossover << endl;
	of << "Mutation Rate        : " << pmutation << endl; 
	of << "Discrete-Gamma Categs: " << ncats << endl;
	of << "Gamma Shape          : " << alpha << endl;
	of << "HKY85+GAmma Kappa    : " << kappa << endl;
	of << "Experiment ID        : " << expid << endl;
	of << "Ramdom Seed          : " << seed << endl;
	of << "Working Path         : " << path << endl;
}



void optimize_solutions( eoPop<PhyloMOEO> &pop, LikelihoodCalculator &lik_calc)
{
	int n = pop.size();
	for(int i=0; i<n; i++)	
	{
		phylotreeIND &sol = pop[i].get_tree();
		lik_calc.set_tree(sol);
		cout << "\noptimizaing tree " << i+1 << " of " << n;
//		cout << endl << "likelihood inicial:" << lik_calc->calculate_likelihood() << endl;
		Newton_LikOptimizer test(lik_calc);
		test.optimize();
		
		pop[i].invalidate();
		//lik_calc->maximizelikelihood();
		//lik_calc->set_tree(*sol);
//		lik_calc->set_tree( *sol);
//		cout << endl << "likelihood final:" << lik_calc->calculate_likelihood() << endl;
	}
}


void readtrees(const char *fname, eoPop<PhyloMOEO> &poptree)
{
	int ntrees;
	string s;
	fstream in_file;
	try{

		in_file.open(fname, ios::in);
		if(!in_file.is_open())
		{
			cout << fname << endl;
			throw ExceptionManager(10);
		}
		poptree.readFrom( in_file );
		return;
	}
	catch(ExceptionManager e)
	{
		e.Report();
	}
	in_file.close();
}




