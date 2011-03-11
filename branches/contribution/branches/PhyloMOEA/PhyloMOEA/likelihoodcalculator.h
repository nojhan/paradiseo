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
#ifndef LIKELIHOODCALCULATOR_H
#define LIKELIHOODCALCULATOR_H
#include <phylotreeIND.h>
#include <SubsModel.h>
#include <probmatrixcontainer.h>
//#include <pthread.h>
#include <tree_limits.h>
/**
@author Waldo Cancino
*/


class LikelihoodCalculator{
private:
	edge bfocus;	// branch focus
	node a,b;	// bfocus vertices (b can be a taxon, a never can be)
	double **prob;	// point to the probability matrix associated to bfocus
	int worker;		// for threading, not used really
	int nrates;
	double *rates_prob; // rate probabilities
	double *rates_freq; // rate frequences
	double alpha;
	node root; // root of the likelihood calculator	
	node oldroot, invalid_node; // old root and first root when change the focus of the likelihood calculations

	// continous allocated memory to store partial results
	double *part_memory_internal, // conditional likelihood of internal nodes 
		*part_memory_taxons,  // conditional likelihood of internal nodes, no longer used
		*part_memory_factors; // correction factors for nodes
	ProbMatrix **part_memory_probmatrix_ptr;
	
	double *site_liks; // site likelihoods

	// maps nodes to the continous memory	
	node_map< double *> Partials; // maps nodes to the corresponding conditional likelihood 
	node_map<double *> Factors;   // maps node to the corresponding correct factors
	edge_map<ProbMatrix **> edgeprobmatrix; // maps the edge to the corresponding probability matrixs

	// external data	
	phylotreeIND *tree_ptr; // point to the tree
	SubstModel *model; // substitution model
	ProbMatrixContainer *probmatrixs; // container of probmatrixs
	Sequences *SeqData; // sequence data

	
	bool invalid_partials_taxons; // when received a new tree, the partials are invalidated
	
	// init partials memory
	void init_partials();
	
	// prepare the post-order tree iterator
	void calculate_partials(node n, node *);
	void calculate_partials_omp(node n, node *, int pos);
	// calculate conditional likelihood for the node father, from the son conditionals
	void calculate_node_partial( node father, node son, edge edgeaux);

	void calculate_node_partial_omp( node father, node son, edge edgeaux, int pos);
	// likelihood sum of partial for the focus branch
	double sum_partials( int pos);
	double sum_partials_a_to_taxon( int pos );
	double sum_partials_a_to_b_notaxon( int pos);

	// sum the site likelihoods 
	double sum_site_liks();
	double sum_site_liks_omp(int pos);

	// allocate partial memory
	void allocate_partials()
	{
		long total_pos = SeqData->pattern_count();
		int ntaxons = SeqData->num_seqs();
		//int nedges = tree_ptr->TREE.number_of_edges();
		unsigned long tamanho = (ntaxons-2) * nrates * total_pos * 4;
		cout << "tamanho de alocacao:" << tamanho << endl;
		part_memory_internal = new double[ tamanho  ];
		tamanho = (2*ntaxons-2)  * total_pos;
		cout << "tamanho de alocacao:" << tamanho << endl;
		part_memory_factors = new double [ tamanho ];
		part_memory_probmatrix_ptr = new ProbMatrix* [ (2*ntaxons-3) * nrates ] ;
		site_liks = new double[total_pos];
		cout << "allocating done..." << endl;
	}

	// destroy partial memory
	void deallocate_partials()
	{
		delete [] part_memory_probmatrix_ptr;
		delete [] part_memory_internal;
		delete [] part_memory_factors;
		delete [] site_liks;
	}

	

	// deprecated
	double sum_partials2( node a, node b, edge edgeaux, int pos, int which=0);
	void build_rates_sites();

public:
	bool corrected;
	LikelihoodCalculator( phylotreeIND &ind, SubstModel &m, ProbMatrixContainer &p, unsigned int nr=1);
	LikelihoodCalculator( phylotreeIND &ind, Sequences &Seqs, SubstModel &m, ProbMatrixContainer &p, unsigned int nr=1);

	// main functions, prepare the object to calculate likelihood
	double calculate_likelihood();
	double calculate_likelihood_omp();
	double get_site_lik(int i) { return site_liks[i]; }
	double calculate_likelihood_exp(edge focus);
	double calculate_all_likelihoods();
	void update_partials();
	void recalculate_cj(node n, node father);
	void set_alpha(double a) { alpha = a; build_rates_sites(); };
	// change the tree '
	void set_tree( phylotreeIND &ind );
	double rate_probabilities(int i) { return rates_prob[i]; }
	int number_rates() { return nrates; }

	
	~LikelihoodCalculator()
	{
		//cout << "deallocando partialas" << endl;
		deallocate_partials();
		//cout << "deallocando o restante" << endl;
		if(prob!=NULL) delete[] prob;
		if(rates_prob!=NULL) delete [] rates_prob;
		if(rates_freq!=NULL) delete [] rates_freq;
	}

	// change the dataset
	void set_data( const Sequences &seqs);
	
	friend void* lik_slave_class(void *);

	// frind class to do optimizations
	friend class Step_LikOptimizer;
};
#endif
