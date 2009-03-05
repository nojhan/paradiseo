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
#include "likelihoodcalculator.h"
#include "treeIterator.h"
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include "utils.h"



#define BL_MIN  1.e-10

//pthread_mutex_t mdone_mutex;

//extern "C" void *lik_slave(void *);

void *lik_slave_class(void *ind)
{
	LikelihoodCalculator *t = (LikelihoodCalculator *)ind;
	if(t->worker == 0)
	{
//		pthread_mutex_lock(&mdone_mutex);
		t->worker = 1;
//		pthread_mutex_unlock(&mdone_mutex);
		t->calculate_partials( t->a, &(t->b));
	}
	else
	{
//		pthread_mutex_lock(&mdone_mutex);
		t->worker = 0;
//		pthread_mutex_unlock(&mdone_mutex);
		t->calculate_partials( t->b, &(t->a));
	}
	return (NULL);
}

void *lik_slave(void *ind) 
{
	LikelihoodCalculator *t = (LikelihoodCalculator *)ind;
	lik_slave_class(t);
 	return NULL;
}


LikelihoodCalculator::LikelihoodCalculator( phylotreeIND &ind, SubstModel &m, ProbMatrixContainer &p, unsigned int nr)
{ 
	if(nr<1)nrates=1;
	else nrates=nr;
	alpha = 2.0;
	part_memory_internal = part_memory_taxons = NULL;
	SeqData = NULL; rates_prob = rates_freq = NULL; prob = NULL;
	model = &m;
	probmatrixs = &p;
	invalid_partials_taxons = true;
	set_data( ind.get_patterns() );
	set_tree(ind);
	build_rates_sites();
}

LikelihoodCalculator::LikelihoodCalculator( phylotreeIND &ind, Sequences &Seqs, SubstModel &m, ProbMatrixContainer &p, unsigned int nr)
{ 
	part_memory_internal = part_memory_taxons = NULL; 
	rates_prob = rates_freq = NULL; prob = NULL;
	if(nr<1)nrates=1;
	else nrates=nr;
	alpha = 2.0;
	SeqData = NULL;
	model = &m;
	probmatrixs = &p;
	invalid_partials_taxons = true;
	set_data( Seqs);
	set_tree(ind);
	build_rates_sites();
}

void LikelihoodCalculator::build_rates_sites()
{
	if(rates_prob!=NULL) delete [] rates_prob;
	if(rates_freq!=NULL) delete [] rates_freq;
	if(prob!=NULL) delete [] prob;

	rates_prob = new double[nrates];
	prob = new double*[nrates]; // points rate matrixs
	rates_freq= new double[nrates];

	if(nrates==1)
	{
		rates_prob[0] = rates_freq[0] = 1.0;
		return;
		
	}
	
/*
	double mean = 0.0;
	
	for (int i = 0; i < nrates; i++)
	{
		rates_prob[i] = gsl_cdf_gamma_Pinv((2.0*i+1.0)/(2.0*nrates), alpha, (1.0)/alpha);
		cout << "quantile : " << (2.0*i+1.0)/(2.0*nrates) << "   " << rates_prob[i] << endl;
		mean += rates_prob[i];
	}
	
	mean = mean/(double) nrates;
	
	cout << "mean : " << mean << endl;

	for (int i = 0; i < nrates; i++)
	{
		rates_prob[i] /= mean;
		rates_freq[i] = 1.0/(double) nrates;
	}
*/
	DiscreteGamma(rates_freq, rates_prob, alpha, alpha, nrates, 0); // 0 = phyml 2.4

/*	for (int i=0; i<nrates; i++) cout << "rate " << i << "  " <<  rates_prob[i] << "  " << rates_freq[i] << endl;
	int k;*/
	//std::cin >> k;

}

// change the default data
void LikelihoodCalculator::set_data( const Sequences &seqs)
{
	// first assignment
	if( SeqData == NULL )
	{
		SeqData = (Sequences *)&seqs;
		allocate_partials(); // first assignment, allocate memoryAlleleString
	}
	else
	// if the data changes, invalidate partial taxons
	{
		if( SeqData != &seqs)
		{
			SeqData = (Sequences *)&seqs;
			deallocate_partials();
			// allocate again
			//cout << "allocating partials" << endl;
			allocate_partials();
			// invalidate partial taxons
			//cout << "warning.... changing patterns..." << endl;
			invalid_partials_taxons = true;
		}
	}
}


void LikelihoodCalculator::set_tree( phylotreeIND &ind )
{
	
	tree_ptr = &ind;
	long total_pos = SeqData->pattern_count();

	set_data( tree_ptr->get_patterns() );

	Partials.init(tree_ptr->TREE);
	graph::node_iterator it = tree_ptr->TREE.nodes_begin();
	graph::node_iterator it2 = tree_ptr->TREE.nodes_end();
	
	// allocate partial matrix for each pattern and internal node (taxon don't have partials)

	for(int i=0, j=0; it!=it2; it++, j++)
	{
		Factors[*it] = part_memory_factors + j * total_pos;
		if(!tree_ptr->istaxon(*it))
		{
			Partials[*it] = part_memory_internal + i * total_pos * nrates* 4; 
			i++;
		}
	}

	graph::edge_iterator ite = tree_ptr->TREE.edges_begin();
	graph::edge_iterator ite2 = tree_ptr->TREE.edges_end();

	for(int i=0; ite!=ite2; i++)
	{
		for(int j=0; j < nrates; j++)
		{
			double len =  tree_ptr->get_branch_length(*ite) * rates_prob[j];
			len = (len < BL_MIN ? BL_MIN : len);
			(part_memory_probmatrix_ptr+i*nrates)[j] = &( (*probmatrixs)[len] );
			edgeprobmatrix[*ite] = part_memory_probmatrix_ptr+i*nrates;
		}
		++ite;
	}
}

double LikelihoodCalculator::calculate_likelihood_exp(edge focus)
{

	double lik;

//	pthread_t threads[2];                /* holds thread info */
//	int ids[2]; 

	worker = 0;


	// select an internal node as root
	bfocus =  focus;
	

	a = tree_ptr->istaxon(bfocus.target()) ? bfocus.source() : bfocus.target();
	b = bfocus.opposite(a);

	//cout << "iniciando partials" << endl;
	tree_ptr->convert_graph_to_tree(a, NULL);
	init_partials();


	// traverse the tree and calculate the partials

	/*for(int i=0; i<2; i++)
		pthread_create(&threads[i], NULL, lik_slave_class, (void*)this);
	for(int i=0; i<2; i++)
		pthread_join(threads[i],NULL);*/
	//cout << "calculando partials..." << endl;
	calculate_partials( a, &b);
	calculate_partials( b, &a);
	//cout << "somando..." << endl;
	// sum all partials
	lik = sum_site_liks();
	return lik;
}


// double LikelihoodCalculator::calculate_all_likelihoods()
// {
// 		oldroot = invalid_node;
// 		graph::edge_iterator it = tree_ptr->TREE.edges_begin();
// 		graph::edge_iterator it_end = tree_ptr->TREE.edges_end();
// 		while(it != it_end)
// 		{
// 			
// 			// select a new root
// 			bfocus = *it;	
// 			a = tree_ptr->istaxon(bfocus.target()) ? bfocus.source() : bfocus.target();
// 			b = bfocus.opposite(a);
// 
// 			for(int i=0; i< nrates; i++)
// 			{
// 				double len = tree_ptr->get_branch_length(bfocus)*rates_prob[i];
// 				len = len < BL_MIN ? BL_MIN : len;
// 				ProbMatrix &p = (*probmatrixs)[ len];
// 				prob[i] = p.p;
// 			}
// 
// 
// 			// make the new tree	
// 			tree_ptr->convert_graph_to_tree(a, NULL);
// 				
// 			if( oldroot!=invalid_node)
// 			{
// 				update_partials(); //cj( oldroot, a, *it);
// 				cout << "likelihood ..." << sum_site_liks() <<endl;
// 			}
// 			else { //first iteration
// 				init_partials();
// 				calculate_partials( a, &b);
// 				calculate_partials( b, &a);
// 				cout << "likelihood ..." << sum_site_liks() << endl;
// 			}
// 
// 			for(int i=0; i< nrates; i++)
// 			{
// 				double factor = rates_prob[i];
// 				double len = tree_ptr->get_branch_length(bfocus);
// 				len = len < BL_MIN ? BL_MIN : len;
// 				probmatrixs->change_matrix( len*factor, tree_ptr->get_branch_length(bfocus)*factor);
// 			}
// 
// 			oldroot = a;
// 			++it;
// 		}
// }

void LikelihoodCalculator::update_partials() //node *oldroot, node newroot, edge newedge)
{

	// father of oldroot
	node tmp = oldroot;
	node father = bfocus.opposite(a);
	do
	{
		recalculate_cj(tmp, father);
		if(tmp == a) break;
		tmp = tmp.in_edges_begin()->source();
	}while(1);
}

// recalculate cj when t changes in one of his sons
// only work for trees
void LikelihoodCalculator::recalculate_cj(node n, node father)
{
	int seqlen = tree_ptr->number_of_positions();
	node::out_edges_iterator it;
	node::out_edges_iterator it_end;
	it = n.out_edges_begin();
	it_end = n.out_edges_end();
	node nodeaux;

	for(int i=0; i<seqlen; i++)
	{
		(Factors)[n][i] = 0;
		for(int r=0; r< nrates; r++)
			for(int k=0; k < 4; k++)
				(Partials)[n][i*nrates*4+r*4+k] = 1;
	}
	while( it != it_end )
	{
		nodeaux = it->target();
		if(nodeaux !=father)calculate_node_partial( n, nodeaux, *it);
		it++;
	}
}


double LikelihoodCalculator::calculate_likelihood()
{

	double lik;

//	pthread_t threads[2];                /* holds thread info */
//	int ids[2]; 

	worker = 0;


	// select an internal node as root
	bfocus =  *(tree_ptr->TREE.edges_begin());
	

	a = tree_ptr->istaxon(bfocus.target()) ? bfocus.source() : bfocus.target();
	b = bfocus.opposite(a);

	
	init_partials();

	// traverse the tree and calculate the partials

	/*for(int i=0; i<2; i++)
		pthread_create(&threads[i], NULL, lik_slave_class, (void*)this);
	for(int i=0; i<2; i++)
		pthread_join(threads[i],NULL);*/
	//cout << "calculando partials..." << endl;
	struct timeval tempo1, tempo2, result;
	gettimeofday(&tempo1, NULL);
	calculate_partials( a, &b);
	calculate_partials( b, &a);
	//cout << "somando..." << endl;
	// sum all partials
	lik = sum_site_liks();
	gettimeofday(&tempo2, NULL);
	timeval_subtract(&result,&tempo2,&tempo1);	
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
	return lik;
}


double LikelihoodCalculator::sum_site_liks(  )
{
	int seqlen = tree_ptr->number_of_positions();
	register double lik = 0;
	register double factor_correct;
	double len;
	
	for(int i=0; i< nrates; i++)
	{
		//len = tree_ptr->get_branch_length(bfocus)*rates_prob[i];
		//len = len < BL_MIN ? BL_MIN : len;
		ProbMatrix *p = edgeprobmatrix[bfocus][i]; //(*probmatrixs)[ len];
		prob[i] = p->p;
	}

	//#pragma omp parallel for private(factor_correct) schedule(dynamic) num_threads(2) reduction(+:lik)
	for(int i=0; i < seqlen; i++)
	{
		factor_correct = Factors[a][i] + Factors[b][i] ;
		site_liks[i] = sum_partials(i);
		//#pragma omp critical
		lik += ( log(site_liks[i]) + factor_correct)* SeqData->pattern_count(i);
	}
	return lik;
}



double LikelihoodCalculator::sum_partials( int pos )
{
	if(tree_ptr->istaxon( b))
	{
		return sum_partials_a_to_taxon( pos );
	}
	else    return sum_partials_a_to_b_notaxon( pos);
}


double LikelihoodCalculator::sum_partials_a_to_taxon( int pos)
{
	double sum = 0;
	unsigned char *meaning;
	char char_state_b = SeqData->pattern_pos( pos, tree_ptr->taxon_id (b) );
	for(int i=0; i<nrates; i++)
	{
		int index = pos*nrates*4 + i*4;
		for(int k=0; k < 4; k++)
		{
			// defined nucleotide
			if( SeqData->is_defined(char_state_b) ) 
				
				sum += rates_freq[i]*SeqData->frequence(k)* prob[i][k*4+char_state_b]* Partials[a][index+k];
			// ambigous nucleotide
			else if (SeqData->is_ambiguous( char_state_b) )
			{
				meaning = SeqData->ambiguos_meaning( char_state_b );
				for(int l=0; l < 4; l++)
					sum += rates_freq[i]*SeqData->frequence(k)* prob[i][k*4+l]* Partials[a][index+k] * meaning[l]; 
						//Partials[b][pos*4+l];
			}
			// gap or undefined
			else 
			{
				for(int l=0; l < 4; l++)
					sum += rates_freq[i]*SeqData->frequence(k)* prob[i][k*4+l]* Partials[a][index+k]; 
			}
		}
	}
	return sum;
}


double LikelihoodCalculator::sum_partials_a_to_b_notaxon( int pos)
{
	double sum = 0;
	
	for(int j=0; j<nrates; j++)
	{
		int index = pos*nrates*4 + j*4;
		for(int k=0; k < 4; k++)
		{
			for(int l=0; l < 4; l++)
			{
					sum += rates_freq[j]*SeqData->frequence(k)* prob[j][k*4+l]
					* Partials[a][index+k] * Partials[b][index+l];
			}
		}
	}
	return sum;
}


// calculate conditional likelihoods 

void LikelihoodCalculator::calculate_node_partial( node father, node son, edge edgeaux)
{
	register double sum;
	int r,i,j;
	//unsigned char l;
	register int seqlen = tree_ptr->number_of_positions();
 	#pragma omp parallel for 
	for(int k=0; k<seqlen;k++)
	{
		long index = k*nrates*4;
		// accumulatre
		Factors[father][k]+=Factors[son][k];
		double corr_factor =   MDBL_MIN;

		for(int r=0; r<nrates; r++)
		{
			double sum = 0;
			//#pragma omp critical
			ProbMatrix *p = edgeprobmatrix[edgeaux][r];

			for(int i=0; i < 4; i++)
			{
				sum = 0;
				if(tree_ptr->istaxon( son))
				{
					unsigned char l=SeqData->pattern_pos( k, tree_ptr->taxon_id( son));
					if(SeqData->is_defined( l) ) sum = p->p_ij_t( i, l );
					else if(SeqData->is_ambiguous( l))
					{
						unsigned char *meaning = SeqData->ambiguos_meaning( l);
						for(int j=0; j < 4; j++)
							sum +=meaning[j]* p->p_ij_t( i, j );	
							//sum +=Partials[son][k*4+j]* p.p_ij_t( i, j );
					}
					else sum = 1;
				}
				else{
					for(int j=0; j < 4; j++)
					{
							sum +=Partials[son][index+ r*4 +j]* p->p_ij_t( i, j );
							
					}
				}
				Partials[father][index + r*4 +i] *= sum;
				corr_factor = ( sum > corr_factor ? sum : corr_factor); 
			}
		}

		if( corr_factor < UMBRAL || corr_factor > (1./LIM_SCALE_VAL))
		{
				//cout << "escalado ..." << endl;
			for(int r=0; r< nrates; r++)
				for(int i=0; i<4; i++)
					Partials[father][index + r*4+ i] /= corr_factor;
			Factors[father][k] += log(corr_factor);
		} 
	}
}


void LikelihoodCalculator::init_partials()
{
	// initialize the values of C for each node	
	long seqlen = tree_ptr->number_of_positions();

	int num_taxons = tree_ptr->number_of_taxons();
	int num_internal_nodes = tree_ptr->TREE.number_of_nodes() - num_taxons;

	long size_part_factors = seqlen*(num_taxons +num_internal_nodes);
	long size_part_memory_internal = nrates*num_internal_nodes*seqlen*4; 
	// correction factors
	//for(long i=0; i<seqlen; i++)
	//		factors[i] = 0;
	for(long i=0; i< seqlen*(num_taxons +num_internal_nodes); i++)
			part_memory_factors[i] = 0;

	// init internal nodes
	for(long i=0; i< size_part_memory_internal; i++)
				part_memory_internal[i] = 1.0;
}

// calculate the values de C_j for all nodes
void LikelihoodCalculator::calculate_partials(node n, node *antecessor)
{
	postorder_Iterator it = tree_ptr->postorder_begin( n, *antecessor);

	while(*it!=n)
	{
		calculate_node_partial( it.ancestor(), *it, it.branch());
		++it;
	}
}



// omp code

double LikelihoodCalculator::calculate_likelihood_omp()
{

	double lik=0;


	// select an internal node as root
	bfocus =  *(tree_ptr->TREE.edges_begin());
	

	a = tree_ptr->istaxon(bfocus.target()) ? bfocus.source() : bfocus.target();
	b = bfocus.opposite(a);

	
	init_partials();

	struct timeval tempo1, tempo2, result;
	

	int seqlen = tree_ptr->number_of_positions();

	for(int i=0; i< nrates; i++)
	{
		//len = tree_ptr->get_branch_length(bfocus)*rates_prob[i];
		//len = len < BL_MIN ? BL_MIN : len;
		ProbMatrix *p = edgeprobmatrix[bfocus][i]; //(*probmatrixs)[ len];
		prob[i] = p->p;
	}


	gettimeofday(&tempo1, NULL);
	
	#pragma omp parallel for reduction(+:lik)
	for(int i=0; i< seqlen; i++)
	{
		calculate_partials_omp( a, &b,i);
		calculate_partials_omp( b, &a,i);
		//cout << "somando..." << endl;
		// sum all partials
		lik += sum_site_liks_omp(i);
	}
	gettimeofday(&tempo2, NULL);
	timeval_subtract(&result,&tempo2,&tempo1);	
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
	return lik;
	
}


double LikelihoodCalculator::sum_site_liks_omp( int pos  )
{
	register double lik = 0;
	register double factor_correct;
	
	//#pragma omp parallel for private(factor_correct) schedule(dynamic) num_threads(2) reduction(+:lik)
	factor_correct = Factors[a][pos] + Factors[b][pos] ;
	site_liks[pos] = sum_partials(pos);
	lik = ( log(site_liks[pos]) + factor_correct)* SeqData->pattern_count(pos);
	return lik;
}


void LikelihoodCalculator::calculate_partials_omp(node n, node *antecessor, int pos)
{
	postorder_Iterator it = tree_ptr->postorder_begin( n, *antecessor);

	while(*it!=n)
	{
		calculate_node_partial_omp( it.ancestor(), *it, it.branch(),pos);
		++it;
	}
}

void LikelihoodCalculator::calculate_node_partial_omp( node father, node son, edge edgeaux, int pos)
{
	register double sum;
	//unsigned char l;
 	//#pragma omp parallel for 
	long index = pos*nrates*4;
		// accumulatre
	Factors[father][pos]+=Factors[son][pos];
	double corr_factor =   MDBL_MIN;

	for(int r=0; r<nrates; r++)
	{
		double sum = 0;
		//#pragma omp critical
		ProbMatrix *p = edgeprobmatrix[edgeaux][r];

		for(int i=0; i < 4; i++)
		{
			sum = 0;
			if(tree_ptr->istaxon( son))
			{
				unsigned char l=SeqData->pattern_pos( pos, tree_ptr->taxon_id( son));
				if(SeqData->is_defined( l) ) sum = p->p_ij_t( i, l );
				else if(SeqData->is_ambiguous( l))
				{
					unsigned char *meaning = SeqData->ambiguos_meaning( l);
					for(int j=0; j < 4; j++)
						sum +=meaning[j]* p->p_ij_t( i, j );	
						//sum +=Partials[son][k*4+j]* p.p_ij_t( i, j );
				}
				else sum = 1;
			}
			else{
				for(int j=0; j < 4; j++)
				{
						sum +=Partials[son][index+ r*4 +j]* p->p_ij_t( i, j );
						
				}
			}
			Partials[father][index + r*4 +i] *= sum;
			corr_factor = ( sum > corr_factor ? sum : corr_factor); 
		}
	}

	if( corr_factor < UMBRAL || corr_factor > (1./LIM_SCALE_VAL))
	{
			//cout << "escalado ..." << endl;
		for(int r=0; r< nrates; r++)
			for(int i=0; i<4; i++)
				Partials[father][index + r*4+ i] /= corr_factor;
		Factors[father][pos] += log(corr_factor);
	} 
}
