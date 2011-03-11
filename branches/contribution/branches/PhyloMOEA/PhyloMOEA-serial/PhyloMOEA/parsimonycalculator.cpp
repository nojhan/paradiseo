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
#include "parsimonycalculator.h"
#include "treeIterator.h"
#include <fstream>

ParsimonyCalculator::ParsimonyCalculator(phylotreeIND &t)
{
	set_internal_memory_allocate = set_taxon_memory_allocate = NULL;
	char_internal_memory_allocate = char_taxon_memory_allocate = NULL;
	set_memory_allocate=NULL;
	char_memory_allocate=NULL;
	parsimony = 0;
	tree_ptr = &t;
	invalid_set_taxons = true;
	SeqData = (Sequences*)&(t.get_patterns());
	set_tree( t);
}


ParsimonyCalculator::~ParsimonyCalculator()
{
	delete [] set_internal_memory_allocate;
	delete [] set_taxon_memory_allocate;
	delete [] char_internal_memory_allocate;
	delete [] char_taxon_memory_allocate;
	delete [] set_memory_allocate;
	delete [] char_memory_allocate;
}



// change the tree for calculate parsimony
void ParsimonyCalculator::set_tree(phylotreeIND &t)
{

	int ntaxons = t.number_of_taxons();
	

	if( set_internal_memory_allocate != NULL )
	{
		Sequences *new_data = (Sequences *)&(t.get_patterns());
		
		if( SeqData!=new_data)
		{
			SeqData = new_data;
			delete [] set_internal_memory_allocate;
			delete [] set_taxon_memory_allocate;
			delete [] char_internal_memory_allocate;
			delete [] char_taxon_memory_allocate;
			//cout << "warning.... changing patterns..." << endl;
			invalid_set_taxons = true;
			set_internal_memory_allocate = new unsigned char[ (2*ntaxons-2) * SeqData->infsite_count() * 5 ];
			set_taxon_memory_allocate = new unsigned char[ ntaxons*SeqData->infsite_count()*5];
			char_internal_memory_allocate = new unsigned char[ (2*ntaxons-2) * SeqData->infsite_count()];
			char_taxon_memory_allocate = new unsigned char[ ntaxons * SeqData->infsite_count()];
		}
		
	}
	else // first assignment, allocate memory
	{
		set_internal_memory_allocate = new unsigned char[ (2*ntaxons-2) * SeqData->infsite_count() * 5 ];
		set_taxon_memory_allocate = new unsigned char[ ntaxons*SeqData->infsite_count()*5];
		char_internal_memory_allocate = new unsigned char[ (2*ntaxons-2) * SeqData->infsite_count()];
		char_taxon_memory_allocate = new unsigned char[ ntaxons * SeqData->infsite_count()];
	}

	tree_ptr = &t;

	set_internal.init(tree_ptr->TREE);
	char_internal.init(tree_ptr->TREE);

	graph::node_iterator it = tree_ptr->TREE.nodes_begin();
	graph::node_iterator it2 = tree_ptr->TREE.nodes_end();

	// initialize internal sets
	for(int i=0 ; it!=it2; it++)
	{
		if(tree_ptr->istaxon(*it))
		{
			set_internal[*it] = set_taxon_memory_allocate + tree_ptr->taxon_id(*it) * SeqData->infsite_count() * 5; 
			char_internal[*it] = char_taxon_memory_allocate + tree_ptr->taxon_id(*it) * SeqData->infsite_count(); 
		}
		else
		{
			set_internal[*it] = set_internal_memory_allocate + i * SeqData->infsite_count() * 5; 
			char_internal[*it] = char_internal_memory_allocate + i * SeqData->infsite_count(); 
			i++;
		}
	}
}


void ParsimonyCalculator::init_sets_chars()
{
	int total_nodes = tree_ptr->TREE.number_of_nodes();
	int ntaxons = tree_ptr->number_of_taxons();
	int num_inf_sites = SeqData->infsite_count();
	unsigned char *set_taxon, *char_taxon, l;

	// init the internal sets
	memset( set_internal_memory_allocate, 1, (2*ntaxons-2)* SeqData->infsite_count() * 5);
	if(!invalid_set_taxons)return;	
	//cout << "warning... inicializando taxons parcimonia..." << endl;
	// init internal set and character for taxons
	memset( set_taxon_memory_allocate, 0, ntaxons*SeqData->infsite_count()*5*sizeof(unsigned char));
	

	for(int k=0; k<ntaxons; k++)
	{
		set_taxon = set_taxon_memory_allocate + k * num_inf_sites * 5; 
		char_taxon = char_taxon_memory_allocate + k * num_inf_sites; 
		for(int j=0; j< num_inf_sites; j++)
		{
			l = SeqData->infsite_pos(j, k);
			// '?' states may be any state
			if ( SeqData->is_ambiguous(l) || SeqData->is_gap(l) || SeqData->is_undefined(l) )
			{
				unsigned char *meaning = SeqData->ambiguos_meaning(l);
				for(int m=0; m<5; m++)set_taxon[j*5+m] = meaning[m];
			}
			else set_taxon[j*5+l] = 1;
			char_taxon[j] = l;
		}
	}
	invalid_set_taxons = false;
}

// initialize set taxon and characters of taxon
void ParsimonyCalculator::init_set_char_taxon(node n)
{
	int num_inf_sites = SeqData->infsite_count();
	unsigned char l;
	// set for taxaon are set to 0
	memset( set_internal[n], 0, num_inf_sites*5*sizeof(unsigned char));
	// informative sites
	int taxon_id = tree_ptr->taxon_id(n);
	for(int j=0; j< num_inf_sites; j++)
	{
		l = SeqData->infsite_pos(j, taxon_id);
		//pattern_pos(*it, taxon_id);
		// '?' states may be any state
		if ( SeqData->is_ambiguous(l) || SeqData->is_gap(l) || SeqData->is_undefined(l) )
		{
			unsigned char *meaning = SeqData->ambiguos_meaning(l);
			for(int k=0; k<5; k++)set_internal[n][j*5+k] = meaning[k];
		}
		else set_internal[n][j*5+l] = 1;
		char_internal[n][j] = l;
	}
}


// calculate the informative sites in the patterns;
void ParsimonyCalculator::save_informative(char *fn)
{
	//std::vector<struct PatternInfo> &vec_patterns = patterns->get_patterns();
	//const Sequences &patterns = tree_ptr->get_patterns();
	char nucleotide;
	string sequence;
	ofstream salida(fn, ios::out);

	for(int j=0; j< tree_ptr->number_of_taxons(); j++)
	{
		sequence.clear();
		for(int i=0 ; i < SeqData->infsite_count(); i++)
		{
			int l = SeqData->infsite_pos(i,j);
			switch(l)
			{
				case 0: nucleotide = 'A'; break;
				case 1: nucleotide = 'C'; break;
				case 2: nucleotide = 'G'; break;
				case 3: nucleotide = 'T'; break;
			}
			for(int k=0; k< SeqData->infsite_count(i);k++)
				sequence += nucleotide;
		}	
		salida.setf(ios::left);
		salida.width(20);
		salida << SeqData->seqname(j) << "\t" << sequence << '\n';
	}
	salida.close();
}


// calculate the intersection of two sets returning the number of
// intersected elements;
int ParsimonyCalculator::set_intersection( unsigned char *a, unsigned char *b, unsigned char *result)
{
	int sum = 0;
	for(int i=0; i<5; i++)
		sum += (result[i] = a[i] && b[i]);
	return sum;
}	


// calculate the union of two sets returning the number of
void ParsimonyCalculator::set_union( unsigned char *a, unsigned char *b, unsigned char *result)
{
	for(int i=0; i<5; i++)
		result[i] = a[i] || b[i];
	
}	



// calculate the parsimony between two sets
int ParsimonyCalculator::set_parsimony( unsigned char *a, unsigned char *b, unsigned char *result)
{
	int intersected = set_intersection(a, b, result);
	if(intersected == 0)
	{
		// no common characters, increase parsimony
		set_union(a,b,result);
		return 1;
	}
	return 0; // intersection, parsimony value remains equal	
}


// calculate a parsimony between the sets of the father and the children
// for all relevant sites
int ParsimonyCalculator::node_parsimony( node a, node b, unsigned char *result)
{
	// calculate parsimony for taxon child, just union
	int sum_parsy = 0;
	int num_inf_sites = SeqData->infsite_count();
	for(int j=0; j< num_inf_sites; j++)
		sum_parsy += set_parsimony( &set_internal[a][j*5], &set_internal[b][j*5], &result[j*5]) *  SeqData->infsite_count(j);	
	return sum_parsy;
}


// first stage of fitch algorithm
void ParsimonyCalculator::fitch_post_order(node n, node *antecessor)
{
	postorder_Iterator it = tree_ptr->postorder_begin( n, *antecessor);
	//it.begin();
	while(*it!=n)
	{
		unsigned char tmpresult[ SeqData->infsite_count()*5];
		parsimony += node_parsimony( it.ancestor(), *it , tmpresult);
			// copy the results to the node father and continue calculating for another nodes
		memcpy( set_internal[it.ancestor()], tmpresult, SeqData->infsite_count()*5*sizeof(unsigned char) );
		++it;
	}
}


// sequence assignment from ancestor sequence
void ParsimonyCalculator::seq_assignment(node n, node ancestor)
{
	int num_inf_sites = SeqData->infsite_count();
	unsigned char parent_char;
	for(int i=0; i< num_inf_sites; i++)	
	{
		parent_char = char_internal[ancestor][i];
		if( set_internal[n][i*5+parent_char] ) char_internal[n][i] = parent_char;
		else
		{
			// get the first character in set
			int j = 0;
			while(!set_internal[n][i*5+j]) j++;
			char_internal[n][i] = j;
		}
	}
}


// phase II of Fitch algorithms: pre-order (internal node sequence assignment)
void ParsimonyCalculator::fitch_pre_order(node n, node *antecessor)
{
	node nodeaux;
    // ignore the taxons
	
	if(tree_ptr->istaxon(n)) return;
	else  seq_assignment( n, *antecessor);
	node::inout_edges_iterator it;
	node::inout_edges_iterator it_end;
	it = n.inout_edges_begin();
	it_end = n.inout_edges_end();
	while( it != it_end )
	{
		if(antecessor==NULL || ( it->source()!=*antecessor  && it->target()!=*antecessor) )   
		{
			nodeaux = it->source() == n ? it->target() : it->source();	
			fitch_pre_order( nodeaux, &n);
		}
		it++;
	}
}

long int ParsimonyCalculator::fitch()
{
	node root_aux;
	
	
	root_aux = tree_ptr->taxon_number(0);

	edge edgeaux = *(root_aux.in_edges_begin()); 
	node nodeaux = edgeaux.source();

	parsimony = 0;
	init_sets_chars();

	// if is an unrooted tree, assign an taxa as root to calculate
	// parismony (see PAUP)
	unsigned char tmp[ SeqData->infsite_count() * 5 ];
	fitch_post_order(nodeaux, &root_aux);
	parsimony += node_parsimony( root_aux, nodeaux, tmp );
	//fitch_pre_order(nodeaux, &root_aux);
	return parsimony;
	//cout << "parsimonia total:" << parsimony;
}	
