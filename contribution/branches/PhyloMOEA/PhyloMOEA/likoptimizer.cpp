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

// implement several optimizers

#include "likoptimizer.h"


// traverse the tree edges and optimize it
void Step_LikOptimizer::optimize()
{
	graph::edge_iterator it, tmp;
	
	oldroot = invalid_node;

	// calculate conditional likelihood
	//cout << "likelihood inicial" << Lik_calc->calculate_likelihood() << endl;

	graph::edge_iterator it_end = tree_ptr->TREE.edges_end();
	
	double sum,aux;
	do{
		it = tree_ptr->TREE.edges_begin();
		sum=0;
		while(it != it_end)
		{
			
			// select a new root
			current_edge = *it;	
			a = tree_ptr->istaxon(current_edge.target()) ? current_edge.source() : current_edge.target();
			b = current_edge.opposite(a);

			Lik_calc->a = a; Lik_calc->b = b; Lik_calc->bfocus = current_edge;

			for(int i=0; i<nrates; i++)
			{
				double len = tree_ptr->get_branch_length(current_edge) * Lik_calc->rate_probabilities(i);
				len = len < BL_MIN ? BL_MIN : len;
				p_current[i] = & ((*Lik_calc->probmatrixs)[len]);
			}
			// make the new tree	
			tree_ptr->convert_graph_to_tree(a, NULL);
				
			if( oldroot!=invalid_node)
				update_partials(); //cj( oldroot, a, *it);
			
			// optimize the branch it
			aux = tree_ptr->get_branch_length(current_edge);
			optimize_branch(); //optimize_branch( root, *it);

			//tolerance
			sum += fabs( tree_ptr->get_branch_length(current_edge) - aux );
			oldroot = a;
			++it;
		}
		cout << '.';
		cout.flush();
	}while(sum>=0.0001);
	//cout << "\nlikelihood final" << Lik_calc->calculate_likelihood() << endl;
}

// recalculate the value of c_j for nodes on the path from newroot to oldroot
void Step_LikOptimizer::update_partials() //node *oldroot, node newroot, edge newedge)
{

	// father of oldroot
	node tmp = oldroot;
	node father = current_edge.opposite(a);
	do
	{
		recalculate_cj(tmp, father);
		if(tmp == a) break;
		tmp = tmp.in_edges_begin()->source();
	}while(1);
	//cout << "partials atualizados...." << endl;
	//Lik_calc->sum_site_liks();
}

// recalculate cj when t changes in one of his sons
// only work for trees
void Step_LikOptimizer::recalculate_cj(node n, node father)
{
	int seqlen = tree_ptr->number_of_positions();
	node::out_edges_iterator it;
	node::out_edges_iterator it_end;
	it = n.out_edges_begin();
	it_end = n.out_edges_end();
	node nodeaux;

	for(int i=0; i<seqlen; i++)
	{
		(*Factors)[n][i] = 0;
		for(int r=0; r< nrates; r++)
			for(int k=0; k < 4; k++)
				(*Partials)[n][i*nrates*4+r*4+k] = 1;
	}
	while( it != it_end )
	{
		nodeaux = it->target();
		if(nodeaux !=father)Lik_calc->calculate_node_partial( n, nodeaux, *it);
		it++;
	}
}

void Newton_LikOptimizer::optimize_branch()
{
	//cout << ".....next branch.........." << endl << endl;
	double alpha = 0;
	//*b = current_edge.source() == *a ? current_edge.target() : current_edge.source();
	double branch_l = tree_ptr->get_branch_length(current_edge);

	// calculate derivates of the probability matrix
	for(int i=0; i<nrates; i++)
		p_current[i]->init_derivate();

	calculate_1d2d_loglikelihood(); // calculate the derivates

	//cout << "likelihood calculada ..." << fnext << endl;

	//gsl_cheb_init (lk_aproximated, &F, 0.0, 1.0);

	// newton direction
	//dk = (f1d/f2d>=0 && f1d>=0) || ((f1d/f2d<0 && f1d<0) )
	//	? f1d/f2d : f1d;
	dk = -f1d/f2d;

	alpha = linear_decreasing();
	
	// update branch lenght
// 	if( alpha!=0 )
// 	{
// 		 cout << fnext << "  ";
// 		 cout << "antigo branch " << branch_l << " novo branch " << branch_l + alpha*dk << endl;
// 	}
	tree_ptr->set_branch_length(current_edge,
				branch_l + alpha*dk);		
	
	for(int i=0; i< nrates; i++)
	{
		double factor = Lik_calc->rate_probabilities(i);
		double len = tree_ptr->get_branch_length(current_edge)*factor;
		len = len < BL_MIN ? BL_MIN : len;
		probmatrixs->change_matrix( branch_l*factor, len );
	}
}

double Newton_LikOptimizer::linear_decreasing()
{
	double alpha = 2;
	double fini, pnext;
	double pini = tree_ptr->get_branch_length(current_edge);
	fini = fnext; 
	//fini = gsl_cheb_eval(lk_aproximated,pini);
	int i=0;
	
	while(1)
	{
		alpha*=0.5;
		// next point
		pnext = pini + alpha*dk;

		
		if(pnext >= BL_MIN && pnext <=BL_MAX)
		{
		// modify the probability matrix with the new branch length
		
			//cout << fnext << " --> " << f1d << " --> "  << f2d << "-->" << dk << endl;
			
			fnext = recalculate_likelihood(pnext);
			//cout << pini << "  -->" << pnext << " -->" << fnext << endl;
			//cin >> i;
		}
		if(i==20)
		{
			alpha=0;break; 
		}

		if(fnext > fini && fpclassify(fnext)!= FP_NAN && fpclassify(fnext)!=FP_INFINITE)
		{
			//cout << fini << " " << fnext << " " << alpha << " " << pnext << endl;
			break;	
		}
		i++;
	}
	//if(fnext<fini) alpha = 0;
	
	return alpha;
}


void Newton_LikOptimizer::calculate_1d2d_loglikelihood()
{
	int seqlen = tree_ptr->number_of_positions();

	f1d = f2d = 0;
	double f_h, f1d_h, f2d_h;
	double fnext2 = 0;
	fnext = 0;
	double factor;
	for(int i=0; i < seqlen; i++)
	{
		factor = exp( (*Factors)[a][i] + (*Factors)[b][i] ); // correction factor 
		f_h = sum_partials( i);
		//f_h = Lik_calc->get_site_lik( i);
		f1d_h = sum_partials1d( i) * factor;
		f2d_h = sum_partials2d( i) * factor;
		// first derivate
		f1d += (f1d_h/f_h) * SeqData->pattern_count(i); 
		// second derivate
		f2d += ( (f_h * f2d_h  - f1d_h*f1d_h ) / (f_h*f_h) ) * SeqData->pattern_count(i);
		// likelihood	
		fnext += (log(f_h ) + (*Factors)[a][i] + (*Factors)[b][i] )* SeqData->pattern_count(i); 
	}
	//cout << "experimental ..." << "  " << fnext << " " << fnext2 << endl;
}


