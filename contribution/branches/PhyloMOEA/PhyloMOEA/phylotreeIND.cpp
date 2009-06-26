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

#include <phylotreeIND.h>
#include <iostream>
#include <map>
#include <sstream>
#include <gsl/gsl_randist.h>
#include <cmath>
#include <iterator>
#include <ExceptionManager.h>

using namespace std;


struct temp_info
{
	int left, right, num_nodes;
	node left_most;
	temp_info() {};
	temp_info(int n) : left(n), right(-1), num_nodes(0) {};
};



// return the element at position pos of the list
template <typename T> const T& select_edge_at_pos( const list <T> &l, int pos) 
{
	if(pos >= l.size())
		return l.back();
	typename list<T>::const_iterator it =l.begin();
	for( int i=0 ; i < pos; i++)it++;
	return *it;
}

template const int& select_edge_at_pos<int>( const list <int> &, int );


double phylotreeIND::randomgsl_alpha = 500;

// ---------------- constructors

// constructor		

phylotreeIND::~phylotreeIND()
{
	graph::edge_iterator it = TREE.edges_begin();
	graph::edge_iterator it_e = TREE.edges_end();
	if(split_bits!=NULL)delete [] split_bits;
}

phylotreeIND::phylotreeIND( RandomNr *g, Sequences &p, gsl_rng *gsr) : 
				random_gsl(gsr), randomNr(g)
{
	seqpatterns = &p;
	root = NULL;
	valid_splits = false;
	split_bits = NULL;

	// init maps

	MAPNODETAXON.init(TREE);
	
	nnode = p.num_seqs();
	MAPTAXONNODE.resize(nnode);
	MAPIDEDGE.resize( 2*nnode - 3);
	NJDISTANCES.resize( 2*nnode - 3);
	//split_bits = new bool [ (2*nnode-3) * (2*nnode-2)];
	// crete the taxons
	for(int i=0; i < nnode; i++) 
	{
		new_taxon(i);
	}
}

// copy constructor
phylotreeIND::phylotreeIND( const phylotreeIND &org) : random_gsl( org.random_gsl), randomNr(org.randomNr)
{
	// copy the data of IND class
	split_bits = NULL;
	copy(org);	
}


// copy an individual
void phylotreeIND::copy( const phylotreeIND &C ) 
{
	root = C.root;
	parsimony = C.parsimony;
	valid_splits = false;
	//	likelihood = C.likelihood;
	nnode = C.nnode;
	seqpatterns = C.seqpatterns;
	// map nodes and exes
	list<node> cnodes = C.TREE.all_nodes();
	list<edge> cedges = C.TREE.all_edges();
	if(split_bits != NULL) delete [] split_bits;
	split_bits = new bool [ (2*nnode-3) * (2*nnode-2)];

	TREE.clear();
	
	MAPNODETAXON.init(TREE);
	MAPIDEDGE.resize( 2*nnode -3);
	NJDISTANCES.resize( 2*nnode -3 );
	MAPTAXONNODE.resize( nnode );

	construct_graph( C, cnodes, cedges);
	// if is a tree constructed, calculate the splits
	if( nnode < TREE.number_of_nodes() ) 
		this->calculate_splits();
}



// clone functions

// copy the entire individual
phylotreeIND* phylotreeIND::clone() const 
{ 

	phylotreeIND* p=new phylotreeIND(*this); 
	return p; 
}


// copy only the taxons and perform an stepwise addition
phylotreeIND* phylotreeIND::randomClone() const 
{
	// clone the individual (only taxons)
	phylotreeIND* p=new phylotreeIND( *this); 
	p->stepwise_addition();
	// set the branch lenghts
	graph::edge_iterator it = p->TREE.edges_begin();
	graph::edge_iterator it_e = p->TREE.edges_end();
	
	// init the edge lenghts
	while( it != it_e)
	{
		p->set_branch_length(*it, randomNr->doubleuniformMinMax(0, 0.05));
		++it;
	}
	return p;
}

// change the data set

void phylotreeIND::set_data( Sequences &s)
{
	if ( s.num_seqs() != seqpatterns->num_seqs() )
	{
		cout << "error --> incompatible data change\n";
		exit(1);
	}
	else if(seqpatterns!=&s)
	{
		//cout << "data change on individual...\n";
		seqpatterns = &s;
	}
}

// create an internal node
node phylotreeIND::new_internal_node()
{
	node n = TREE.new_node();
	//assert( n.id() < 2*nnode-2);
	MAPNODETAXON[n] = -1;
	return n;
}
// create a node that maps the sequence id
node phylotreeIND::new_taxon( int id)
{
	node n = new_internal_node();
	//node n = TREE.new_node();
	//assert( n.id() < 2*nnode-2);
	MAPNODETAXON[n] = id;
	MAPTAXONNODE[id] = n;
	return n;
}

edge phylotreeIND::new_branch( node source, node dest)
{
	edge edgeaux = TREE.new_edge(source, dest);
	MAPIDEDGE[edgeaux.id()] = edgeaux;
	return edgeaux;
}

edge phylotreeIND::new_branch( node source, node dest, double bl)
{
	edge edgeaux = new_branch(source, dest);
	set_branch_length(edgeaux, bl);
	return edgeaux;
}

void phylotreeIND::remove_branch( edge edgeaux)
{
	MAPIDEDGE[edgeaux.id()] = invalid_edge;
	TREE.del_edge(edgeaux);
}

// select edge and a pos
edge phylotreeIND::select_edge() const
{
	int j;
	edge edgeaux;
	do
	{
		j =randomNr->uniform0Max(MAPIDEDGE.size());
		edgeaux = MAPIDEDGE[j];
	}while (edgeaux == invalid_edge);
	return edgeaux;
}

// select an edge off-side subtree defined by source_edge
edge  phylotreeIND::select_edge_outsidetree( edge source_edge) const
{
	return choose_edge_fromside( source_edge.id() * TREE.number_of_nodes(), false);
}


// select and edge from a side of a split
edge phylotreeIND::choose_edge_fromside( int idx, bool side ) const
{
	edge edgeaux;
	do
	{
		edgeaux = select_edge();
		assert( edgeaux.source().id() < 2*nnode-2);
		assert( edgeaux.target().id() < 2*nnode-2);
	}
	while( split_bits[ idx + edgeaux.source().id()] !=side  && 
		split_bits[ idx + edgeaux.target().id()] != side);
	return edgeaux;
}


// select and edge from a side of a split
// true = inside
// false = outside
edge phylotreeIND::choose_edge_fromside_2( struct split_info &info, bool inside ) const
{
	edge edgeaux;
	bool chosen = false;
	do
	{
		edgeaux = select_edge();
		struct split_info *info2 = interior_edge[edgeaux];
		if( is_internal(edgeaux) )
		{
		      int taxon_map = splitstable[ taxon_id(edgeaux.target()) ].map_to_node;
		      if( inside)
			  chosen = ( taxon_map  >= info.left  && taxon_map <= info.right );
		      else
			  chosen = ( taxon_map  < info.left  || taxon_map > info.right );
		}
	        else 
		{
		      if ( inside )
			  chosen = ( info.left <= info2->left && info.right >= info2->right );
		      else
			  chosen = ( info.left > info2->right || info.right < info2->left );
		}

	}
	while( !chosen );
	return edgeaux;
}



// genetic operators
// change subtrees to form childs
void phylotreeIND::crossover(float pcross, const phylotreeIND& dad, phylotreeIND*& sis, phylotreeIND*& bro) const
{
	// prevent discard qualifiers erros (because const function)
	phylotreeIND &mom = (phylotreeIND &)(*this);
	phylotreeIND &dady = (phylotreeIND &)dad;
	sis = dady.clone();
	bro = mom.clone();
	phylotreeIND &sister = (phylotreeIND &)(*sis);
	phylotreeIND &brother = (phylotreeIND &)(*bro);
	if( randomNr->flipP(pcross) )
	{
		// swap subtreesf
		mom.export_subtree( sister );
		dady.export_subtree( brother);
		sister.invalidate_splits();
		brother.invalidate_splits();
	}
}

void phylotreeIND::mutate(float pmut)
{
	double type = randomNr->ran2();
	//double type = 0.5;
	if( !splits_valid() ) calculate_splits();
	if( randomNr->flipP(pmut) )
	{
		// both kinds of mutatio if type>0.67
		// topological mutation
		if( type < 0.34 || type > 0.67 )
		{
			//TBR();
		  	//change_subtrees();
			NNI();	
			//SPR();
			invalidate_splits();
			
		// branch mutation
		}
		if( type > 0.34 )
		{
			// no topological change
			mutate_branch_lenght( pmut);
			//this->invalidateCaches();
		}
	}
}


void phylotreeIND::mutate_branch_lenght(float pmut)
{
	double gamma; 
	
	graph::edge_iterator it = TREE.edges_begin();
	graph::edge_iterator it_e = TREE.edges_end();
	while(it!= it_e)
	{	
		if (randomNr->flipP(0.5))
		{
			gamma =  gsl_ran_gamma (random_gsl, randomgsl_alpha, (1.0/randomgsl_alpha) );
			if( get_branch_length(*it) * gamma != 0 )
				set_branch_length( *it, get_branch_length(*it) * gamma);
		}
		++it;
	}
}


// select an internal edge randomly
edge phylotreeIND::choose_internal_edge() const
{
	edge edgeaux;
	do
		edgeaux = select_edge(); 
	while( istaxon(edgeaux.source()) || istaxon(edgeaux.target()) );
	return edgeaux;
}


// create a graph of n nodes whithout edges
void phylotreeIND::init()
{
   
}





// implements the crossover_operator of GAML (Lewis, 1998)
// place the result in TREE2
void phylotreeIND::crossover_gaml(phylotreeIND &TREE2)
{
	graph::edge_iterator edgeaux = TREE.edges_end(); // point to subtree in parent 1
	edgeaux--;edgeaux--;
	node root1, parent1; // point to the root of the subtree in parent1
	list<edge> subtree_edges;
	list<node> subtree_nodes;
	// select the root of the tree to be pruned (the root is market with 1 in the split
	// corresponding to edgeaux
	root1 = split(*edgeaux, edgeaux->source()) ? edgeaux->source() : edgeaux->target();
	// select the root of the subtree pruned
	parent1 = root1 == edgeaux->source() ? edgeaux->target() : edgeaux->source();
	// save the distance of edgeaux
	double dist = get_branch_length(*edgeaux);
	
	obtain_subtree( root1, &parent1, &subtree_edges, &subtree_nodes);
	
	// eliminate the taxons in subtree from TREE2
	list<node>::iterator it= subtree_nodes.begin();
	list<node>::iterator it_e= subtree_nodes.end();
	while( it != it_e )
	{
		if( istaxon(*it) ) TREE2.remove_taxon( MAPNODETAXON[*it] );
		it++;
	}
		
	
	// point to the last node in TREE2
	graph::node_iterator root2 = TREE2.TREE.nodes_end(); root2--;
	
	// construct the subtree1 in the TREE2
	TREE2.construct_graph( *this, subtree_nodes, subtree_edges);
	// connect the subtree created in an edge of TREE2;
	graph::edge_iterator edgeaux2 = TREE2.TREE.edges_begin();
	// the root of the new subtree in TREE2 is root2++ given the root2 was
	// inserted at end of the list of nodes
	TREE2.insert_node_in_edge(  *(++root2) , *edgeaux2, dist);
}



// copy a subtree to another individual, preventing the duplicated nodes
// when applied two times is the same that interchange subtrees
// note: dest tree MUST have the same number of taxas (related to the same problem)
void phylotreeIND::export_subtree( phylotreeIND &dest)
{
	 
	// DEBUG INFO
	graph::edge_iterator itdebug = TREE.edges_begin();
	
	// subtree nodes and edges
	list<node> subtree_nodes;
	list<edge> subtree_edges;
	// points to the root of the subtree and the parent of the subtree
	node root_subtree, parent_subtree;
	edge source_edge, dest_edge;
	// select the subtree to be exported
	source_edge =  select_edge();
				//select_edge_at_pos( TREE.all_edges(),  j);
	// save the distance of the edge comming to the root_subtree
	double dist = get_branch_length(source_edge);
	
	root_subtree = split(source_edge,source_edge.source()) ? source_edge.source() : source_edge.target();
	// points to the parent of the subtree in order to do the percurse
	
	// DEBUG INFO
	//cout << "raiz da arvore" << root_subtree << endl;
	
	parent_subtree = source_edge.opposite(root_subtree); 
	// obtain the subtree
	obtain_subtree( root_subtree, &parent_subtree, &subtree_edges, &subtree_nodes);
	// eliminate duplicate taxons in dest
	list<node>::iterator it = subtree_nodes.begin();
	list<node>::iterator it_e = subtree_nodes.end();

	while(it != it_e)
	{
	     if( istaxon(*it) ) dest.remove_taxon( MAPNODETAXON[ *it] );
		++it;
	}
		
	// DEBUG INFO
	//cout << "arvore destino podada" << endl;
	//dest.printtree();	
	
	// select the destination edge in dest
	//dest_edge = select_edge_at_pos( dest.TREE.all_edges(),  j);
	dest_edge = dest.select_edge();

	// help to find the root of the subtree regrafted
	
	// DEBUG INFO
	//cout << "eixo regrafting" << dest_edge << endl;
	
	graph::node_iterator dest_root_subtree = --dest.TREE.nodes_end();
	// copy the subtree in dest
	dest.construct_graph( *this, subtree_nodes, subtree_edges);
	//  now, as dest_parent_node point the last node in dest before insertion,
	//  then dest_parent_node++ point to the root of the subtree recently inserted
	//  Only remain insert it to the dest_edge in order to connect the tree
	dest.insert_node_in_edge( *(++dest_root_subtree), dest_edge, dist);
	
	//DEBUG INFO
	itdebug = dest.TREE.edges_begin();
}

// parsigal crossover work in rooted an unrooted trees
// note: crossover_gaml and parsigal 
void phylotreeIND::crossover_parsigal( phylotreeIND & TREE2,phylotreeIND & SON1, phylotreeIND & SON2)
{
	list<node> subtree1_nodes, subtree2_nodes;
	list<edge> subtree1_edges, subtree2_edges;
	graph::edge_iterator crossoverpoint1, crossoverpoint2, edgeaux1, edgeaux2;
	node root1, root2;
	node parent1, parent2;
	
	edgeaux1 = TREE.edges_begin();
	edgeaux1++;edgeaux1++;edgeaux1++;
	edgeaux2 = TREE2.TREE.edges_end();
	edgeaux2--;edgeaux2--;
	
	SON1.copy(*this);
	SON2.copy(TREE2);
	
	
	// works from both rooted and unrooted trees
	root1 = split(*edgeaux1,edgeaux1->source()) ? edgeaux1->source() : edgeaux1->target();
	root2 = TREE2.split(*edgeaux2,edgeaux2->source()) ? edgeaux2->source() : edgeaux2->target();
	
	// its imposible to swap the entire tree
	if(root!=NULL && (root1 == *root || root2 == *TREE2.root)) return;
	
	// again works from both rooted and unrooted
	parent1 = root1==edgeaux1->source() ? edgeaux1->target() : edgeaux1->source();
	parent2 = root2==edgeaux2->source() ? edgeaux2->target() : edgeaux2->source();
	
	
	// point to the position to graft the subtrees
			
	obtain_subtree( root1, &parent1, &subtree1_edges, &subtree1_nodes);
	TREE2.obtain_subtree( root2, &parent2, &subtree2_edges, &subtree2_nodes);
	
	// eliminate taxons from the subtree1 in SON2 and subtree2 in SON1
	
	list<node>::iterator it = subtree1_nodes.begin();
	list<node>::iterator it_e = subtree1_nodes.end();
	while(it != it_e)
	{
	     if( istaxon(*it) ) SON2.remove_taxon( MAPNODETAXON[ *it] );
		++it;
	}
	it = subtree2_nodes.begin();
	it_e = subtree2_nodes.end();
	while(it != it_e)
	{
		if( TREE2.istaxon(*it) ) SON1.remove_taxon( TREE2.MAPNODETAXON[ *it] );
		++it;
	}
	std::copy( SON1.TREE.edges_begin()	, SON1.TREE.edges_end(), ostream_iterator<edge>( cout, " , " ));
	cout << endl;
	std::copy( SON2.TREE.edges_begin()	, SON2.TREE.edges_end(), std::ostream_iterator<edge>( cout, " , " ));
	cout << endl;
		
	// select the crossover sites in son1 and son2
	
	crossoverpoint1 = SON1.TREE.edges_begin();
	crossoverpoint2 = SON2.TREE.edges_begin();
	
	// select the root of the subtree to be conected
	graph::node_iterator connect_1 = SON1.TREE.nodes_end(); connect_1--;
	graph::node_iterator connect_2 = SON2.TREE.nodes_end(); connect_2--;
	
	SON1.construct_graph( TREE2, subtree2_nodes, subtree2_edges);
	SON2.construct_graph( *this, subtree1_nodes, subtree1_edges);
	
	// conect to the parent 1
	SON1.insert_node_in_edge( *(++connect_1), *crossoverpoint1);
	SON2.insert_node_in_edge( *(++connect_2), *crossoverpoint2);
	
	for(int i=0; i < SON1.nnode; i++) cout << i << "--> " << SON1.MAPTAXONNODE[i] << endl;
	for(int i=0; i < SON2.nnode; i++) cout << i << "--> " << SON2.MAPTAXONNODE[i] << endl;
	
}




void phylotreeIND::remove_taxon( int id)
{
	node delnode = MAPTAXONNODE[id];
	// verify remaining two nodes in tree
	if (TREE.number_of_nodes() < 3) TREE.del_node(delnode);
	else
	{
		edge edgeaux = *delnode.in_edges_begin();
		node parent = edgeaux.source();
		remove_branch(edgeaux);
		TREE.del_node(delnode);
		// added by me
		collapse_node(parent);
	}
}



// this function gerate the list of edges that are separated by the
// edge edgeaux
void phylotreeIND::separategraphs(edge edgeaux, list<edge> &graph1, list<edge> &graph2)
{
        // map to the visited edges
	node nodeaux = edgeaux.target();
	obtain_subtree(edgeaux.source(), &nodeaux, &graph1, NULL);
	nodeaux = edgeaux.source();
	obtain_subtree(edgeaux.target(), &nodeaux, &graph2, NULL);
}


void phylotreeIND::calculate_splits_exp()
{
	// init the split memory
	edge edgeaux = choose_internal_edge();
	allocate_split_memory();
	memset( split_bits, false, (2*nnode-3)*(2*nnode-2)*sizeof(bool));
	//cout << "memoria alocada......" << endl;
	postorder_Iterator it = postorder_begin( edgeaux.source());

	// calculate the splits
	while(*it!=edgeaux.source())
	{
		int idx_begin1 = it.branch().id()*TREE.number_of_nodes();
		split_bits[ idx_begin1 + (*it).id() ] = true;
		int ones = 0;
		if( !istaxon(*it) )
		{
			child_Iterator son = child_begin( *it, it.ancestor() );
			child_Iterator son_end = child_end( *it, it.ancestor() );

			while( son != son_end)
			{
				int idx_begin2 = son.branch().id()*TREE.number_of_nodes();
				for(int i=0; i< TREE.number_of_nodes(); i++)
				{
					split_bits[idx_begin1 + i] = split_bits[idx_begin1 + i] || split_bits[idx_begin2 + i];
					if( split_bits[idx_begin2 + i] ) ones++;
				}
				// the son splits is not used anymore
				if (ones >= TREE.number_of_nodes()/2)invert_split( son.branch() );
				++son;
			}
		}
		++it;
	}
	valid_splits = true;
	//print_splits();
}

void phylotreeIND::calculate_splits()
{
	allocate_split_memory();
	memset( split_bits, false, (2*nnode-3)*(2*nnode-2)*sizeof(bool));
	//fill(split_bits.begin(), split_bits.end(), false);
	if( root!=NULL)    
	{
		int i;
		cin >> i;
		node::out_edges_iterator it = root->out_edges_begin();
		node::out_edges_iterator it_e = root->out_edges_end();
		while ( it != it_e )
		{
			calculate_splits_from_edge2( *it, it->target()); 
			it++;
		}
	}
	else
	{
		graph::edge_iterator edgeaux_it = TREE.edges_begin();
		graph::edge_iterator edgeaux_it_e = TREE.edges_end();
		edge edgeaux;
		// select an internal edge	
		while( edgeaux_it!= edgeaux_it_e )
		{
				if( ! istaxon( edgeaux_it->target() ) ) break;
				edgeaux_it++;
		}	
		// else select any edge
		if(edgeaux_it==TREE.edges_end())edgeaux_it=TREE.edges_begin();
		
		edgeaux = *edgeaux_it;
			
		calculate_splits_from_edge2( edgeaux, edgeaux.source()); 
		// clean the splits
		memset( split_bits + edgeaux.id()*TREE.number_of_nodes(), false,
				TREE.number_of_nodes());
		calculate_splits_from_edge2( edgeaux, edgeaux.target());
	
		int ones = 0;
		int nn = TREE.number_of_nodes();
		int idx_begin = edgeaux.id() * nn;
		for( int i=0; i < nn; i++) 
			if( split_bits[ idx_begin + i] )ones++;
		if (ones >= TREE.number_of_nodes()/2) 
				invert_split( edgeaux );
		// display splits
	}
	valid_splits = true;
}



void phylotreeIND::calculate_splits_from_edge2(edge edgeaux, node from)
{
	node::inout_edges_iterator it = from.inout_edges_begin();
	node::inout_edges_iterator it_e = from.inout_edges_end();
	//cout << "chamando --> " << edgeaux << "," << from << endl;
	//SPLITS[edgeaux].clear();
	//SPLITS[edgeaux].init(TREE,0);
	int idx_begin1 = edgeaux.id()*TREE.number_of_nodes();
	// clean the split
	//memset( &split_bits[idx_begin1], 0, TREE.number_of_nodes());
	if( edgeaux.id() >=  2*nnode-3 || from.id() >= 2*nnode-2)
	{
		cout << "edge_id:" << edgeaux.id() << "   node_id:" << from.id() << endl;
		assert( edgeaux.id() < 2*nnode-3 && from.id() < 2*nnode-2); 
	}
	split_bits[ idx_begin1 + from.id() ] = true;
	//SPLITS[edgeaux][from] = 1;
	if( istaxon(from) )	return;
	

   	while( it != it_e )
	{
		if( *it != edgeaux )
		{
			node target = it->source() == from ? it->target() : it->source();
			calculate_splits_from_edge2(*it, target);

			int n = TREE.number_of_nodes();
			int ones = 0;
			int idx_begin2 = it->id()*TREE.number_of_nodes();
			for(int i=0; i<n; i++)
			{
				split_bits[idx_begin1 + i] = split_bits[idx_begin1 + i] || split_bits[idx_begin2 + i];
				if( split_bits[idx_begin2 + i] ) ones++;
			}
			if (ones >= TREE.number_of_nodes()/2) 
				invert_split( *it );
		}
		it++;
	}
	// invert the split in the case that the graph is unrooted
}


void phylotreeIND::print_splits() const
{
	graph::edge_iterator it = TREE.edges_begin();
	graph::edge_iterator it_e = TREE.edges_end();
	while( it!= it_e)
	{
		cout << get_split_key( *it) << endl;
		//print_split(*it);
		++it;
	}
}

void phylotreeIND::print_split(edge edgeaux) const
{
      graph::node_iterator  it = TREE.nodes_begin();
	 graph::node_iterator  it_e = TREE.nodes_end();
      cout << "eixo " << edgeaux << "divide" << endl;
	 while( it != it_e )
	 {
	    if( !split(edgeaux,*it) )  cout << *it << " - ";
	    else cout << " *" << *it << " - ";
	    it++;
	 }
	 cout << endl;
}


// invert the zeros and ones in the split
void phylotreeIND::invert_split( edge edgeaux)
{
	int n= TREE.number_of_nodes();
	int idx_begin = edgeaux.id()*n;
	for(int i=0; i< n; i++) 
		split_bits[idx_begin+i] = !split_bits[idx_begin+i];
}





// return if an edgeaux is what set of the split in consensus
// 0 -> the taxas in edgeaux are in the "0" side
// 1 -> the taxas in edgeaux are in the "1" side
int phylotreeIND::split_set_edge( edge consensus, edge edgeaux)
{
	graph::node_iterator it = TREE.nodes_begin();
	graph::node_iterator it_e = TREE.nodes_end();
	int setconsensus = -1;
	int setedgeaux, setedgeaux2;
	bool flag = true;
	while(it != it_e)
	{
		// only check taxons
		if(istaxon(*it))
		{
			if(setconsensus == -1){
				setedgeaux = split(edgeaux,*it);
				setconsensus =  split(consensus,*it);
			}
			// checks only for taxon from split in edgeaux whose split taxon values = setedgeaux
			// if this value don't coincide with setconsensus then the corresponding set
			// in split of edgeaux is  the opposite and finalize the loop
			else	if( split(edgeaux,*it) == setedgeaux &&  split(consensus,*it) != setconsensus )
				flag = false;
			else if( split(edgeaux,*it) != setedgeaux )
			{
			     setedgeaux2 = split(consensus,*it);
				if(!flag)break;
			}
		}	
		++it;
	}
	return ( flag ? setconsensus : setedgeaux2 );
}



void phylotreeIND::stepwise_addition()
{
	list<node> auxlist = TREE.all_nodes();
	edge edgeaux;
	node node1, node2;
	
	// choose two nodes randomly add
	int j = randomNr->uniform0Max( auxlist.size() );
	int k = randomNr->uniform0Max( auxlist.size() );
	while(k == j)k = randomNr->uniform0Max(TREE.number_of_nodes());
	
	// choose the nodes
	node1 = select_edge_at_pos( auxlist, j);
	node2 = select_edge_at_pos( auxlist, k);
	
	// first pick two nodes and insert edge in it
	//edgeaux = TREE.new_edge(node1, node2);

	//MAPIDEDGE[edgeaux.id()] = edgeaux;

	edgeaux = new_branch(node1, node2);
		
	auxlist.remove( node1);
	auxlist.remove( node2);
	// the remaining edges
	while( auxlist.size()>0 )
	{
			j = randomNr->uniform0Max( auxlist.size() );
			node1 = select_edge_at_pos( auxlist, j);
			insert_node_in_edge(node1, edgeaux);
			auxlist.remove( node1);
			// selec the next edge to add a new node
			j = randomNr->uniform0Max( TREE.number_of_edges() );
			edgeaux = select_edge_at_pos( TREE.all_edges(), j);
	}
	calculate_splits();
}


void phylotreeIND::start_decomposition()
{
	list<node> auxlist = TREE.all_nodes();
	node node1, node2, cluster;
	edge edgeaux;
	while( auxlist.size() > 2)
	{
		// pick two nodes followingan criteria
		//cluster =  TREE.new_node();
		cluster =  new_internal_node();
		new_branch(cluster, node1);
		new_branch(cluster, node2);
		//edgeaux = TREE.new_edge( cluster, node1 );
		//MAPIDEDGE[edgeaux.id()] = edgeaux;
		//edgeaux = TREE.new_edge( cluster, node2 );
		//MAPIDEDGE[edgeaux.id()] = edgeaux;
		auxlist.remove( node1);
		auxlist.remove( node2);
		auxlist.push_back( cluster );
	} 
	new_branch( cluster, *(auxlist.begin()) );
	//edgeaux = TREE.new_edge( cluster, *(auxlist.begin()) ); 
	//MAPIDEDGE[edgeaux.id()] = edgeaux;
}





void phylotreeIND::SPR()
{
	
	
	// spr no makes sense in trees with less than four  edges (three or less taxons)
	if( TREE.number_of_edges() <5) return;
	
	// choose and edge and divides the three
	edge edgeaux1 = select_edge(); 
				//select_edge_at_pos( TREE.all_edges(),  j);
	
	node nodeaux, delnode, nodeaux2;

	// because the edgeaux will be deleted we save it split
	int idx_begin = edgeaux1.id() * TREE.number_of_nodes();
		
	// save the distance of edgeaux1
	double d_edgeaux1 = get_branch_length(edgeaux1);
	
	nodeaux = split(edgeaux1,edgeaux1.source()) ? edgeaux1.source() : edgeaux1.target();
	// select the source preventing remove a taxon
	delnode = edgeaux1.opposite(nodeaux); 
	// prune edge1 and graft in edge 2
	//cout << "eixo escolhido " << edgeaux1 << endl;
	remove_branch(edgeaux1);
	//MAPIDEDGE[edgeaux1.id()] = invalid_edge;
	//TREE.del_edge(edgeaux1);
	collapse_node(delnode);

	edge edgeaux2 = choose_edge_fromside(idx_begin, false);

	// finally, regraft in edgeaux2
	// insert the subtree rooted at nodeaux in edgeaux2;
	insert_node_in_edge(nodeaux, edgeaux2, d_edgeaux1);

}

void phylotreeIND::TBR()
{
	// tbr no makes sense in trees with less than four  edges (three or less taxons)
	if( TREE.number_of_edges() <7) return;
     	list<edge> tree1, tree2;	
	edge edgeaux1, edgeaux2, prune_edge;
	node source, dest, nodeaux1, nodeaux2;

	// select the source and dest edges
	prune_edge = choose_internal_edge();
	// the graph was divide by removing edgeaux1

	int idx_begin = prune_edge.id() * TREE.number_of_nodes();


	source = prune_edge.source();
	dest =  prune_edge.target();
	//TREE.del_edge(prune_edge);
	remove_branch(prune_edge);
	// delete the source and target nodes*/
	collapse_node(source);
	collapse_node(dest);

	edgeaux1 = choose_edge_fromside( idx_begin, true);
	edgeaux2 = choose_edge_fromside( idx_begin, false);
	// create the new nodes to be reconected in the respective subtrees
	nodeaux1 = divide_edge(edgeaux1);
	nodeaux2 = divide_edge(edgeaux2);
	// reconect the tree
	new_branch(nodeaux1, nodeaux2);
	//edgeaux1 = TREE.new_edge(nodeaux1, nodeaux2);
	//MAPIDEDGE[edgeaux1.id()] = edgeaux1;
}

//  subtrees in the tree
void phylotreeIND::change_subtrees()
{
     edge source_edge, dest_edge;
     list <edge> remain_edges;
	node root_subtree1, root_subtree2, parent_subtree1, parent_subtree2;
	// select the edge of the subtrees
	source_edge = select_edge();
	root_subtree1 = split(source_edge, source_edge.source())
						? source_edge.source() : source_edge.target();
	// points to the parent of the subtree in order to do the percurse
	parent_subtree1 = source_edge.opposite(root_subtree1); 

	dest_edge = select_edge_outsidetree(source_edge);

	bool lado = !split(dest_edge,root_subtree1);
	root_subtree2 = split(dest_edge,dest_edge.source()) == lado 
				? dest_edge.source() : dest_edge.target();
		
	parent_subtree2 = dest_edge.opposite(root_subtree2); 	

	// reconecting the nodes
	if( parent_subtree1 == source_edge.source() )
		source_edge.change_target(root_subtree2);
	else	
		source_edge.change_source(root_subtree2);
	if( parent_subtree2 == dest_edge.source() )
		dest_edge.change_target(root_subtree1);
	else	
		dest_edge.change_source(root_subtree1);

	if( istaxon(source_edge.source()) ) source_edge.reverse();
	if( istaxon(dest_edge.source()) ) dest_edge.reverse();
}


// divides an edge by creating a new node and it with the
// edge extrems and return the new node;
// set distance dist to new edge to be inserted
void phylotreeIND::insert_node_in_edge(node nodeaux, edge edgeaux, double dist)
{ 
     // divide an edgeaux
	node new_internal_node = divide_edge(edgeaux);
	// new_node is not a taxon
	//MAPNODETAXON[new_internal_node] = -1;
	// connect the new node
	new_branch( new_internal_node, nodeaux, dist);
	//edge new_edge = TREE.new_edge( new_internal_node, nodeaux);
	//set_branch_length(new_edge, dist);

	//MAPIDEDGE[new_edge.id()] = new_edge;
}


// breaks an edge with an new root edge that joins the edgeaux extremes
node phylotreeIND::divide_edge(edge edgeaux)
{
     // create the node that divides edgeaux
     //node r = TREE.new_node();
	//assert(r.id() < 2*nnode -2);
	// because the edge is divided, the distance of the edge is also divided
	node r = new_internal_node();
	double d = get_branch_length(edgeaux)/2;
	// join the edgeaux vertices to root
	new_branch( r, edgeaux.source(), d);
	//edge new_edge = TREE.new_edge( r, edgeaux.source());
	// each part of the edge receives the half of the original distance
	//set_branch_length(new_edge, d);
	set_branch_length(edgeaux, d);
	//MAPIDEDGE[new_edge.id()] = new_edge;
	edgeaux.change_source(r);
	return r;
}


void phylotreeIND::collapse_node(node delnode)
{
	edge edgeaux; // new edge to be inserted
    // case the delnode have a degree 3, the graph is valid
    if(delnode.degree() > 2) return;
    // the node is incomplete, remove them and link the adjacent noddes
    else if(delnode.degree() == 2)
    {
    		node adj1, adj2;
		node::inout_edges_iterator it1, it2;
		it1 = it2 = delnode.inout_edges_begin();
		it2++;
		adj1 = it1->opposite(delnode); 
		adj2 = it2->opposite(delnode); 
		// if adj1 is a taxon, connect adj2->adj1
		if( istaxon(adj1) )
			new_branch( adj2, adj1, get_branch_length(*it1) + get_branch_length(*it2)); 
			//edgeaux = TREE.new_edge( adj2, adj1);
		else
			new_branch( adj1, adj2, get_branch_length(*it1) + get_branch_length(*it2)); 
			//edgeaux = TREE.new_edge( adj1, adj2);
		// set the new distance as the sum of edge lenghts
		//set_branch_length( edgeaux, get_branch_length(*it1) + get_branch_length(*it2) );
		//MAPIDEDGE[edgeaux.id()] = edgeaux;
		remove_branch( *it1);
		remove_branch( *it2);
		//MAPIDEDGE[it1->id()] = invalid_edge;
		//MAPIDEDGE[it2->id()] = invalid_edge;
    	}
	// invalidate edges
    	TREE.del_node(delnode);
}


void phylotreeIND::collapse_zero_edges()
{
	graph::edge_iterator it = TREE.edges_begin();
	graph::edge_iterator it_end = TREE.edges_end();
	graph::edge_iterator tmp;
	node source, dest;
	while(it!=it_end)
	{
		tmp = it;
		++tmp;
		if(is_internal( *it)  && get_branch_length( *it) == 0)
		{
			source = it->source();
			dest = it->target();
			//cout << "apagando branch" << endl;
			remove_branch(*it);
			//cout << "reconectando nos..." << endl;
			reconnect_nodes( source, dest );
		}
		it = tmp;
	}
}

// connect all adjacent nodes from source to dest and delete source
void phylotreeIND::reconnect_nodes(node source, node dest)
{
	node::inout_edges_iterator it = source.inout_edges_begin();
	node::inout_edges_iterator it_end = source.inout_edges_end();
	node::inout_edges_iterator tmp_it;
	edge tmp;
	while(it != it_end)
	{
		tmp = *it;
		tmp_it = it;
		++tmp_it;
		if( it->source() == source)
			tmp.change_source( dest);
		else if(it->target() == source)
			tmp.change_target( dest);
		it = tmp_it;
	}
	TREE.del_node(source);
}




// select an neighboor defined by the edge edgeaux and node start
edge phylotreeIND::choose_neighboor(edge edgeaux, node start) const
{
	
	int n_neighboors = start.degree();
	// choose an neighbooord edge differnet from edgeaux
	node::inout_edges_iterator it;
	do
	{
		int k  = randomNr->uniform0Max( n_neighboors );
		it = start.inout_edges_begin();
		for(int i=0; i < k; i++) it++;
	}while( *it == edgeaux );
	// select the neighboor
	return *it;
}


void phylotreeIND::taxa_swap()
{
	node node1, node2;
	node1 = taxon_number( randomNr->uniform0Max( nnode ) );
	node2 = taxon_number( randomNr->uniform0Max( nnode ) );
	edge extern_edge1 = *node1.inout_edges_begin();
	edge extern_edge2 = *node2.inout_edges_begin();
	extern_edge1.change_target(node2);
	extern_edge2.change_target(node1);
}



void phylotreeIND::NNI()
{
   edge select_edge;
   edge edge_neighboor1, edge_neighboor2;
   node node_neighboor1, node_neighboor2;
   
    // select an internal edge
	select_edge = choose_internal_edge();

    
    // choose the neighboors
    edge_neighboor1 = choose_neighboor( select_edge, select_edge.source() );
    
    node_neighboor1 = edge_neighboor1.opposite( select_edge.source() );
    
    edge_neighboor2 = choose_neighboor( select_edge, select_edge.target() );
    
    node_neighboor2 = edge_neighboor2.opposite( select_edge.target() );
				  
	
	//change the neighboors
    
	if( edge_neighboor1.source() == node_neighboor1 )	  
		edge_neighboor1.change_source( node_neighboor2);
	else
		edge_neighboor1.change_target( node_neighboor2);
		
	if( edge_neighboor2.source() == node_neighboor2 )	  
		edge_neighboor2.change_source( node_neighboor1);
	else
		edge_neighboor2.change_target( node_neighboor1);
		
	if(istaxon( edge_neighboor1.source() ))edge_neighboor1.reverse();	
	if(istaxon( edge_neighboor2.source() ))edge_neighboor2.reverse();	
}



void phylotreeIND::insert_root(edge edgeaux)
{
     // first divide the edge with a new node that connects
	// its vertices
	root = new node(divide_edge(edgeaux));
	// finally convert the graph to thee with  root nodeaux
	convert_graph_to_tree(*root, NULL);
	
}


void phylotreeIND::convert_graph_to_tree(node n, node *antecessor)
{
	edge edgeaux;
	// reverse all the incoming edges that are not coming from antecessor
	node::inout_edges_iterator it,tmp;
	it = tmp = n.inout_edges_begin();
	node::inout_edges_iterator it_e = n.inout_edges_end();
	while(it!= it_e)
	{
		//neccesary if the variable "it" is deferenced
		tmp++;
		edgeaux = *it;
		if(antecessor==NULL || edgeaux.source()!=*antecessor)
		{
		     // reverse the edge and deferefence it
			if( edgeaux.source() != n) 
				edgeaux.reverse();
			convert_graph_to_tree(edgeaux.target(), &n);
		}
		it=tmp;
	}
}

// return the list of nodes and edges that conform the tree rooted at node root

void phylotreeIND::obtain_subtree(node n, node *antecessor, list <edge> *edgelist, list <node> *nodelist)
{
    //cout << "chamado con ...." << n;
    //cout << *antecessor << endl;
	node::inout_edges_iterator it, tmp, it_e;
	it = tmp = n.inout_edges_begin();  
	it_e = n.inout_edges_end();  
	edge edgeaux;
	// 
	if(nodelist)nodelist->push_back(n);
	while( it != it_e )
	{
		if(antecessor==NULL || (it->source()!=*antecessor && it->target()!=*antecessor) )   
		{
			if(edgelist) edgelist->push_back(*it);
			if(edgelist->size() > TREE.number_of_edges())
			{
				cout << "hubo un error..." << endl;
				int lll;
				cin >> lll;
			}
			if( it->source() == n )
				obtain_subtree( it->target(), &n,  edgelist, nodelist);
			else obtain_subtree( it->source(), &n, edgelist, nodelist);
		}
	it++;
    }
}


bool phylotreeIND::isparent(node parent, node children) const
{
	if( parent == *root) return true;
	else
	{
		while( children.indeg()!=0 )
		{
	   		if (parent == children) return true;
	   		children = children.in_edges_begin()->source();
		}
	}
	return false;
}

node phylotreeIND::firstcommonancestor(node node1, node node2) const
{
	if( isparent( node1, node2 )) return node1;
	else
	{
		while(node2.indeg()!=0 )
		{
			if ( isparent( node2 , node1) ) 
				return node2;
			node2 = node2.in_edges_begin()->source();
		}
	}
	// root is the parent
	return *root;
}

node phylotreeIND::firstcommonancestor( list<node> &listnode) const
{
	// points to the second element
	list<node>::iterator it = ++listnode.begin();
	list<node>::iterator it_e = listnode.end();
	node ancestor = *listnode.begin();
	while(it!= it_e) { ancestor = firstcommonancestor( ancestor, *it); ++it; }
	return ancestor;
}




// return true when edgeaux is an internal edge
bool phylotreeIND::is_internal(edge edgeaux) const
{
	return ( !istaxon(edgeaux.source()) && !istaxon(edgeaux.target()) );
}

// parsimony for rooted and non-rooted threes


// construct an graph with nodes and edges in the lists provided
void phylotreeIND::construct_graph( const phylotreeIND &G, list <node> &gnodes, list <edge> &gedges)
{
     // maps the nodes from list nodes to the new nodes in tree
     map <node, node> fromto;
	list<node>::iterator it = gnodes.begin();
	list<node>::iterator it_e = gnodes.end();
	edge aux;
	node n;
	while(it!=it_e)
	{
		//cout << "insertando taxa " << *it << G.MAPNODETAXON[*it] << endl;
		if( G.istaxon(*it) )
			n = new_taxon( G.MAPNODETAXON[*it] );
		else
			n = new_internal_node();
		fromto.insert( pair<node, node> (*it, n));
		/*node n = TREE.new_node();
		assert(n.id() < 2*nnode -2);
		fromto.insert( pair<node, node> (*it, n));
		MAPNODETAXON[n] = G.MAPNODETAXON[*it];
		if(G.istaxon(*it))
		{
			inserido = true;
			MAPTAXONNODE[ MAPNODETAXON[n] ] = n;
		}*/
		++it;
	}
	list<edge>::iterator it2 = gedges.begin();
	list<edge>::iterator it2_e = gedges.end();
	while(it2!=it2_e)
	{
		new_branch( fromto[it2->source()], fromto[it2->target()], G.get_branch_length(*it2));
		//aux = TREE.new_edge( fromto[it2->source()], fromto[it2->target()] );
		//set_branch_length( aux, G.get_branch_length(*it2) );
		//MAPIDEDGE[aux.id()] = aux;	
		++it2;
	}
}



void phylotreeIND::printtree() const
{
	graph::node_iterator it, it_e;
	node::out_edges_iterator it2, it2_e;
	it = TREE.nodes_begin();
	it_e = TREE.nodes_end();
	while(it!=it_e)
	{
		if(istaxon(*it)) cout << "*" << "(" << MAPNODETAXON[*it] << ")" << *it << " :";
		it2 = it->out_edges_begin();
		it2_e = it->out_edges_end();
		for( ; it2 != it2_e; it2++)
			cout << *it2 << "(" << get_branch_length(*it2) << ")" << ", " ;
		cout << endl;
		++it;
	}
}


void phylotreeIND::printNewick(ostream &os) 
{
	//edge edgeaux = choose_internal_edge();
	//string s = "(";
	//newick_traverse( edgeaux.source(), NULL, edgeaux, s);
	//s += ");\n";
	string s = newick_traverse2(true, true);
	os << s << endl;
}


string phylotreeIND::newick_traverse2( bool brlens, bool longnames) 
{
	string s;
	ostringstream s_2;
	s_2.precision(15);
	s_2.setf(ios::fixed);

	edge edgeaux = * ( MAPTAXONNODE[0].inout_edges_begin());
	node root_traverse = edgeaux.opposite( MAPTAXONNODE[0] );

	convert_graph_to_tree( root_traverse, NULL);

	postorder_Iterator it = postorder_begin( root_traverse );
	postorder_Iterator it2 = postorder_end( root_traverse );

	// imprimir ordem dos filhos
// 	child_Iterator it3 = child_begin( root_traverse);
// 	child_Iterator it4 = child_end( root_traverse);
// 	
// 	cout << "ordem de percorrido" << endl;
// 
// 	while (it3!=it4)
// 	{
// 		if(istaxon(*it3)) cout << taxon_id(*it3) << ",";
// 		else cout << "I," << endl;
// 		++it3;
// 	}

// 	cout << endl;
	int prev_nivel = 1;
	while(it!= it2)
	{
		s_2.str("");
		if( istaxon(*it) )
		{
			//cout << "gravando\t" << MAPNODETAXON[*it] << endl;
			for(int i=0; i < it.stack_size() - prev_nivel; i++) s+="(";
			if( longnames)
				s_2 << seqpatterns->seqname( MAPNODETAXON[*it] );
			else s_2 << MAPNODETAXON[*it];
			if ( brlens)
			    s_2 << ":" << get_branch_length( it.branch() );
			s_2 << ",";	
			s += s_2.str();
			
		}
		else
		{
			if(s[ s.size() -1] == ',') s[s.size()-1] = ')';
			else s+=")";
			if( *it != edgeaux.source() && brlens)
				s_2 << ":" << get_branch_length( it.branch() );
			s_2 << ",";	
			s+= s_2.str();
		}
		prev_nivel = it.stack_size();
		++it;
	}
	s[s.size()-1] = ';';
	return s;
}



void phylotreeIND::newick_traverse( node n, node *ancestor, edge edgeaux, string &s)
{
	ostringstream s_2;
	s_2.precision(15);
	s_2.setf(ios::fixed);
	if(istaxon(n)) {
		
		s_2 << seqpatterns->seqname( MAPNODETAXON[n] ) << ":" << get_branch_length(edgeaux);
		s+= s_2.str();
		return;
	}
	else if(ancestor!=NULL)s+='(';
	node::inout_edges_iterator it, it_e;
	it =  n.inout_edges_begin();  
	it_e = n.inout_edges_end();  
	node nodeaux;
	bool first_son = true;
	while( it != it_e )
	{
		if(ancestor==NULL || (it->source()!=*ancestor && it->target()!=*ancestor) )   
		{
			if(!first_son)s += ",";
			first_son=false;
			nodeaux = it->source() == n ? it->target() : it->source();
			newick_traverse( nodeaux, &n, *it, s );
			
		}
		it++;
	}
	if(ancestor!=NULL)s_2 << "):" << get_branch_length(edgeaux);
	s+= s_2.str();
}

void phylotreeIND::read_newick( string newickstring)
{
	TREE.clear();
	edge invalid_edge;
	list<node> pilha;
	int pos = 0;
	node father;
	edge internal_edge;
	string taxon_name;
	double blen;
	char token;
	int tottaxons  = 0;
	int totinternals = 0;
	try{
	do
	{
		token = newickstring[pos];
		switch(token)
		{
			case '(':
			{
				//father = TREE.new_node();
				father = new_internal_node();
				//cout << "no interno..." << father.id() << endl;
				totinternals++;
				pilha.push_back(father);
				break;
			}
			case ')':
			{
				node tmp = father;
				assert(pilha.size() >0);
				pilha.pop_back();
				if(pilha.size()>0)
				{
					father = pilha.back();
					internal_edge = new_branch( father, tmp);	
					//internal_edge = TREE.new_edge(father, tmp);
					//MAPIDEDGE[internal_edge.id()] = internal_edge;
					//cout << "eixo interno..." << internal_edge.id() << endl;
				}
				break;
			}
			case ',':
				break;
			case ':':
			{
				read_bl(newickstring, blen, (++pos));
				//cout << "longitude de ramo:" << blen << endl;
//				cout << "lendo brlen" << blen << endl;
				set_branch_length( internal_edge, blen);	
				break;
			}
			case ';':
				break;
			default:
			{
				read_taxonname_bl(newickstring, taxon_name, blen, pos);
				// create the taxon
				
				cout << "agregando taxon" << taxon_name << endl;
				string short_taxonname = taxon_name.substr(0,10);
				int taxon_id = seqpatterns->search_taxon_name(short_taxonname);
				//cout << "tentando agregar taxon ..." << taxon_name;
				if( taxon_id<0)
				{
					cout << "falla al agregar taxon..." << taxon_name;
					throw ExceptionManager(11);
					//assert( taxon_id >=0 );
				}
				//cout << "...sucess" << endl;
				
				//cout << "eixo interno..." << e.id() << endl;
				//cout << "longitude de ramo:" << blen << endl;
				node n = new_taxon( taxon_id);
				tottaxons++;
				//cout << "taxon..." << n.id() << endl;
				new_branch( father, n, blen);
				//edge e = TREE.new_edge( father, n);
				//MAPIDEDGE[e.id()] = e;
				//set_branch_length( e, blen);	
				//MAPNODETAXON[n] = taxon_id;
				//MAPTAXONNODE[taxon_id] = n;
				break;
			}
		}
		pos++;
	}while (token!=';');
	}
	catch(ExceptionManager e)
	{
		e.Report();
	}
	cout << "total... --> nos:" << TREE.number_of_nodes() << "  eixos:" << TREE.number_of_edges() << endl;
	cout << "contados..--> taxons:" << tottaxons << "  internals:" << totinternals << endl;
}


void phylotreeIND::read_newick2(string newickstring)
{
	//cout << "leyendo arvore\n";
	TREE.clear();
	edge invalid_edge;
	list<node> pilha;
	int pos = 0;
	node father;
	edge internal_edge;
	string taxon_name;
	double blen;
	char token;
	int tottaxons  = 0;
	int totinternals = 0;
	int taxon_id;
	do
	{
		token = newickstring[pos];
		switch(token)
		{
			case '(':
			{
				//father = TREE.new_node();
				father = new_internal_node();
				//cout << "no interno..." << father.id() << endl;
				totinternals++;
				pilha.push_back(father);
				break;
			}
			// close internal node
			case ')':
			{
				pos++;
				if( read_taxonname_bl(newickstring, taxon_name, blen, pos) == -1)
				{
					cout << "sintax error in position " << pos << endl;
					cout << newickstring << endl;
					throw ExceptionManager(11);
					exit(1);
				}
				node tmp = father;
				assert(pilha.size() >0);
				pilha.pop_back();
				if(pilha.size()>0)
				{
					//cout << "warning... ignorin internal taxon_name" << taxon_name 
					//	<< " with blen:" << blen << endl;
					if(blen>1.e+05){
						cout << "warning ... blen >1";
						blen=1.e+05;
					}
					if(blen == 0){
						 //cout << "warning.... blen = 0 , posicao " << pos << endl;
						 //blen = BL_MIN;
					}
					if(blen <0){
						cout << "warning.... blen negativo " << blen << endl;
						cout << "posicao " << pos << endl;
						cout << "taxonname" << taxon_name  << endl;
						blen=0;
					}
					father = pilha.back();
					internal_edge = new_branch( father, tmp);	
					set_branch_length( internal_edge, blen);
				}
				break;
			}
			case ',':
			case ';':
				break;
			default:
			{
				if(read_taxonname_bl(newickstring, taxon_name, blen, pos) == -1)
				{
					cout << "sintax error at position " << pos << endl;
					cout << newickstring << endl;
					exit(1);
			
				}
				// create the taxon
				// check if  a taxon is a number (no long name!)
				istringstream convert(taxon_name);
				if( convert >> taxon_id)
				{
					if(taxon_id < 0 || taxon_id > ( number_of_taxons() -1) )
					taxon_id=-1;
						
				}
				// long name
				else
				{
				
					string short_taxonname = taxon_name.substr(0,10);
					taxon_id = seqpatterns->search_taxon_name(short_taxonname);
// 				cout << "tentando agregar taxon ..." << taxon_name << "con blen" << blen 
// 					<< "posicon " << pos << endl;
				}
				if( taxon_id<0)
				{
					cout << "falla al agregar taxon..." << taxon_name;
					throw ExceptionManager(11);
					assert( taxon_id >=0 );
				}
				//cout << "leyendo\t" << taxon_id << endl;
				node n = new_taxon( taxon_id);
				tottaxons++;
				if(blen<0)
				{
					cout << "negative set to 0";
					blen=0;
				}
				new_branch( father, n, blen);
				break;
			}
		}
		pos++;
	}while (token!=';');
	//cout << "total... --> nos:" << TREE.number_of_nodes() << "  eixos:" << TREE.number_of_edges() << endl;
	//cout << "contados..--> taxons:" << tottaxons << "  internals:" << totinternals << endl;
}


int phylotreeIND::read_taxonname_bl( string &s, string &taxonname, double &blen, int &pos)
{
	int first_pos = s.size();

	int first_par = s.find( ')', pos); //first parentesis
	first_pos = (first_par !=string::npos && first_par < first_pos)
				? first_par : first_pos;

	int first_comma = s.find( ',', pos);
	first_pos = (first_comma !=string::npos && first_comma < first_pos)
				? first_comma : first_pos;
	int first_points = s.find( ':', pos);
	first_pos = (first_points !=string::npos && first_points < first_pos)
				? first_points : first_pos;
	int first_semicolon = s.find( ';', pos);

	first_pos = (first_semicolon !=string::npos && first_semicolon < first_pos)
				? first_semicolon : first_pos;

	// assert
	if(first_pos == string::npos) return -1; // error !
	// get_taxon_name
	taxonname = s.substr( pos, first_pos - pos );
	if( first_pos == first_points)
	{
		 pos = first_pos+1;
		 return read_bl( s, blen, pos );
	}
	//else cout << "no brlens..." << endl;
	pos = first_pos-1;
	return 0;
}

int phylotreeIND::read_bl( string &s, double &blen, int &pos)
{
	int first_pos = s.find( ';', pos); // final semicolor ex. (a,b):xx.yy;
	int first_par = s.find( ')', pos); //first parentesis
	first_pos = (first_par !=string::npos && first_par < first_pos)
			? first_par : first_pos;
	int first_comma = s.find( ',', pos);
	first_pos = (first_comma !=string::npos && first_comma < first_pos)
			? first_comma : first_pos;

	if(first_pos == string::npos) return -1;
	
	//assert(first_pos != string::npos);

	string bl = s.substr( pos, first_pos - pos );
	istringstream convert(bl);
	if (convert >> blen) 
	{
		pos = first_pos-1;
		return 0;
	}
	else return -1;
	//blen = atof(bl.c_str());
	//return first_pos - 1;
}

string phylotreeIND::get_split_key( edge edgeaux) const
{
	int n= TREE.number_of_nodes();
	int idx_begin = edgeaux.id()*n;
	string s;
	graph::node_iterator it = TREE.nodes_begin();
	graph::node_iterator it_e = TREE.nodes_end();
	s.resize(nnode);
	while(it!=it_e)
	{
		if(istaxon(*it))
			s[ MAPNODETAXON[*it] ] =  
				split_bits[ idx_begin + it->id() ] == 1 ? '*':'.';
		++it;
	}
	return s;
}

string phylotreeIND::get_invert_split_key( edge edgeaux) const
{
	int n= TREE.number_of_nodes();
	int idx_begin = edgeaux.id()*n;
	string s;
	s.resize(nnode);
	graph::node_iterator it = TREE.nodes_begin();
	graph::node_iterator it_e = TREE.nodes_end();
	while(it!=it_e)
	{
		if(istaxon(*it))
			s[ MAPNODETAXON[*it] ] =  
				split_bits[ idx_begin + it->id() ] == 1 ? '.':'*';
		++it;
	}
	return s;
}


double phylotreeIND::compare_topology(phylotreeIND &other)
{
	map <string, edge>  split_repository;
	map<string,edge>::iterator it_aux;
	graph &tree2 = other.TREE;
	int num_different_edges=0;
	// create repository for splits in tree1
	graph::edge_iterator it= TREE.edges_begin();
	graph::edge_iterator it_e= TREE.edges_end();
	while(it!=it_e)
	{
		if(!istaxon(it->target()))
		{
			split_repository[get_split_key(*it)] = *it;
		}
		++it;
	}
	
	// count for different splits
	it = tree2.edges_begin();
	it_e = tree2.edges_end();

	while(it!=it_e)
	{
		if(!istaxon(it->target()))
		{
		if( (it_aux = split_repository.find( other.get_split_key(*it))) != split_repository.end()
		|| (it_aux = split_repository.find( other.get_invert_split_key(*it)) ) !=split_repository.end() )
				split_repository.erase( it_aux );
		else
			num_different_edges++;
		}
		//}
		++it;
	}
		// different tree1->tree2 + different tree2->tree1 (remaining)
	return num_different_edges + split_repository.size();
}


// day algorithms

double phylotreeIND::compare_topology_2(phylotreeIND &other)
{
	// hash table
	int *hash_table;
	// node mapes
	int *map_nodes;
	int node_count = 0;
	int n = number_of_taxons();
	int good_edges = 0;

	node_map<struct temp_info> interior_node(TREE, temp_info(n));
	
		
	
	// allocate memory
	hash_table = new int[n*2];
	for(int i=0; i<n*2; i++)hash_table[i] = -1;
	map_nodes = new int[n];

	// step 1
	// select last taxon as root
	node invalid_node;
	node root1 = taxon_number( n-1);

	// step 2 and 3+
	postorder_Iterator it = postorder_begin( root1);
/*
	while( *it != root1 )
	{
		struct temp_info &father_info = interior_node [ it.ancestor() ] ;
		struct temp_info &current_info = interior_node [ *it ] ;
			
		//cout << " node " << *it << " ancestral" << it.ancestor() << endl;
		if( istaxon(*it) )
		{
			// update the map
			map_nodes[ taxon_id( *it) ] = node_count;
			
			// check if is the leftmost
			if( father_info.left == n  ) //left_most == invalid_node
			{
				//father_info.left_most = *it;
				father_info.left = node_count;
			}
			//else father_info.right = node_count;	
			node_count++;
		}
		else
		{
			int idx;
			current_info.right = node_count-1;
			if( father_info.left == n  ) //left_most == invalid_node
			{
				idx = current_info.right*2;
				// step 3 copy to the hash table
				//father_info.left_most = *it;
				father_info.left = current_info.left;
				//father_info.right = current_info.right;
			}
			else
			{
				idx = current_info.left*2;
				father_info.right = current_info.right;
			}
			// fill hash table
			hash_table[ idx ] = current_info.left;
			hash_table[ idx+1 ] = current_info.right;

		}
		++it;
	}
*/

	int l, r;
	while( *it != root1 )
	{
		struct temp_info &father_info = interior_node [ it.ancestor() ] ;
		struct temp_info &current_info = interior_node [ *it ] ;
			
		//cout << " node " << *it << " ancestral" << it.ancestor() << endl;
		if( istaxon(*it) )
		{
			// update the map
			map_nodes[ taxon_id( *it) ] = r = node_count;
			
			// check if is the leftmost
			if( father_info.left == n /*left_most == invalid_node*/ )
			{
				//father_info.left_most = *it;
				father_info.left = node_count;
			}
			//else father_info.right = node_count;	
			node_count++;
			++it;
		}
		else
		{
			int idx;
			l = current_info.left;
			++it;
			if (istaxon(*it) || *it==root1) idx = (r)*2;
			else idx = l*2;
			current_info.right = r;
			if( father_info.left == n /*left_most == invalid_node*/ )
			{
				father_info.left = current_info.left;
			}
			//else
			//{
			//	father_info.right = current_info.right;
			//}
			// fill hash table
			hash_table[ idx ] = l;
			hash_table[ idx+1 ] = r;
		}
	}

	// step 4
	node root2 = other.taxon_number( n-1);
	// father of root2
	node no_root2 = root2.in_edges_begin()->source();

	interior_node.init(other.TREE, temp_info(n));

	it = postorder_begin( no_root2, root2);
	while( *it != no_root2)
	{
		struct temp_info &father_info = interior_node [ it.ancestor() ] ;
		struct temp_info &current_info = interior_node [ *it ] ;

		if( istaxon(*it) )
		{
			if( father_info.left_most == invalid_node ) father_info.left_most = *it;
			if( map_nodes[ other.taxon_id( *it)  ] < father_info.left) father_info.left = map_nodes[ other.taxon_id(*it) ];
			if( map_nodes[ other.taxon_id( *it)  ] > father_info.right) father_info.right = map_nodes[ other.taxon_id(*it) ];
			father_info.num_nodes++;
		}
		else
		{

			// check hash tables
			if ( current_info.right - current_info.left + 1 ==
				current_info.num_nodes)
			{		
				if(	(hash_table[ current_info.left*2 ] == current_info.left &&
					hash_table[ current_info.left*2+1 ] == current_info.right)
					||
					(hash_table[ current_info.right*2 ] == current_info.left &&
					hash_table[ current_info.right*2+1 ] == current_info.right) )
					good_edges++;
			}

			if( father_info.left_most == invalid_node )father_info.left_most = *it;
			if( current_info.left < father_info.left) father_info.left = current_info.left;
			if( current_info.right > father_info.right) father_info.right = current_info.right;
			father_info.num_nodes += current_info.num_nodes;
		}	
		++it;
	}
	delete [] map_nodes;
	delete [] hash_table;
	return	TREE.number_of_edges() - number_of_taxons() - good_edges +
		other.TREE.number_of_edges() - number_of_taxons() - good_edges;
}



double phylotreeIND::robinson_foulds_distance(phylotreeIND &other, int debug)
{
	map <string, edge>  split_repository;
	graph &tree2 = other.TREE;
	double distance = 0;
	// create repository for splits in tree1
	graph::edge_iterator it= TREE.edges_begin();
	graph::edge_iterator it_e= TREE.edges_end();
	map<string,edge>::iterator it_aux;

	while(it != it_e)
	{
		split_repository[get_split_key(*it)] = *it; ++it;
	}
	
	// count for different splits
	it = tree2.edges_begin();
	it_e = tree2.edges_end();

	while(it!=it_e)
	{
		if( (it_aux = split_repository.find( other.get_split_key(*it))) != split_repository.end()
		|| (it_aux = split_repository.find( other.get_invert_split_key(*it)) ) != split_repository.end() )
		{
			distance += fabs( other.get_branch_length(*it) 
					- get_branch_length( (*it_aux).second ));	
			split_repository.erase( it_aux );
		}
		// edge in tree2 not found in tree1
		else
		{
			 distance += pow(other.get_branch_length(*it),2);
		}
		++it;
	}
	// edges in tree1 not found in tree2
	it_aux = split_repository.begin();
	while( it_aux != split_repository.end() )
	{
		distance += get_branch_length( (*it_aux).second );
		++it_aux;
	}
	return distance;
}

void phylotreeIND::printtreeinfo() const
{
	cout << this->TREE;
	graph::node_iterator it=TREE.nodes_begin();
	graph::node_iterator it_e=TREE.nodes_end();
	cout << "taxa information" << endl;
	while(it!=it_e)
	{
		if(istaxon(*it))cout << " * " << MAPNODETAXON[*it];
		cout << endl;
		++it;
	}
}



double phylotreeIND::compare_topology_3(phylotreeIND &other)
{
	// hash table
	int n = number_of_taxons();
	struct temp_info	**hash_table;  // hash that points struct info
	struct temp_info 	*interior_node_info = new struct temp_info[n-1];
	struct temp_info 	*interior_node_info_2 = new struct temp_info[n-1];
	int idx_interior = 0;
	node_map<struct temp_info*> interior_node(TREE, NULL);
	node_map<struct temp_info*> interior_node_2(other.TREE, NULL);
	edge_map<struct temp_info*> interior_edge(TREE, NULL);
	// node mapes
	int *map_nodes;
	int node_count = 0;
	int good_edges = 0;

	
	// allocate memory
	hash_table = new struct temp_info*[n];
	for(int i=0; i<n; i++)hash_table[i] = NULL;
	map_nodes = new int[n];

	// step 1
	// select last taxon as root
	node invalid_node;
	node root1 = taxon_number( n-1);

	// step 2 and 3
	postorder_Iterator it = postorder_begin( root1);

	int l, r;
	while( *it != root1 )
	{
		struct temp_info *father_info = interior_node [ it.ancestor() ] ;
		struct temp_info *current_info = interior_node [ *it ] ;
			
		//cout << " node " << *it << " ancestral" << it.ancestor() << endl;
		if( istaxon(*it) )
		{
			// update the map
			map_nodes[ taxon_id( *it) ] = r = node_count;
			
			// check if is the leftmost
			if( father_info == NULL )
			{
				interior_node [ it.ancestor() ] = father_info = &interior_node_info[idx_interior];
				idx_interior++;
				//father_info.left_most = *it;
				father_info->left = node_count;
			}
			//else father_info.right = node_count;	
			node_count++;
			++it;
		}
		else
		{
			int idx;
			l = current_info->left;
			interior_edge[ it.branch() ] = current_info;
			
			if( father_info == NULL )
			{
				interior_node [ it.ancestor() ] = father_info = &interior_node_info[idx_interior];
				idx_interior++;
				father_info->left = current_info->left;
			}

			++it;
			if (istaxon(*it) || *it==root1) idx = r;
			else idx = l;
			
			current_info->right = r;
			// fill hash table
			hash_table[ idx ] = current_info;
		}
	}
	// step 4
	node root2 = other.taxon_number( n-1);
	// father of root2
	node no_root2 = root2.in_edges_begin()->source();

	interior_node_2.init(other.TREE, NULL);

	it = postorder_begin( no_root2, root2);


	idx_interior = 0;

	while( *it != no_root2)
	{
		struct temp_info *father_info = interior_node_2 [ it.ancestor() ] ;
		struct temp_info *current_info = interior_node_2 [ *it ] ;

		if( istaxon(*it) )
		{
			
			if( father_info == NULL)
			{
				interior_node_2 [ it.ancestor() ] = father_info = &interior_node_info_2[idx_interior];
				father_info->left = n;
				father_info->right= -1;
				father_info->num_nodes = 0;
				idx_interior++;
				//.left_most == invalid_node ) father_info.left_most = *it;
			}
			if( map_nodes[ other.taxon_id( *it)  ] < father_info->left) father_info->left = map_nodes[ other.taxon_id(*it) ];
			if( map_nodes[ other.taxon_id( *it)  ] > father_info->right) father_info->right = map_nodes[ other.taxon_id(*it) ];
			father_info->num_nodes++;
		}
		else
		{


			// check hash tables
			if ( current_info->right - current_info->left + 1 ==
				current_info->num_nodes)
			{		
				//cout << "inicio checkeando hash" << current_info->left << "  " << current_info->right << endl;
				//cout << "hash table adress" << hash_table[current_info->left] << "  " << hash_table[current_info->right] << endl;
				if(	hash_table[current_info->left]!=NULL)
					if(	hash_table[current_info->left]->left == current_info->left &&
					hash_table[ current_info->left]->right == current_info->right) good_edges++;
				if(hash_table[current_info->right]!=NULL)
					if(hash_table[ current_info->right]->left == current_info->left &&
					hash_table[ current_info->right]->right == current_info->right) good_edges++;
				//cout << "fin checkeando hash " << good_edges << endl;
			}

			if( father_info == NULL)
			{
				interior_node_2 [ it.ancestor() ] = father_info = &interior_node_info_2[idx_interior];
				father_info->left = n;
				father_info->right= -1;
				father_info->num_nodes = 0;
				idx_interior++;
				//.left_most == invalid_node ) father_info.left_most = *it;
			}
			if( current_info->left < father_info->left) father_info->left = current_info->left;
			if( current_info->right > father_info->right) father_info->right = current_info->right;
			father_info->num_nodes += current_info->num_nodes;
		}	
		++it;
	}
	interior_edge.clear();
	interior_node.clear();
	interior_node_2.clear();
	delete [] interior_node_info;
	delete [] interior_node_info_2;
	delete [] map_nodes;
	delete [] hash_table;
	return	TREE.number_of_edges() - number_of_taxons() - good_edges +
		other.TREE.number_of_edges() - number_of_taxons() - good_edges;
}


void phylotreeIND::calculate_splits4()
{
	// hash table
	int n = number_of_taxons();
	//struct split_info	**hash_table;  // hash that points struct info
	//struct split_info 	*interior_node_info = new struct split_info[n-1];
	splitstable.resize(n);

	int idx_interior = 0;
	interior_node.init(TREE, NULL);
	interior_edge.init(TREE, NULL);
	// node mapes
	//int *map_nodes;
	int node_count = 0;

	int temp2 = 2;
 
	
	// step 1
	// select last taxon as root
	node root1 = taxon_number( n-1);

	// step 2 and 3
	postorder_Iterator it = postorder_begin( root1);

	int l, r;
	while( *it != root1 )
	{
		struct split_info *father_info = interior_node [ it.ancestor() ] ;
		struct split_info *current_info = interior_node [ *it ] ;
	    
			
		//cout << " node " << *it << " ancestral" << it.ancestor() << endl;
		if( istaxon(*it) )
		{
			// update the map
			splitstable[ taxon_id( *it) ].map_to_node = r = node_count;
			splitstable[ node_count ].node_to_map = taxon_id( *it);
			// check if is the leftmost
			if( father_info == NULL )
			{
				interior_node [ it.ancestor() ] = father_info = &(splitstable[idx_interior]);
				idx_interior++;
				//father_info.left_most = *it;
				father_info->left = node_count;
			}
			//else father_info.right = node_count;	
			node_count++;
			++it;
		}
		else
		{
			int idx;
			l = current_info->left;
			interior_edge[ it.branch() ] = current_info;
			if( *it == it.branch().source() ) current_info->side = 0;
			else current_info->side = 1;
			
			if( father_info == NULL )
			{
				interior_node [ it.ancestor() ] = father_info = &(splitstable[idx_interior]);
				idx_interior++;
				father_info->left = current_info->left;
			}

			++it;
			if (istaxon(*it) || *it==root1) idx = r;
			else idx = l;
			
			current_info->right = r;
			// fill hash table
			splitstable[ idx ].hash = current_info;
		}
	}
}



double phylotreeIND::compare_topology_4(phylotreeIND &other)
{
	// hash table
	int n = number_of_taxons();

	node_map<struct split_info*> interior_node_2(other.TREE, NULL);
	vector<struct split_info> split_table2;
	split_table2.resize(n);

	int node_count = 0;
	int good_edges = 0;

	
	// allocate memory
	// step 4
	node root2 = other.taxon_number( n-1);
	// father of root2
	node no_root2 = root2.in_edges_begin()->source();

	postorder_Iterator it = postorder_begin( no_root2, root2);


	int idx_interior = 0;

	while( *it != no_root2)
	{
		struct split_info *father_info = interior_node_2 [ it.ancestor() ] ;
		struct split_info *current_info = interior_node_2 [ *it ] ;

		if( istaxon(*it) )
		{
			
			if( father_info == NULL)
			{
				interior_node_2 [ it.ancestor() ] = father_info = &(split_table2[idx_interior]);
				father_info->left = n;
				father_info->right= -1;
				father_info->num_nodes = 0;
				idx_interior++;
				//.left_most == invalid_node ) father_info.left_most = *it;
			}
			if( splitstable[ other.taxon_id( *it)  ].map_to_node < father_info->left) father_info->left = splitstable[ other.taxon_id( *it)  ].map_to_node;
			if( splitstable[ other.taxon_id( *it)  ].map_to_node > father_info->right) father_info->right = splitstable[ other.taxon_id( *it)  ].map_to_node;
			father_info->num_nodes++;
		}
		else
		{


			// check hash tables
			if ( current_info->right - current_info->left + 1 ==
				current_info->num_nodes)
			{		
				//cout << "inicio checkeando hash" << current_info->left << "  " << current_info->right << endl;
				//cout << "hash table adress" << hash_table[current_info->left] << "  " << hash_table[current_info->right] << endl;
				if( splitstable[current_info->left].hash !=NULL)
					if(	(splitstable[current_info->left].hash)->left == current_info->left &&
					(splitstable[current_info->left].hash)->right == current_info->right) good_edges++;
				if( splitstable[current_info->right].hash!=NULL)
					if((splitstable[ current_info->right].hash)->left == current_info->left &&
					(splitstable[ current_info->right].hash)->right == current_info->right) good_edges++;
				//cout << "fin checkeando hash " << good_edges << endl;
			}

			if( father_info == NULL)
			{
				interior_node_2 [ it.ancestor() ] = father_info = &(split_table2[idx_interior]);
				father_info->left = n;
				father_info->right= -1;
				father_info->num_nodes = 0;
				idx_interior++;
				//.left_most == invalid_node ) father_info.left_most = *it;
			}
			if( current_info->left < father_info->left) father_info->left = current_info->left;
			if( current_info->right > father_info->right) father_info->right = current_info->right;
			father_info->num_nodes += current_info->num_nodes;
		}	
		++it;
	}
	interior_node_2.clear();
	return	TREE.number_of_edges() - number_of_taxons() - good_edges +
		other.TREE.number_of_edges() - number_of_taxons() - good_edges;
}



void phylotreeIND::print_splits_2() const
{
	graph::edge_iterator it = TREE.edges_begin();
	graph::edge_iterator it_e = TREE.edges_end();
	while( it!= it_e)
	{
		cout << get_split_key_2( *it) << endl;
		//print_split(*it);
		++it;
	}
}

string phylotreeIND::get_split_key_2( edge edgeaux) const
{
	int n= number_of_taxons();
	int idx_begin = edgeaux.id()*n;
	string s(n, '.');
	if(is_internal(edgeaux))
	{
	    struct split_info *ref = interior_edge[edgeaux];
	    //cout << ref->left << "   "  << ref->right << "   ";
	    for(int i= ref->left; i <= ref->right; i++)
	    {
		s[ splitstable[i].node_to_map ] = '*';
		//cout << splitstable[i].node_to_map << "  " ;
	    }
	}
	else 	    
	  s[ MAPNODETAXON[edgeaux.target()] ] =  '*';
	return s;
}
