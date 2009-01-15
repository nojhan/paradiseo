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



#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef _phylotreeIND_H_
#define _phylotreeIND_H_
#define BL_MIN  1.e-10

#include <RandomNr.h>
#include <GTL/graph.h>
#include <gsl/gsl_rng.h>
#include <valarray>
#include <Sequences.h>
#include <treeIterator.h>
#include <stack>
#include <tree_limits.h>


class phylotreeIND 
{
	private:
		bool		valid_splits;
		edge		invalid_edge;
		node_map<int> MAPNODETAXON; // ecah node maps a taxon id in the vector
		vector<node>  MAPTAXONNODE; // each taxon number points to a node
		vector<edge>  MAPIDEDGE;
		valarray<double> NJDISTANCES;  // distances stored in the edges
		//edge_map< node_map<int> > SPLITS;  // the splits that each edge separate the graph
		bool *split_bits;
		Sequences *seqpatterns;

		int nnode; // number of nodes
		static double randomgsl_alpha; // shape paramter of gamma distribution
		gsl_rng *random_gsl; // GSL (GNU Scientific Library) random generator for gamma numbers)
		node *root;
		int parsimony;


		void init();
		void SPR(); //SPR operator
		void NNI(); // NNI operator
		void TBR(); // TBR operator
		void taxa_swap(); // taxa swap operator
		void collapse_node ( node );

		void insert_root ( edge );
		node divide_edge ( edge );
		void insert_node_in_edge ( node, edge, double d=0 );

		bool isparent ( node, node ) const;
		node firstcommonancestor ( node, node ) const;
		node firstcommonancestor ( list<node> & ) const;
		void calculate_splits_from_edge2 ( edge, node );
		int split_set_edge ( edge , edge );
		void print_split ( edge ) const;
		void obtain_subtree ( node , node *, list <edge> *, list<node> * );
		void construct_graph ( const phylotreeIND &, list <node>& , list <edge>& );
		void change_subtrees();
		void separate_subtree_from_edge ( edge, list<edge> &, bool );
		void crossover_gaml ( phylotreeIND & );

		void crossover_parsigal ( phylotreeIND &,phylotreeIND &, phylotreeIND & );
		void crossover_gaphyl ( phylotreeIND & );
		void invert_split ( edge );
		void remove_taxon ( int );
		void start_decomposition();
		void stepwise_addition();
		void separategraphs ( edge , list<edge> &, list<edge> & );
		void visit_edges_from_node ( node, list<edge> & );


		void mutate_branch_lenght ( float );
		edge select_edge_outsidetree ( edge source_edge ) const;
		edge choose_edge_fromside ( int id, bool side ) const;
		edge choose_neighboor ( edge, node ) const;


		int read_taxonname_bl ( string &s, string &taxonname, double &blen, int &pos );
		int read_bl ( string &s, double &blen, int &pos );
		edge new_branch ( node, node );
		edge new_branch ( node, node, double );
		void remove_branch ( edge );

		void reconnect_nodes ( node source, node dest );
		node new_taxon ( int id );
		node new_internal_node();

	public:
		RandomNr  *randomNr;
		phylotreeIND& operator= (const phylotreeIND& ind) { copy(ind); return *this; }
		virtual phylotreeIND* clone() const;
		virtual phylotreeIND* randomClone() const;


		GTL::graph TREE; // final tree calculated by NJ
		// constructors
		void loadsequences ( char * );
		void copy ( const phylotreeIND & );
		edge select_edge() const;
		phylotreeIND ( const phylotreeIND &org );
		phylotreeIND ( RandomNr *g, Sequences &p, gsl_rng *gslr );
		~phylotreeIND();

		// genetic operators
		void export_subtree ( phylotreeIND &dest );
		virtual void crossover ( float pcross, const phylotreeIND& dad, phylotreeIND*& sis, phylotreeIND*& bro ) const;
		virtual void mutate ( float pcross );

		//IND& operator= (const IND& ind) { copy( (phylotreeIND &)ind); return *this; }
		//phylotreeIND& operator= (phylotreeIND& ind) { copy(ind); return *this; }

		edge choose_internal_edge() const;
		void convert_graph_to_tree ( node, node * );

		void calculate_splits();
		void calculate_splits_exp();
		inline void invalidate_splits() { valid_splits = false; }
		inline void remove_split_memory()
		{
			if ( split_bits!=NULL ) delete [] split_bits;
			invalidate_splits();
			split_bits=NULL;
		}

		inline void allocate_split_memory()
		{
			if ( split_bits==NULL ) split_bits = new bool [ ( 2*nnode-3 ) * ( 2*nnode-2 ) ];
		}

		string get_split_key ( edge edgeaux ) const;
		string get_invert_split_key ( edge edgeaux ) const;

		bool is_internal ( edge ) const;

		void print_splits() const;

		void set_data ( Sequences &s );

		void collapse_zero_edges();

		inline double get_branch_length ( edge edgeaux ) const { return NJDISTANCES[edgeaux.id() ]; }
		inline int taxon_id ( node nodeaux ) const { return  MAPNODETAXON[nodeaux]; }
		inline int number_of_taxons() const { return seqpatterns->num_seqs(); }
		inline const Sequences & get_patterns() { return *seqpatterns; }
		inline Sequences *get_patterns2() { return seqpatterns; }
		inline node taxon_number ( int n ) const { return MAPTAXONNODE[n]; };
		inline int number_of_positions() const { return seqpatterns->pattern_count(); }
		inline void set_branch_length ( edge edgeaux, double f )
		{
			if ( f < BL_MIN )
				NJDISTANCES[edgeaux.id() ] = BL_MIN;
			else if ( f> BL_MAX )
				NJDISTANCES[edgeaux.id() ] = BL_MAX;
			else
				NJDISTANCES[edgeaux.id() ] = f;
		}
		inline bool istaxon ( node nodeaux ) const { return ( nodeaux.degree() <= 1 ); }
		inline edge edge_number ( int n ) const { return MAPIDEDGE[n]; }
		inline bool splits_valid() const { return valid_splits; }
		inline bool split ( edge edgeaux, node nodeaux ) const
		{ return split_bits[ edgeaux.id() * TREE.number_of_nodes() + nodeaux.id() ]; }

		void read_newick ( string newickstring );
		void read_newick2 ( string newickstring );
		void printtree() const;
		void printtreeinfo() const;
		void printNewick ( ostream &os );
		void newick_traverse ( node n, node *ancestor, edge edgeaux, string &s );
		string newick_traverse2 ( bool brlens=true, bool longnames=true );
		double compare_topology ( phylotreeIND &other );
		double compare_topology_2 ( phylotreeIND &other );
		double compare_topology_3 ( phylotreeIND &other );
		double robinson_foulds_distance ( phylotreeIND &other, int debug=0 );
		

		// iterator
		postorder_Iterator postorder_begin ( node root, node father ) const
		{
			postorder_Iterator it = postorder_Iterator ( root, father );
			it.first_node();
			return it;
		}

		postorder_Iterator postorder_begin ( node root ) const
		{
			postorder_Iterator it = postorder_Iterator ( root );
			it.first_node();
			return it;
		}

		preorder_Iterator preorder_begin ( node root, node father ) const
		{
			preorder_Iterator it = preorder_Iterator ( root, father );
			it.first_node();
			return it;
		}

		preorder_Iterator preorder_begin ( node root ) const
		{
			preorder_Iterator it = preorder_Iterator ( root );
			it.first_node();
			return it;
		}


		postorder_Iterator postorder_end ( node root, node father ) const
		{
			postorder_Iterator it = postorder_Iterator ( root, father );
			it.last_node();
			return it;
		}

		postorder_Iterator postorder_end ( node root ) const
		{
			postorder_Iterator it = postorder_Iterator ( root );
			it.last_node();
			return it;
		}

		preorder_Iterator preorder_end ( node root, node father ) const
		{
			preorder_Iterator it = preorder_Iterator ( root, father );
			it.last_node();
			return it;
		}

		preorder_Iterator preorder_end ( node root ) const
		{
			preorder_Iterator it = preorder_Iterator ( root );
			it.last_node();
			return it;
		}


		child_Iterator child_begin ( node root, node father ) const
		{
			child_Iterator it = child_Iterator ( root, father );
			it.first_node();
			return it;
		}

		child_Iterator child_begin ( node root ) const
		{
			child_Iterator it = child_Iterator ( root );
			it.first_node();
			return it;
		}

		child_Iterator child_end ( node root, node father ) const
		{
			child_Iterator it = child_Iterator ( root, father );
			it.last_node();
			return it;
		}

		child_Iterator child_end ( node root ) const
		{
			child_Iterator it = child_Iterator ( root );
			it.last_node();
			return it;
		}


};

template<typename T> const T& select_edge_at_pos ( const list <T> &, int );
#endif

