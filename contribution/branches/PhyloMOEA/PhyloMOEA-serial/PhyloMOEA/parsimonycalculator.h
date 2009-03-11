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
#ifndef PARSIMONYCALCULATOR_H
#define PARSIMONYCALCULATOR_H

#include <phylotreeIND.h>

/**
@author Waldo Cancino
*/
class ParsimonyCalculator
{
	private:
		phylotreeIND *tree_ptr;  // point to a tree;
		unsigned char *set_memory_allocate;
		unsigned char *char_memory_allocate;
		unsigned char *set_internal_memory_allocate; // sequences for sets in an internal node
		unsigned char *set_taxon_memory_allocate; // sequences for sets in an taxon nodes
		unsigned char *char_internal_memory_allocate; // assignment chars for internal nodes
		unsigned char *char_taxon_memory_allocate; // assignment chars for taxon nodes
		bool invalid_set_taxons;
		long int parsimony;
		node_map<unsigned char*> set_internal; // internal set for fitch phase I
		node_map<unsigned char*> char_internal; // internal characters for fitch phase II
		Sequences *SeqData; // maintain the current pattern
		int set_intersection( unsigned char *a, unsigned char *b, unsigned char *result);
		void set_union( unsigned char *a, unsigned char *b, unsigned char *result);
		int set_parsimony( unsigned char *a, unsigned char *b, unsigned char *result);
		int node_parsimony( node a, node b, unsigned char *result);
		void fitch_post_order(node n, node *antecessor);
		void seq_assignment(node n, node ancestor);
		void fitch_pre_order(node n, node *antecessor);
		void init_set_char_taxon(node n);
		void init_sets_chars();

	public:
		void set_tree(phylotreeIND &t);
		long int fitch();
		void save_informative(char *fn);
    		ParsimonyCalculator(phylotreeIND &t);
    		~ParsimonyCalculator();

};
#endif
