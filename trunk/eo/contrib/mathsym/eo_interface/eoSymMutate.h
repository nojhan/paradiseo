/*	    
 *             Copyright (C) 2005 Maarten Keijzer
 *
 *          This program is free software; you can redistribute it and/or modify
 *          it under the terms of version 2 of the GNU General Public License as 
 *          published by the Free Software Foundation. 
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program; if not, write to the Free Software
 *          Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef SYMMUTATE_H
#define SYMMUTATE_H

#include <gen/TreeBuilder.h>
#include <gen/NodeSelector.h>

#include <eoSym.h>
#include <eoOp.h>

template <class EoType>
class eoSymSubtreeMutate : public eoMonOp<EoType> {
    
	TreeBuilder& subtree_builder;
	NodeSelector& node_selector;
    public :
	
	eoSymSubtreeMutate(TreeBuilder& _subtree_builder, NodeSelector& _node_selector)
	: subtree_builder(_subtree_builder), node_selector(_node_selector) {}
	
	
	bool operator()(EoType& tomutate) {
	    unsigned xover_point = node_selector.select_node(tomutate).idx();
	    // create subtree
	    Sym newtree = subtree_builder.build_tree(6, true); // TODO, parameterize
	    static_cast<Sym&>(tomutate) = insert_subtree(tomutate, xover_point, newtree);
	    return true;
	}
    
};

/** Class for doing node mutation
 * Two parameters:
 * 
 *	mutation_rate (the rate at which to do mutation)
 *	is_rate_absolute : don't rescale the rate to the size of the tree
 */

extern bool mutate(Sym& sym, double p, const LanguageTable& table);

template <class EoType>
class eoSymNodeMutate : public eoMonOp<EoType> {
    
	LanguageTable& table;
	double own_mutation_rate;
	bool   own_is_rate_absolute;
	

    public:
	
	double& mutation_rate;
	bool& is_rate_absolute;
	    
	eoSymNodeMutate(LanguageTable& _table)
	:   table(_table),
	    own_mutation_rate(1.0),
	    own_is_rate_absolute(false), // this means a probability of node mutation of 1/sym.size()
	    mutation_rate(own_mutation_rate),
	    is_rate_absolute(own_is_rate_absolute)
	{}

	eoSymNodeMutate(LanguageTable& _table, double& _mutation_rate, bool& _is_rate_absolute) 
	:   table(_table),
	    mutation_rate(_mutation_rate),
	    is_rate_absolute(_is_rate_absolute) 
	{}

	
	bool operator()(EoType& _eo) {
	    double p = mutation_rate;
	    if (!is_rate_absolute) p /= _eo.size();

	    return mutate(_eo, p, table);
	}
	
};

/** 
 * Simple constant mutation class, adds gaussian noise (configurable variance) to the individuals
 **/
extern bool mutate_constants(Sym& sym, double stdev);
template <class EoType>
class eoSymConstantMutate : public eoMonOp<EoType> {
    
    double& stdev;

    
    public :
    eoSymConstantMutate(double& _stdev) : stdev(_stdev) {}

    bool operator()(EoType& _eo) {
	return mutate_constants(_eo, stdev);
    }
    
    
};

#endif
