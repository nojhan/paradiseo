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

#include <TreeBuilder.h>
#include <NodeSelector.h>

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
	    unsigned xover_point = node_selector.select_node(tomutate);
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

template <class EoType>
class eoSymNodeMutate : public eoMonOp<EoType> {
    
	LanguageTable& table;
	double own_mutation_rate;
	bool   own_is_rate_absolute;
	
	// these two can (should?) move to an impl file
	bool mutate(Sym& sym, double p) {
	    std::pair<Sym, bool> r = do_mutate(sym, p);
	    sym = r.first;
	    return r.second;
	}
	
	std::pair<Sym, bool> do_mutate(Sym sym, double p) {
	    
	    bool changed = false;
	    SymVec args = sym.args();
	    if (rng.flip(p)) {
		token_t new_token = table.get_random_function( args.size());
		if (new_token != sym.token()) changed = true;
		sym = Sym(new_token, args);
	    }

	    for (unsigned i = 0; i < args.size(); ++i) {
		std::pair<Sym,bool> r = do_mutate(args[i], p);	
		changed |= r.second;
		if (r.second) 
		    args[i] = r.first;
	    }

	    if (changed)
		return std::make_pair(Sym(sym.token(), args), true);
	    // else
	    return std::make_pair(sym, false);
	}
	
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

	    return mutate(_eo, p);
	}
	
};

#endif
