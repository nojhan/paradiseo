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

#ifndef EOSYMINIT_H
#define EOSYMINIT_H

#include <eoInit.h>
#include <gen/TreeBuilder.h>

/** Default initializer, Koza style */
template <class EoType>
class eoSymInit : public eoInit<EoType> {
    
    TreeBuilder& builder;
    
    double	own_grow_prob;
    unsigned	own_max_depth;
    
    
    double& grow_prob;
    unsigned& max_depth;

    public:
	
	/** By default build ramped half and half with max depth 6 */
	eoSymInit(TreeBuilder& _builder) 
	:   builder(_builder), 
	    own_grow_prob(0.5), 
	    own_max_depth(6),
	    grow_prob(own_grow_prob), 
	    max_depth(own_max_depth) 
	{}

	/** Control the grow_prob and max_depth externally */
	eoSymInit(TreeBuilder& _builder, double& _grow_prob, unsigned& _max_depth) 
	:   builder(_builder), 
	    grow_prob(_grow_prob), 
	    max_depth(_max_depth) 
	{}
   
	/** build the tree */
	void operator()(EoType& tree) {
	    int depth_to_use = rng.random(max_depth-2) + 2; // two levels minimum
	    builder.build_tree(tree, depth_to_use, rng.flip(grow_prob));
	}
    
};

#endif

