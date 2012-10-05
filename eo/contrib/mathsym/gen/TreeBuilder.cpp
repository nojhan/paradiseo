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

#include <utils/eoRNG.h>
#include "TreeBuilder.h"

Sym TreeBuilder::make_terminal() const {
    if (rng.flip(vcprob)) {
	return table.get_random_var();
    }

    return table.get_random_const();
}

Sym TreeBuilder::build_tree(unsigned max_depth,  bool grow) const {
    if (max_depth == 0 || grow && rng.random(2) == 0) { 
	return make_terminal();
    }

    // pick a random function, no matter what arity
    
    functor_t func = table.get_random_function();
    
    SymVec args(func.arity);
    
    for (unsigned i = 0; i < args.size(); ++i) {
	args[i] = build_tree(max_depth-1, grow);
    }

    return Sym(func.token, args);
}

