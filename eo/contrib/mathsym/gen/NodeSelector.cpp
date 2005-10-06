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

#include "NodeSelector.h"
#include "Sym.h"

#include <utils/eoRNG.h>

unsigned RandomNodeSelector::select_node(Sym sym) const {
    return rng.random(sym.size());
}

unsigned BiasedNodeSelector::select_node(Sym sym) const {
    
    unsigned p = rng.random(sym.size());
    for (unsigned i = 0; i < nRounds; ++i) {
	Sym res = get_subtree(sym, p);
	
	if (res.args().size() > 0) break;
	
	p = rng.random(sym.size());
    }
    
    return p;
}

