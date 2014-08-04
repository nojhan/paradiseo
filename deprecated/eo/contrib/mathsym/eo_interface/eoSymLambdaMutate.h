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

#ifndef SYMLAMBDAMUTATE_H
#define SYMLAMBDAMUTATE_H

#include <eoOp.h>

class NodeSelector;
class Sym;
extern Sym compress(Sym, NodeSelector&);
extern Sym expand(Sym, NodeSelector&);


template <class EoType>
class eoSymLambdaMutate : public eoMonOp<EoType> {
    NodeSelector& selector;
    public :
	eoSymLambdaMutate(NodeSelector& s) : selector(s) {}
	
	bool operator()(EoType& tomutate) {
	    if (rng.flip()) {
		tomutate.set( expand(tomutate, selector));
	    } else {
		tomutate.set( compress(tomutate, selector));
	    }
	    return true; 
	}
    
};


#endif
