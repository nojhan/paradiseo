// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoToricCellularEasyEA.h"

// (c) OPAC Team, LIFL, 2002

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: cahon@lifl.fr
*/

#ifndef eoToricCellularEasyEA_h
#define eoToricCellularEasyEA_h

#include <eoCellularEasyEA.h>

#include <math.h>

template <class EOT> class eoToricCellularEasyEA : public eoCellularEasyEA <EOT> {
  
public :
  
  eoToricCellularEasyEA (eoContinue <EOT> & _cont,
			 eoEvalFunc <EOT> & _eval,
			 eoSelectOne <EOT> & _sel_neigh,
			 eoBinOp <EOT> & _cross,
			 eoMonOp <EOT> & _mut,
			 eoSelectOne <EOT> & _sel_repl
			 ) :
    eoCellularEasyEA <EOT> (_cont,
			    _eval,
			    _sel_neigh,
			    _cross,
			    _mut,
			    _sel_repl) {
    
  }
  
  eoToricCellularEasyEA (eoContinue <EOT> & _cont,
			 eoEvalFunc <EOT> & _eval,
			 eoSelectOne <EOT> & _sel_neigh,
			 eoQuadOp <EOT> & _cross,
			 eoMonOp <EOT> & _mut,
			 eoSelectOne <EOT> & _sel_child,
			 eoSelectOne <EOT> & _sel_repl
			 ) :
    eoCellularEasyEA <EOT> (_cont,
			    _eval,
			    _sel_neigh,
			    _cross,
			    _mut,
			    _sel_child,
			    _sel_repl) {
    
  }

  // Take care :-). The size of the population must be a square number ! (9, 16, ...)

  virtual eoPop <EOT> neighbours (const eoPop <EOT> & pop, int rank) {
    
    int dim2 = pop.size () ;
    int dim = (int) sqrt (dim2) ;
    int j = rank ;

    eoPop <EOT> neigh ;
    neigh.push_back (pop [j < dim ? dim2 - dim + j : j - dim]) ;
    neigh.push_back (pop [(j + dim) % dim2]) ;
    neigh.push_back (pop [(j + 1) % dim != 0 ? j + 1 : j + 1 - dim]) ;
    neigh.push_back (pop [j % dim != 0 ? j - 1 : j + dim - 1]) ;
   
    return neigh ;
  } 
  
} ;

#endif
