// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "peoSeqPopEval.h"

// (c) OPAC Team, LIFL, August 2005

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

#ifndef __peoSeqPopEval_h
#define __peoSeqPopEval_h

#include <eoEvalFunc.h>

#include "peoPopEval.h"

template <class EOT> class peoSeqPopEval : public peoPopEval <EOT> {

public :

  /** Ctor */  
  peoSeqPopEval (eoEvalFunc <EOT> & __eval);
    
  /** */
  void operator () (eoPop <EOT> & __pop); 
    
private :
  
  eoEvalFunc <EOT> & eval; /* Underlying evaluator */

};

template <class EOT> 
peoSeqPopEval <EOT> :: peoSeqPopEval (eoEvalFunc <EOT> & __eval
				      ) : eval (__eval) {
  
}

template <class EOT> 
void peoSeqPopEval <EOT> :: operator () (eoPop <EOT> & __pop) {
  
  for (unsigned i = 0; i < __pop.size (); i ++)
    eval (__pop [i]);
}

#endif
