// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoPopEval.h"

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

#ifndef eoPopEval_h
#define eoPopEval_h

#include <eoEvalFunc.h>
#include <apply.h>
#include <eoPopAlgo.h>

/**
   An evaluator which computes fitnesses of a given population in a
   sequentially manner. 
 */

template <class EOT> class eoPopEval : public eoPopAlgo <EOT> {
  
public :

  /**
     Constructor
  */

  eoPopEval (eoEvalFunc <EOT> & _eval) : eval(_eval) {
    
  }

  /**
     Values sequentially each EO from the pop. given in parameter.
   */
  
  void operator () (eoPop <EOT> & _pop) {
    apply <EOT> (eval, _pop) ;
  }

private :
  
  eoEvalFunc <EOT> & eval ;
} ;

#endif

