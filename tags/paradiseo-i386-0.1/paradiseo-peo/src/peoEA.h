// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "peoEA.h"

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

#ifndef __peoEA_h
#define __peoEA_h

#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoSelect.h>
#include <eoPopEvalFunc.h>
#include <eoReplacement.h>

#include "runner.h"
#include "peoPopEval.h"
#include "peoTransform.h"
#include "peo_debug.h"

template <class EOT> class peoEA : public Runner {

public :

  /** Constructor */
  peoEA (eoContinue <EOT> & __cont,
	 peoPopEval <EOT> & __pop_eval,
	 eoSelect <EOT> & __select,
	 peoTransform <EOT> & __trans,
	 eoReplacement <EOT> & __replace
	 );
  
  void run ();
  
  void operator () (eoPop <EOT> & __pop);

private :

  eoContinue <EOT> & cont;
  peoPopEval <EOT> & pop_eval;
  eoSelect <EOT> & select;
  peoTransform <EOT> & trans;
  eoReplacement <EOT> & replace;
  eoPop <EOT> * pop;
	      
};

template <class EOT> peoEA <EOT> :: peoEA (eoContinue <EOT> & __cont,
					   peoPopEval <EOT> & __pop_eval,
					   eoSelect <EOT> & __select,
					   peoTransform <EOT> & __trans,	    
					   eoReplacement <EOT> & __replace
					   ) : cont (__cont),
					       pop_eval (__pop_eval),
					       select (__select),
					       trans (__trans),
					       replace (__replace) {
  
  trans.setOwner (* this);
  pop_eval.setOwner (* this);
}

template <class EOT> void peoEA <EOT> :: operator () (eoPop <EOT> & __pop
						      ) {

  pop = & __pop;
}

template <class EOT> void peoEA <EOT> :: run () {
  
  printDebugMessage ("performing the first evaluation of the population.");
  pop_eval (* pop);
  
  do {
    eoPop <EOT> off;
    printDebugMessage ("performing the selection step.");
    select (* pop, off);
    trans (off);
    printDebugMessage ("performing the evaluation of the population.");
    pop_eval (off);
    printDebugMessage ("performing the replacement of the population.");
    replace (* pop, off);    
    printDebugMessage ("deciding of the continuation.");
  
  } while (cont (* pop));    
}

#endif
