// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoDistEvalEasyEA.h"

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

#ifndef eoDistEvalEasyEA_h
#define eoDistEvalEasyEA_h

#include <eoEasyEA.h>
#include <paradisEO/comm/eoListener.h>
#include <paradisEO/eval/eoDistPopEvalFunc.h>

/**
   Intended for evaluation CPU cost evolutionnary algorithm ...
 */

template <class EOT> class eoDistEvalEasyEA : public eoEasyEA <EOT> {
  
public :
  
  /**
     Constructor. Caution ! The provided label is the one
     of distributed evaluators to search for and not those of
     the master program, which is isn't necessary.
   */
  
  eoDistEvalEasyEA  (eoListener <EOT> & _listen,
		     eoEasyEA <EOT> & _ea,
		     string _label
		     ) :
    pop_eval (eoDistPopEvalFunc <EOT> (_listen, _label, _ea.eval)),
    eoEasyEA <EOT> (_ea.continuator,
		    pop_eval,
		    _ea.breed,
		    _ea.replace
		    ) {
  }

private :

  eoDistPopEvalFunc <EOT> pop_eval ;
  
} ;

#endif
