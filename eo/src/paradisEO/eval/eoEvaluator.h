// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoEvaluator.h"

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

#ifndef eoEvaluator_h
#define eoEvaluator_h

#include <paradisEO/eoPopAgent.h>
#include <paradisEO/eval/eoPopEval.h>
#include <eoEvalFunc.h>

#include <string>

/**
   A special pop. based agent, applied to eval sub_populations.
   Rather intended for evolutionnary algorithms, with evaluations
   performed remotely.
*/

template <class EOT> class eoEvaluator : public eoPopAgent <EOT> {
  
public :

  /**
     Constructor
  */ 
  
  eoEvaluator (std::string _label,
	       eoListener <EOT> & _listen,
	       eoEvalFunc <EOT> & _eval
	       ) :
    pop_eval (_eval),
    eoPopAgent <EOT> (_label,
		      _listen,
		      pop_eval) {
    
  }
  
private :
  
  eoPopEval <EOT> pop_eval ;

} ;

#endif
