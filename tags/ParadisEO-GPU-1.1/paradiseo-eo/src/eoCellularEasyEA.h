// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoCellularEasyEA.h"

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

#ifndef eoCellularEasyEA_h
#define eoCellularEasyEA_h

#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoSelectOne.h>
#include <eoPopEvalFunc.h>
#include <eoAlgo.h>
#include <eoOp.h>

/**
   The abstract cellular easy algorithm.

   @ingroup Algorithms
 */
template <class EOT> class eoCellularEasyEA : public eoAlgo <EOT> {

public :

  /**
     Two constructors
   */

  eoCellularEasyEA (eoContinue <EOT> & _cont, // Stop. criterion
                    eoEvalFunc <EOT> & _eval, // Evaluation function
                    eoSelectOne <EOT> & _sel_neigh, // To choose a partner
                    eoBinOp <EOT> & _cross, // Cross-over operator
                    eoMonOp <EOT> & _mut, // Mutation operator
                    eoSelectOne <EOT> & _sel_repl /* Which to keep between the new
                                                     child and the old individual ? */
                    ) :
    cont (_cont),
    eval (_eval),
    popEval (_eval),
    sel_neigh (_sel_neigh),
    cross (_cross),
    mut (_mut),
    sel_child (eoSelectFirstOne ()),
    sel_repl (_sel_repl) {

  }

  eoCellularEasyEA (eoContinue <EOT> & _cont,
                    eoEvalFunc <EOT> & _eval,
                    eoSelectOne <EOT> & _sel_neigh,
                    eoQuadOp <EOT> & _cross,
                    eoMonOp <EOT> & _mut,
                    eoSelectOne <EOT> & _sel_child, /* To choose one from
                                                       the both children */
                    eoSelectOne <EOT> & _sel_repl
                    ) :
    cont (_cont),
    eval (_eval),
    popEval (_eval),
    sel_neigh (_sel_neigh),
    cross (_cross),
    mut (_mut),
    sel_child (_sel_child),
    sel_repl (_sel_repl) {

  }

  /**
     For a given population.
   */

  void operator () (eoPop <EOT> & pop) {

    do {

      for (unsigned i = 0 ; i < pop.size () ; i ++) {

        // Who are neighbouring to the current individual ?
        eoPop <EOT> neigh = neighbours (pop, i) ;

        // To select a partner
        EOT part, old_sol = pop [i] ;
        part = sel_neigh (neigh) ;

        // To perform cross-over
        cross (pop [i], part) ;

        // To perform mutation
        mut (pop [i]) ;
        mut (part) ;

        pop [i].invalidate () ;
        part.invalidate () ;
        eval (pop [i]) ;
        eval (part) ;

        // To choose one of the two children ...
        eoPop <EOT> pop_loc ;
        pop_loc.push_back (pop [i]) ;
        pop_loc.push_back (part) ;

        pop [i] = sel_child (pop_loc) ;

        // To choose only one between the new made child and the old individual
        pop_loc.clear () ;
        pop_loc.push_back (pop [i]) ;

        pop_loc.push_back (old_sol) ;

        pop [i] = sel_repl (pop_loc) ;
      }

    } while (cont (pop)) ;
  }

protected :

  virtual eoPop <EOT> neighbours (const eoPop <EOT> & pop, int rank) = 0 ;

private :

  eoContinue <EOT> & cont ;
  eoEvalFunc <EOT> & eval ;
  eoPopLoopEval <EOT> popEval ;
  eoSelectOne <EOT> & sel_neigh ;
  eoBF <EOT &, EOT &, bool> & cross ;
  eoMonOp <EOT> & mut ;
  eoSelectOne <EOT> & sel_child ;
  eoSelectOne <EOT> & sel_repl ;

  class eoSelectFirstOne : public eoSelectOne <EOT> {

  public :

    const EOT & operator () (const eoPop <EOT> & pop) {

      return pop [0] ;
    }

  } ;

} ;

#endif
