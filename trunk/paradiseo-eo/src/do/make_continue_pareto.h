// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_continue_pareto.h
// (c) Maarten Keijzer, Marc Schoenauer and GeNeura Team, 2000
/* 
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@inria.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _make_continue_pareto_h
#define _make_continue_pareto_h

/*
Contains the templatized version of parser-based choice of stopping criterion
for Pareto optimization (e.g. no "... without improvement" criterion
It can then be instantiated, and compiled on its own for a given EOType
(see e.g. in dir ga, ga.cpp)
*/

// Continuators - all include eoContinue.h
#include <eoCombinedContinue.h>
#include <eoGenContinue.h>
#include <eoEvalContinue.h>
#include <eoFitContinue.h>
#ifndef _MSC_VER
#include <eoCtrlCContinue.h>  // CtrlC handling (using 2 global variables!)
#endif

  // also need the parser and param includes
#include <utils/eoParser.h>
#include <utils/eoState.h>


/////////////////// helper function ////////////////
template <class Indi>
eoCombinedContinue<Indi> * make_combinedContinue(eoCombinedContinue<Indi> *_combined, eoContinue<Indi> *_cont)
{
  if (_combined)		   // already exists
    _combined->add(*_cont);
  else
    _combined = new eoCombinedContinue<Indi>(*_cont);
  return _combined;
}

///////////// The make_continue function
template <class Indi>
eoContinue<Indi> & do_make_continue_pareto(eoParser& _parser, eoState& _state, eoEvalFuncCounter<Indi> & _eval)
{
  //////////// Stopping criterion ///////////////////
  // the combined continue - to be filled
  eoCombinedContinue<Indi> *continuator = NULL;

  // for each possible criterion, check if wanted, otherwise do nothing

  // First the eoGenContinue - need a default value so you can run blind
  // but we also need to be able to avoid it <--> 0
  eoValueParam<unsigned>& maxGenParam = _parser.createParam(unsigned(100), "maxGen", "Maximum number of generations () = none)",'G',"Stopping criterion");

    if (maxGenParam.value()) // positive: -> define and store
      {
	eoGenContinue<Indi> *genCont = new eoGenContinue<Indi>(maxGenParam.value());
	_state.storeFunctor(genCont);
	// and "add" to combined
	continuator = make_combinedContinue<Indi>(continuator, genCont);
      }

#ifndef _MSC_VER
    // the CtrlC interception (Linux only I'm afraid)
    eoCtrlCContinue<Indi> *ctrlCCont;
    eoValueParam<bool>& ctrlCParam = _parser.createParam(true, "CtrlC", "Terminate current generation upon Ctrl C",'C', "Stopping criterion");
    if (_parser.isItThere(ctrlCParam))
      {
	ctrlCCont = new eoCtrlCContinue<Indi>;
	// store
	_state.storeFunctor(ctrlCCont);
	// add to combinedContinue
	continuator = make_combinedContinue<Indi>(continuator, ctrlCCont);
      }
#endif

    // now check that there is at least one!
    if (!continuator)
      throw std::runtime_error("You MUST provide a stopping criterion");
  // OK, it's there: store in the eoState
  _state.storeFunctor(continuator);

  // and return
    return *continuator;
}

#endif
