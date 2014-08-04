// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_continue.h
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
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _make_continue_h
#define _make_continue_h

/*
Contains the templatized version of parser-based choice of stopping criterion
It can then be instantiated, and compiled on its own for a given EOType
(see e.g. in dir ga, ga.cpp)
*/

// Continuators - all include eoContinue.h
#include <eoCombinedContinue.h>
#include <eoGenContinue.h>
#include <eoSteadyFitContinue.h>
#include <eoEvalContinue.h>
#include <eoFitContinue.h>
#ifndef _MSC_VER
#include <eoCtrlCContinue.h>  // CtrlC handling (using 2 global variables!)
#endif

  // also need the parser and param includes
#include <utils/eoParser.h>
#include <utils/eoState.h>


/////////////////// the stopping criterion ////////////////
 /**
 * @ingroup Builders
 */
template <class Indi>
eoCombinedContinue<Indi> * make_combinedContinue(eoCombinedContinue<Indi> *_combined, eoContinue<Indi> *_cont)
{
  if (_combined)                   // already exists
    _combined->add(*_cont);
  else
    _combined = new eoCombinedContinue<Indi>(*_cont);
  return _combined;
}

/**
 *
 * @ingroup Builders
 */
template <class Indi>
eoContinue<Indi> & do_make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<Indi> & _eval)
{
  //////////// Stopping criterion ///////////////////
  // the combined continue - to be filled
  eoCombinedContinue<Indi> *continuator = NULL;

  // for each possible criterion, check if wanted, otherwise do nothing

  // First the eoGenContinue - need a default value so you can run blind
  // but we also need to be able to avoid it <--> 0
  eoValueParam<unsigned>& maxGenParam = _parser.getORcreateParam(unsigned(100), "maxGen", "Maximum number of generations () = none)",'G',"Stopping criterion");

    if (maxGenParam.value()) // positive: -> define and store
      {
        eoGenContinue<Indi> *genCont = new eoGenContinue<Indi>(maxGenParam.value());
        _state.storeFunctor(genCont);
        // and "add" to combined
        continuator = make_combinedContinue<Indi>(continuator, genCont);
      }

  // the steadyGen continue - only if user imput
  eoValueParam<unsigned>& steadyGenParam = _parser.createParam(unsigned(100), "steadyGen", "Number of generations with no improvement",'s', "Stopping criterion");
  eoValueParam<unsigned>& minGenParam = _parser.createParam(unsigned(0), "minGen", "Minimum number of generations",'g', "Stopping criterion");
    if (_parser.isItThere(steadyGenParam))
      {
        eoSteadyFitContinue<Indi> *steadyCont = new eoSteadyFitContinue<Indi>
          (minGenParam.value(), steadyGenParam.value());
        // store
        _state.storeFunctor(steadyCont);
        // add to combinedContinue
        continuator = make_combinedContinue<Indi>(continuator, steadyCont);
      }

  // Same thing with Eval - but here default value is 0
  eoValueParam<unsigned long>& maxEvalParam
      = _parser.getORcreateParam((unsigned long)0, "maxEval",
                                 "Maximum number of evaluations (0 = none)",
                                 'E', "Stopping criterion");

    if (maxEvalParam.value()) // positive: -> define and store
      {
        eoEvalContinue<Indi> *evalCont = new eoEvalContinue<Indi>(_eval, maxEvalParam.value());
        _state.storeFunctor(evalCont);
        // and "add" to combined
        continuator = make_combinedContinue<Indi>(continuator, evalCont);
      }
    /*
  // the steadyEval continue - only if user imput
  eoValueParam<unsigned>& steadyGenParam = _parser.createParam(unsigned(100), "steadyGen", "Number of generations with no improvement",'s', "Stopping criterion");
  eoValueParam<unsigned>& minGenParam = _parser.createParam(unsigned(0), "minGen", "Minimum number of generations",'g', "Stopping criterion");
    if (_parser.isItThere(steadyGenParam))
      {
        eoSteadyGenContinue<Indi> *steadyCont = new eoSteadyFitContinue<Indi>
          (minGenParam.value(), steadyGenParam.value());
        // store
        _state.storeFunctor(steadyCont);
        // add to combinedContinue
        continuator = make_combinedContinue<Indi>(continuator, steadyCont);
      }
    */
    // the target fitness
    eoFitContinue<Indi> *fitCont;
    eoValueParam<double>& targetFitnessParam = _parser.createParam(double(0.0), "targetFitness", "Stop when fitness reaches",'T', "Stopping criterion");
    if (_parser.isItThere(targetFitnessParam))
      {
        fitCont = new eoFitContinue<Indi>
          (targetFitnessParam.value());
        // store
        _state.storeFunctor(fitCont);
        // add to combinedContinue
        continuator = make_combinedContinue<Indi>(continuator, fitCont);
      }

#ifndef _MSC_VER
    // the CtrlC interception (Linux only I'm afraid)
    eoCtrlCContinue<Indi> *ctrlCCont;
    eoValueParam<bool>& ctrlCParam = _parser.createParam(false, "CtrlC", "Terminate current generation upon Ctrl C",'C', "Stopping criterion");
    if (ctrlCParam.value())
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
