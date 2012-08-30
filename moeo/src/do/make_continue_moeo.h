/*
* <make_continue_moeo.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/
//-----------------------------------------------------------------------------

#ifndef MAKE_CONTINUE_MOEO_H_
#define MAKE_CONTINUE_MOEO_H_

#include <eoCombinedContinue.h>
#include <eoGenContinue.h>
#include <eoEvalContinue.h>
#include <eoFitContinue.h>
#include <eoTimeContinue.h>
#ifndef _MSC_VER
#include <eoCtrlCContinue.h>
#endif
#include <utils/eoParser.h>
#include <utils/eoState.h>


/**
 * Helper function
 * @param _combined the eoCombinedContinue object
 * @param _cont the eoContinue to add
 */
template <class MOEOT>
eoCombinedContinue<MOEOT> * make_combinedContinue(eoCombinedContinue<MOEOT> *_combined, eoContinue<MOEOT> *_cont)
{
  if (_combined)		   // already exists
    _combined->add(*_cont);
  else
    _combined = new eoCombinedContinue<MOEOT>(*_cont);
  return _combined;
}


/**
 * This functions allows to build a eoContinue for multi-objective optimization from the parser (partly taken from make_continue_pareto.h)
 * @param _parser the parser
 * @param _state to store allocated objects
 * @param _eval the funtions evaluator
 */
template <class MOEOT>
eoContinue<MOEOT> & do_make_continue_moeo(eoParser& _parser, eoState& _state, eoEvalFuncCounter<MOEOT> & _eval)
{
  // the combined continue - to be filled
  eoCombinedContinue<MOEOT> *continuator = NULL;
  // First the eoGenContinue - need a default value so you can run blind
  // but we also need to be able to avoid it <--> 0
  eoValueParam<unsigned int>& maxGenParam = _parser.createParam((unsigned int)(100), "maxGen", "Maximum number of generations (0 = none)",'G',"Stopping criterion");
  if (maxGenParam.value()) // positive: -> define and store
    {
      eoGenContinue<MOEOT> *genCont = new eoGenContinue<MOEOT>(maxGenParam.value());
      _state.storeFunctor(genCont);
      // and "add" to combined
      continuator = make_combinedContinue<MOEOT>(continuator, genCont);
    }
  // maxEval
  eoValueParam<unsigned long>& maxEvalParam = _parser.getORcreateParam((unsigned long)(0), "maxEval", "Maximum number of evaluations (0 = none)", 'E', "Stopping criterion");
  if (maxEvalParam.value())
    {
      eoEvalContinue<MOEOT> *evalCont = new eoEvalContinue<MOEOT>(_eval, maxEvalParam.value());
      _state.storeFunctor(evalCont);
      // and "add" to combined
      continuator = make_combinedContinue<MOEOT>(continuator, evalCont);
    }
  // maxTime
  eoValueParam<unsigned long>& maxTimeParam = _parser.getORcreateParam((unsigned long)(0), "maxTime", "Maximum running time in seconds (0 = none)", 'T', "Stopping criterion");
  if (maxTimeParam.value()) // positive: -> define and store
    {
      eoTimeContinue<MOEOT> *timeCont = new eoTimeContinue<MOEOT>(maxTimeParam.value());
      _state.storeFunctor(timeCont);
      // and "add" to combined
      continuator = make_combinedContinue<MOEOT>(continuator, timeCont);
    }
  // CtrlC
#ifndef _MSC_VER
  // the CtrlC interception (Linux only I'm afraid)
  eoCtrlCContinue<MOEOT> *ctrlCCont;
  eoValueParam<bool>& ctrlCParam = _parser.createParam(true, "CtrlC", "Terminate current generation upon Ctrl C",'C', "Stopping criterion");
  if (ctrlCParam.value())
    {
      ctrlCCont = new eoCtrlCContinue<MOEOT>;
      // store
      _state.storeFunctor(ctrlCCont);
      // add to combinedContinue
      continuator = make_combinedContinue<MOEOT>(continuator, ctrlCCont);
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

#endif /*MAKE_CONTINUE_MOEO_H_*/
