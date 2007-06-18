// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_continue_moeo.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
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
    eoValueParam<unsigned>& maxGenParam = _parser.createParam(unsigned(100), "maxGen", "Maximum number of generations (0 = none)",'G',"Stopping criterion");
    if (maxGenParam.value()) // positive: -> define and store
    {
        eoGenContinue<MOEOT> *genCont = new eoGenContinue<MOEOT>(maxGenParam.value());
        _state.storeFunctor(genCont);
        // and "add" to combined
        continuator = make_combinedContinue<MOEOT>(continuator, genCont);
    }
    // maxEval
    eoValueParam<unsigned long>& maxEvalParam = _parser.getORcreateParam((unsigned long)0, "maxEval", "Maximum number of evaluations (0 = none)", 'E', "Stopping criterion");
    if (maxEvalParam.value())
    {
        eoEvalContinue<MOEOT> *evalCont = new eoEvalContinue<MOEOT>(_eval, maxEvalParam.value());
        _state.storeFunctor(evalCont);
        // and "add" to combined
        continuator = make_combinedContinue<MOEOT>(continuator, evalCont);
    }
    // maxTime
    eoValueParam<unsigned long>& maxTimeParam = _parser.getORcreateParam((unsigned long)0, "maxTime", "Maximum running time in seconds (0 = none)", 'T', "Stopping criterion");
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
    if (_parser.isItThere(ctrlCParam))
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
