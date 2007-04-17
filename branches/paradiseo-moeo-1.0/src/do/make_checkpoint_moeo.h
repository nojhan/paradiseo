// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_checkpoint_moeo.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MAKE_CHECKPOINT_MOEO_H_
#define MAKE_CHECKPOINT_MOEO_H_

#include <stdlib.h>
#include <sstream>
#include <eoContinue.h>
#include <eoEvalFuncCounter.h>
#include <utils/checkpointing>
#include <utils/selectors.h>
#include <utils/eoParser.h>
#include <utils/eoState.h>
#include <moeoArchiveUpdater.h>
#include <moeoArchiveObjectiveVectorSavingUpdater.h>
#include <metric/moeoBinaryMetricSavingUpdater.h>
#include <metric/moeoContributionMetric.h>
#include <metric/moeoEntropyMetric.h>

bool testDirRes(std::string _dirName, bool _erase);

/**
 * This functions allows to build an eoCheckPoint for multi-objective optimization from the parser (partly taken from make_checkpoint_pareto.h)
 * @param _parser the parser
 * @param _state to store allocated objects
 * @param _eval the funtions evaluator
 * @param _continue the stopping crietria
 * @param _pop the population
 * @param _archive the archive of non-dominated solutions
 */
template < class MOEOT >
eoCheckPoint < MOEOT > & do_make_checkpoint_moeo (eoParser & _parser, eoState & _state, eoEvalFuncCounter < MOEOT > & _eval, eoContinue < MOEOT > & _continue, eoPop < MOEOT > & _pop, moeoArchive < MOEOT > & _archive)
{
    eoCheckPoint < MOEOT > & checkpoint = _state.storeFunctor(new eoCheckPoint < MOEOT > (_continue));
    /* the objective vector type */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;

    ///////////////////
    // Counters
    //////////////////
    // is nb Eval to be used as counter?
    //bool useEval = _parser.getORcreateParam(true, "useEval", "Use nb of eval. as counter (vs nb of gen.)", '\0', "Output").value();
    // Create anyway a generation-counter parameter
    eoValueParam<unsigned> *generationCounter = new eoValueParam<unsigned>(0, "Gen.");
    // Create an incrementor (sub-class of eoUpdater).
    eoIncrementor<unsigned> & increment = _state.storeFunctor( new eoIncrementor<unsigned>(generationCounter->value()) );
    // Add it to the checkpoint
    checkpoint.add(increment);
    // dir for DISK output
    std::string & dirName =  _parser.getORcreateParam(std::string("Res"), "resDir", "Directory to store DISK outputs", '\0', "Output").value();
    // shoudl we empty it if exists
    eoValueParam<bool>& eraseParam = _parser.getORcreateParam(true, "eraseDir", "erase files in dirName if any", '\0', "Output");
    bool dirOK = false;		   // not tested yet

    // Dump of the whole population
    //-----------------------------
    bool printPop = _parser.getORcreateParam(false, "printPop", "Print sorted pop. every gen.", '\0', "Output").value();
    eoSortedPopStat<MOEOT> * popStat;
    if ( printPop ) // we do want pop dump
    {
        popStat = & _state.storeFunctor(new eoSortedPopStat<MOEOT>);
        checkpoint.add(*popStat);
    }

    //////////////////////////////////
    // State savers
    //////////////////////////////
    // feed the state to state savers
    // save state every N  generation
    eoValueParam<unsigned>& saveFrequencyParam = _parser.createParam(unsigned(0), "saveFrequency", "Save every F generation (0 = only final state, absent = never)", '\0', "Persistence" );
    if (_parser.isItThere(saveFrequencyParam))
    {
        // first make sure dirName is OK
        if (! dirOK )
            dirOK = testDirRes(dirName, eraseParam.value()); // TRUE
        unsigned freq = (saveFrequencyParam.value()>0 ? saveFrequencyParam.value() : UINT_MAX );
#ifdef _MSVC
        std::string stmp = dirName + "\generations";
#else
        std::string stmp = dirName + "/generations";
#endif
        eoCountedStateSaver *stateSaver1 = new eoCountedStateSaver(freq, _state, stmp);
        _state.storeFunctor(stateSaver1);
        checkpoint.add(*stateSaver1);
    }
    // save state every T seconds
    eoValueParam<unsigned>& saveTimeIntervalParam = _parser.getORcreateParam(unsigned(0), "saveTimeInterval", "Save every T seconds (0 or absent = never)", '\0',"Persistence" );
    if (_parser.isItThere(saveTimeIntervalParam) && saveTimeIntervalParam.value()>0)
    {
        // first make sure dirName is OK
        if (! dirOK )
            dirOK = testDirRes(dirName, eraseParam.value()); // TRUE
#ifdef _MSVC
        std::string stmp = dirName + "\time";
#else
        std::string stmp = dirName + "/time";
#endif
        eoTimedStateSaver *stateSaver2 = new eoTimedStateSaver(saveTimeIntervalParam.value(), _state, stmp);
        _state.storeFunctor(stateSaver2);
        checkpoint.add(*stateSaver2);
    }

    ///////////////////
    // Archive
    //////////////////
    // update the archive every generation
    bool updateArch = _parser.getORcreateParam(true, "updateArch", "Update the archive at each gen.", '\0', "Evolution Engine").value();
    if (updateArch)
    {
        moeoArchiveUpdater < MOEOT > * updater = new moeoArchiveUpdater < MOEOT > (_archive, _pop);
        _state.storeFunctor(updater);
        checkpoint.add(*updater);
    }
    // store the objective vectors contained in the archive every generation
    bool storeArch = _parser.getORcreateParam(false, "storeArch", "Store the archive's objective vectors at each gen.", '\0', "Output").value();
    if (storeArch)
    {
        if (! dirOK )
            dirOK = testDirRes(dirName, eraseParam.value()); // TRUE
#ifdef _MSVC
        std::string stmp = dirName + "\arch";
#else
        std::string stmp = dirName + "/arch";
#endif
        moeoArchiveObjectiveVectorSavingUpdater < MOEOT > * save_updater = new moeoArchiveObjectiveVectorSavingUpdater < MOEOT > (_archive, stmp);
        _state.storeFunctor(save_updater);
        checkpoint.add(*save_updater);
    }
    // store the contribution of the non-dominated solutions
    bool cont = _parser.getORcreateParam(false, "contribution", "Store the contribution of the archive at each gen.", '\0', "Output").value();
    if (cont)
    {
        if (! dirOK )
            dirOK = testDirRes(dirName, eraseParam.value()); // TRUE
#ifdef _MSVC
        std::string stmp = dirName + "\contribution";
#else
        std::string stmp = dirName + "/contribution";
#endif
        moeoContributionMetric < ObjectiveVector > * contribution = new moeoContributionMetric < ObjectiveVector >;
        moeoBinaryMetricSavingUpdater < MOEOT > * contribution_updater = new moeoBinaryMetricSavingUpdater < MOEOT > (*contribution, _archive, stmp);
        _state.storeFunctor(contribution_updater);
        checkpoint.add(*contribution_updater);
    }
    // store the entropy of the non-dominated solutions
    bool ent = _parser.getORcreateParam(false, "entropy", "Store the entropy of the archive at each gen.", '\0', "Output").value();
    if (ent)
    {
        if (! dirOK )
            dirOK = testDirRes(dirName, eraseParam.value()); // TRUE
#ifdef _MSVC
        std::string stmp = dirName + "\entropy";
#else
        std::string stmp = dirName + "/entropy";
#endif
        moeoEntropyMetric < ObjectiveVector > * entropy = new moeoEntropyMetric < ObjectiveVector >;
        moeoBinaryMetricSavingUpdater < MOEOT > * entropy_updater = new moeoBinaryMetricSavingUpdater < MOEOT > (*entropy, _archive, stmp);
        _state.storeFunctor(entropy_updater);
        checkpoint.add(*entropy_updater);
    }

    // and that's it for the (control and) output
    return checkpoint;
}

#endif /*MAKE_CHECKPOINT_MOEO_H_*/
