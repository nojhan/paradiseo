/*
* <make_checkpoint_moeo.h>
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

#ifndef MAKE_CHECKPOINT_MOEO_H_
#define MAKE_CHECKPOINT_MOEO_H_

#include <limits.h>
#include <stdlib.h>
#include <sstream>
#include <eoContinue.h>
#include <eoEvalFuncCounter.h>
#include <utils/checkpointing>
#include <utils/selectors.h>
#include <utils/eoParser.h>
#include <utils/eoState.h>
#include <metric/moeoContributionMetric.h>
#include <metric/moeoEntropyMetric.h>
#include <utils/moeoArchiveUpdater.h>
#include <utils/moeoArchiveObjectiveVectorSavingUpdater.h>
#include <utils/moeoBinaryMetricSavingUpdater.h>


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
eoCheckPoint < MOEOT > & do_make_checkpoint_moeo (eoParser & _parser, eoState & _state, eoEvalFuncCounter < MOEOT > & /*_eval*/, eoContinue < MOEOT > & _continue, eoPop < MOEOT > & _pop, moeoArchive < MOEOT > & _archive)
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
  eoValueParam<unsigned int> *generationCounter = new eoValueParam<unsigned int>(0, "Gen.");
  // Create an incrementor (sub-class of eoUpdater).
  eoIncrementor<unsigned int> & increment = _state.storeFunctor( new eoIncrementor<unsigned int>(generationCounter->value()) );
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
  eoValueParam<unsigned int>& saveFrequencyParam = _parser.createParam((unsigned int)(0), "saveFrequency", "Save every F generation (0 = only final state, absent = never)", '\0', "Persistence" );
  if (_parser.isItThere(saveFrequencyParam))
    {
      // first make sure dirName is OK
      if (! dirOK )
        dirOK = testDirRes(dirName, eraseParam.value()); // TRUE
      unsigned int freq = (saveFrequencyParam.value()>0 ? saveFrequencyParam.value() : UINT_MAX );
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
  eoValueParam<unsigned int>& saveTimeIntervalParam = _parser.getORcreateParam((unsigned int)(0), "saveTimeInterval", "Save every T seconds (0 or absent = never)", '\0',"Persistence" );
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
