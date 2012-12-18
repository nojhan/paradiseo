 /* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*- */

//-----------------------------------------------------------------------------
// make_checkpoint_assembled.h
// Marc Wintermantel & Oliver Koenig
// IMES-ST@ETHZ.CH
// March 2003

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
             mak@dhi.dk
*/
//-----------------------------------------------------------------------------

#ifndef _make_checkpoint_assembled_h
#define _make_checkpoint_assembled_h

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <climits>
#include <vector>
#include <string>

#include <eoScalarFitnessAssembled.h>
#include <utils/selectors.h>
#include <EO.h>
#include <eoEvalFuncCounter.h>
#include <utils/checkpointing>

// at the moment, in utils/make_help.cpp
// this should become some eoUtils.cpp with corresponding eoUtils.h
bool testDirRes(std::string _dirName, bool _erase);
/////////////////// The checkpoint and other I/O //////////////

/** Of course, Fitness needs to be an eoScalarFitnessAssembled!!!
 *
 *
 * @ingroup Builders
 * */
template <class EOT>
eoCheckPoint<EOT>& do_make_checkpoint_assembled(eoParser& _parser, eoState& _state, eoEvalFuncCounter<EOT>& _eval, eoContinue<EOT>& _continue)
{

    // SOME PARSER PARAMETERS
    // ----------------------
    std::string dirName = _parser.getORcreateParam(std::string("Res"), "resDir",
                                                   "Directory to store DISK outputs",
                                                   '\0', "Output").value();
    bool erase = _parser.getORcreateParam(true, "eraseDir",
                                          "Erase files in dirName if any",
                                          '\0', "Output").value();
    bool gnuplots = _parser.getORcreateParam(true, "plots",
                                             "Plot stuff using GnuPlot",
                                             '\0', "Output").value();
    bool printFile = _parser.getORcreateParam(true, "printFile",
                                              "Print statistics file",
                                              '\0', "Output").value();

  eoValueParam<unsigned>& saveFrequencyParam
      = _parser.getORcreateParam(unsigned(0), "saveFrequency",
                                 "Save every F generation (0 = only final state, absent = never)",
                                 '\0', "Persistence" );

  testDirRes(dirName, erase); // TRUE

  // CREATE CHECKPOINT FROM eoContinue
  // ---------------------------------
  eoCheckPoint<EOT> *checkpoint = new eoCheckPoint<EOT>(_continue);
  _state.storeFunctor(checkpoint);

  // GENERATIONS
  // -----------
  eoIncrementorParam<unsigned> *generationCounter = new eoIncrementorParam<unsigned>("Gen.");
  _state.storeFunctor(generationCounter);
  checkpoint->add(*generationCounter);

  // TIME
  // ----
  eoTimeCounter * tCounter = NULL;
  tCounter = new eoTimeCounter;
  _state.storeFunctor(tCounter);
  checkpoint->add(*tCounter);

  // ACCESS DESCRIPTIONS OF TERMS OF FITNESS CLASS
  // ---------------------------------------------
  // define a temporary fitness instance
  typedef typename EOT::Fitness Fit;
  Fit fit;
  std::vector<std::string> fitness_descriptions = fit.getDescriptionVector();
  unsigned nTerms = fitness_descriptions.size();

  // STAT VALUES OF A POPULATION
  // ---------------------------

  // average vals
  std::vector<eoAssembledFitnessAverageStat<EOT>* > avgvals( nTerms );
  for (unsigned i=0; i < nTerms; ++i){
    std::string descr = "Avg. of " + fitness_descriptions[i];
    avgvals[i] = new eoAssembledFitnessAverageStat<EOT>(i, descr);
    _state.storeFunctor( avgvals[i] );
    checkpoint->add( *avgvals[i] );
  }

  // best vals
  std::vector<eoAssembledFitnessBestStat<EOT>* > bestvals( nTerms );
  for (unsigned j=0; j < nTerms; ++j){
    std::string descr = fitness_descriptions[j] + " of best ind.";
    bestvals[j] = new eoAssembledFitnessBestStat<EOT>(j, descr);
    _state.storeFunctor( bestvals[j] );
    checkpoint->add( *bestvals[j] );
  }

  // STDOUT
  // ------
  eoStdoutMonitor *monitor = new eoStdoutMonitor(/*false FIXME remove this deprecated prototype*/);
  _state.storeFunctor(monitor);
  checkpoint->add(*monitor);
  monitor->add(*generationCounter);
  monitor->add(_eval);
  monitor->add(*tCounter);

  // Add best fitness
  monitor->add( *bestvals[0] );

  // Add all average vals
  for (unsigned l=0; l < nTerms; ++l)
    monitor->add( *avgvals[l] );

  // GNUPLOT
  // -------
  if (gnuplots ){
    std::string stmp;

    // Histogramm of the different fitness vals
    eoScalarFitnessStat<EOT> *fitStat = new eoScalarFitnessStat<EOT>;
    _state.storeFunctor(fitStat);
    checkpoint->add(*fitStat);
#ifdef HAVE_GNUPLOT
        // a gnuplot-based monitor for snapshots: needs a dir name
     eoGnuplot1DSnapshot *fitSnapshot = new eoGnuplot1DSnapshot(dirName);
     _state.storeFunctor(fitSnapshot);
    // add any stat that is a vector<double> to it
    fitSnapshot->add(*fitStat);
    // and of course add it to the checkpoint
    checkpoint->add(*fitSnapshot);

    std::vector<eoGnuplot1DMonitor*> gnumonitors(nTerms, NULL );
    for (unsigned k=0; k < nTerms; ++k){
      stmp = dirName + "/gnuplot_" + fitness_descriptions[k] + ".xg";
      gnumonitors[k] = new eoGnuplot1DMonitor(stmp,true);
      _state.storeFunctor(gnumonitors[k]);
      checkpoint->add(*gnumonitors[k]);
      gnumonitors[k]->add(*generationCounter);
      gnumonitors[k]->add(*bestvals[k]);
      gnumonitors[k]->add(*avgvals[k]);
    }
#endif
  }

  // WRITE STUFF TO FILE
  // -------------------
  if( printFile ){
    std::string stmp2 = dirName + "/eoStatistics.sav";
    eoFileMonitor *fileMonitor = new eoFileMonitor(stmp2);
    _state.storeFunctor(fileMonitor);
    checkpoint->add(*fileMonitor);
    fileMonitor->add(*generationCounter);
    fileMonitor->add(_eval);
    fileMonitor->add(*tCounter);

    for (unsigned i=0; i < nTerms; ++i){
      fileMonitor->add(*bestvals[i]);
      fileMonitor->add(*avgvals[i]);
    }

  }

  // STATE SAVER
  // -----------
  // feed the state to state savers

  if (_parser.isItThere(saveFrequencyParam)) {

    unsigned freq = (saveFrequencyParam.value() > 0 ? saveFrequencyParam.value() : UINT_MAX );
    std::string stmp = dirName + "/generations";
    eoCountedStateSaver *stateSaver1 = new eoCountedStateSaver(freq, _state, stmp);
    _state.storeFunctor(stateSaver1);
    checkpoint->add(*stateSaver1);
  }

  // and that's it for the (control and) output
  return *checkpoint;
}

#endif
