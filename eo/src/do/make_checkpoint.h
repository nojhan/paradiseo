// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_checkpoint.h
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

#ifndef _make_checkpoint_h
#define _make_checkpoint_h

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <climits>

#include <eoScalarFitness.h>
#include <utils/selectors.h> // for minimizing_fitness()
#include <EO.h>
#include <eoEvalFuncCounter.h>
#include <utils/checkpointing>

// at the moment, in utils/make_help.cpp
// this should become some eoUtils.cpp with corresponding eoUtils.h
bool testDirRes(std::string _dirName, bool _erase);
/////////////////// The checkpoint and other I/O //////////////

/**
 *
 * CHANGE (March 2008): now receiving an eoValueParam instead of an eoEvalFuncCounter. This function is just interested
 * in the value of the parameter calculated on the evaluation function, not in the actual function itself!!
 *
 * @ingroup Builders
 */
template <class EOT>
eoCheckPoint<EOT>& do_make_checkpoint(eoParser& _parser, eoState& _state, eoValueParam<unsigned long>& _eval, eoContinue<EOT>& _continue)
{
    // first, create a checkpoint from the eoContinue
    eoCheckPoint<EOT> *checkpoint = new eoCheckPoint<EOT>(_continue);

    _state.storeFunctor(checkpoint);

    ////////////////////
    // Signal monitoring
    ////////////////////

#ifndef _MSC_VER
    // the CtrlC monitoring interception
    eoSignal<EOT> *mon_ctrlCCont;
    eoValueParam<bool>& mon_ctrlCParam = _parser.createParam(false, "monitor-with-CtrlC", "Monitor current generation upon Ctrl C",0, "Stopping criterion");
    if (mon_ctrlCParam.value())
      {
        mon_ctrlCCont = new eoSignal<EOT>;
        // store
        _state.storeFunctor(mon_ctrlCCont);
        // add to checkpoint
	checkpoint->add(*mon_ctrlCCont);
      }
#endif

    ///////////////////
    // Counters
    //////////////////

    // is nb Eval to be used as counter?
    eoValueParam<bool>& useEvalParam = _parser.createParam(true, "useEval", "Use nb of eval. as counter (vs nb of gen.)", '\0', "Output");
    eoValueParam<bool>& useTimeParam = _parser.createParam(true, "useTime", "Display time (s) every generation", '\0', "Output");
    // if we want the time, we need an eoTimeCounter
    eoTimeCounter * tCounter = NULL;

    // Create anyway a generation-counter
    // Recent change (03/2002): it is now an eoIncrementorParam, both
    // a parameter AND updater so you can store it into the eoState
    eoIncrementorParam<unsigned> *generationCounter = new eoIncrementorParam<unsigned>("Gen.");

    // store it in the state
    _state.storeFunctor(generationCounter);

    // And add it to the checkpoint,
    checkpoint->add(*generationCounter);

    // dir for DISK output
    eoValueParam<std::string>& dirNameParam =  _parser.createParam(std::string("Res"), "resDir", "Directory to store DISK outputs", '\0', "Output - Disk");

    // shoudl we empty it if exists
    eoValueParam<bool>& eraseParam = _parser.createParam(true, "eraseDir", "erase files in dirName if any", '\0', "Output - Disk");

    bool dirOK = false;            // not tested yet

    /////////////////////////////////////////
    // now some statistics on the population:
    /////////////////////////////////////////

    /**
     * existing stats as of today, April 10. 2001
     *
     * eoBestFitnessStat : best value in pop - type EOT::Fitness
     * eoAverageStat     : average value in pop - type EOT::Fitness
     * eoSecondMomentStat: average + stdev - type std::pair<double, double>
     * eoSortedPopStat   : whole population - type std::string (!!)
     * eoScalarFitnessStat: the fitnesses - type std::vector<double>
     */

    // Best fitness in population
    //---------------------------
    eoValueParam<bool>& printBestParam = _parser.createParam(true, "printBestStat", "Print Best/avg/stdev every gen.", '\0', "Output");
    eoValueParam<bool>& plotBestParam = _parser.createParam(false, "plotBestStat", "Plot Best/avg Stat", '\0', "Output - Graphical");
    eoValueParam<bool>& fileBestParam = _parser.createParam(false, "fileBestStat", "Output bes/avg/std to file", '\0', "Output - Disk");

    eoBestFitnessStat<EOT> *bestStat = NULL;
    if ( printBestParam.value() || plotBestParam.value() || fileBestParam.value() )
	// we need the bestStat for at least one of the 3 above
	{
	    bestStat = new eoBestFitnessStat<EOT>;
	    // store it
	    _state.storeFunctor(bestStat);
	    // add it to the checkpoint
	    checkpoint->add(*bestStat);
	    // check if monitoring with signal
	    if ( mon_ctrlCParam.value() )
		mon_ctrlCCont->add(*bestStat);
	}

    // Average fitness alone
    //----------------------
    eoAverageStat<EOT> *averageStat = NULL; // do we need averageStat?
    if ( printBestParam.value() || plotBestParam.value() || fileBestParam.value() ) // we need it for gnuplot output
	{
	    averageStat = new eoAverageStat<EOT>;
	    // store it
	    _state.storeFunctor(averageStat);
	    // add it to the checkpoint
	    checkpoint->add(*averageStat);
	    // check if monitoring with signal
	    if ( mon_ctrlCParam.value() )
		mon_ctrlCCont->add(*averageStat);
	}

    // Second moment stats: average and stdev
    //---------------------------------------
    eoSecondMomentStats<EOT> *secondStat = NULL;
    if ( printBestParam.value() || fileBestParam.value() ) // we need it for screen output or file output
	{
	    secondStat = new eoSecondMomentStats<EOT>;
	    // store it
	    _state.storeFunctor(secondStat);
	    // add it to the checkpoint
	    checkpoint->add(*secondStat);
	    // check if monitoring with signal
	    if ( mon_ctrlCParam.value() )
		mon_ctrlCCont->add(*secondStat);
	}

    // Dump of the whole population
    //-----------------------------
    eoSortedPopStat<EOT> *popStat = NULL;
    eoValueParam<bool>& printPopParam = _parser.createParam(false, "printPop", "Print sorted pop. every gen.", '\0', "Output");

    if ( printPopParam.value() ) // we do want pop dump
	{
	    popStat = new eoSortedPopStat<EOT>;
	    // store it
	    _state.storeFunctor(popStat);
	    // add it to the checkpoint
	    checkpoint->add(*popStat);
	    // check if monitoring with signal
	    if ( mon_ctrlCParam.value() )
		mon_ctrlCCont->add(*popStat);
	}

    // do we wnat some histogram of fitnesses snpashots?
    eoValueParam<bool> plotHistogramParam = _parser.createParam(false, "plotHisto", "Plot histogram of fitnesses", '\0', "Output - Graphical");

    ///////////////
    // The monitors
    ///////////////

    // do we want an eoStdoutMonitor?
    bool needStdoutMonitor = printBestParam.value()
        || printPopParam.value() ;

    // The Stdout monitor will print parameters to the screen ...
    if ( needStdoutMonitor )
	{
	    eoStdoutMonitor *monitor = new eoStdoutMonitor(/*false FIXME remove this deprecated prototype*/);
	    _state.storeFunctor(monitor);

	    // when called by the checkpoint (i.e. at every generation)
	    // check if monitoring with signal
	    if ( ! mon_ctrlCParam.value() )
		checkpoint->add(*monitor);
	    else
		mon_ctrlCCont->add(*monitor);

	    // the monitor will output a series of parameters: add them
	    monitor->add(*generationCounter);

	    if (useEvalParam.value()) // we want nb of evaluations
		monitor->add(_eval);
	    if (useTimeParam.value()) // we want time
		{
		    tCounter = new eoTimeCounter;
		    _state.storeFunctor(tCounter);
		    // check if monitoring with signal
		    if ( ! mon_ctrlCParam.value() )
			checkpoint->add(*tCounter);
		    else
			mon_ctrlCCont->add(*tCounter);
		    monitor->add(*tCounter);
		}

	    if (printBestParam.value())
		{
		    monitor->add(*bestStat);
		    monitor->add(*secondStat);
		}

	    if ( printPopParam.value())
		monitor->add(*popStat);
	}

    // first handle the dir test - if we need at least one file
    if ( ( fileBestParam.value() || plotBestParam.value() ||
           plotHistogramParam.value() )
         && !dirOK )               // just in case we add something before
	dirOK = testDirRes(dirNameParam.value(), eraseParam.value()); // TRUE

    if (fileBestParam.value())    // A file monitor for best & secondMoment
	{
#ifdef _MSVC
	    std::string stmp = dirNameParam.value() + "\best.xg";
#else
	    std::string stmp = dirNameParam.value() + "/best.xg";
#endif
	    eoFileMonitor *fileMonitor = new eoFileMonitor(stmp);
	    // save and give to checkpoint
	    _state.storeFunctor(fileMonitor);
	    checkpoint->add(*fileMonitor);
	    // and feed with some statistics
	    fileMonitor->add(*generationCounter);
	    fileMonitor->add(_eval);
	    if (tCounter)              // we want the time as well
		{
		    //      std::cout << "On met timecounter\n";
		    fileMonitor->add(*tCounter);
		}
	    fileMonitor->add(*bestStat);
	    fileMonitor->add(*secondStat);
	}

#if defined(HAVE_GNUPLOT)
    if (plotBestParam.value())    // an eoGnuplot1DMonitor for best & average
	{
	    std::string stmp = dirNameParam.value() + "/gnu_best.xg";
	    eoGnuplot1DMonitor *gnuMonitor = new eoGnuplot1DMonitor(stmp,minimizing_fitness<EOT>());
	    // save and give to checkpoint
	    _state.storeFunctor(gnuMonitor);
	    checkpoint->add(*gnuMonitor);
	    // and feed with some statistics
	    if (useEvalParam.value())  // do we want eval as X coordinate
		gnuMonitor->add(_eval);
	    else if (tCounter)         // or time?
		gnuMonitor->add(*tCounter);
	    else                       // default: generation
		gnuMonitor->add(*generationCounter);
	    gnuMonitor->add(*bestStat);
	    gnuMonitor->add(*averageStat);
	}

    // historgram?
    if (plotHistogramParam.value()) // want to see how the fitness is spread?
	{
	    eoScalarFitnessStat<EOT> *fitStat = new eoScalarFitnessStat<EOT>;
	    _state.storeFunctor(fitStat);
	    checkpoint->add(*fitStat);
	    // a gnuplot-based monitor for snapshots: needs a dir name
	    eoGnuplot1DSnapshot *fitSnapshot = new eoGnuplot1DSnapshot(dirNameParam.value());
	    _state.storeFunctor(fitSnapshot);
	    // add any stat that is a std::vector<double> to it
	    fitSnapshot->add(*fitStat);
	    // and of course add it to the checkpoint
	    checkpoint->add(*fitSnapshot);
	}

#endif

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
		dirOK = testDirRes(dirNameParam.value(), eraseParam.value()); // TRUE

	    unsigned freq = (saveFrequencyParam.value()>0 ? saveFrequencyParam.value() : UINT_MAX );
#ifdef _MSVC
	    std::string stmp = dirNameParam.value() + "\generations";
#else
	    std::string stmp = dirNameParam.value() + "/generations";
#endif
	    eoCountedStateSaver *stateSaver1 = new eoCountedStateSaver(freq, _state, stmp);
	    _state.storeFunctor(stateSaver1);
	    checkpoint->add(*stateSaver1);
	}

    // save state every T seconds
    eoValueParam<unsigned>& saveTimeIntervalParam = _parser.createParam(unsigned(0), "saveTimeInterval", "Save every T seconds (0 or absent = never)", '\0',"Persistence" );
    if (_parser.isItThere(saveTimeIntervalParam) && saveTimeIntervalParam.value()>0)
	{
	    // first make sure dirName is OK
	    if (! dirOK )
		dirOK = testDirRes(dirNameParam.value(), eraseParam.value()); // TRUE

#ifdef _MSVC
	    std::string stmp = dirNameParam.value() + "\time";
#else
	    std::string stmp = dirNameParam.value() + "/time";
#endif
	    eoTimedStateSaver *stateSaver2 = new eoTimedStateSaver(saveTimeIntervalParam.value(), _state, stmp);
	    _state.storeFunctor(stateSaver2);
	    checkpoint->add(*stateSaver2);
	}

    // and that's it for the (control and) output
    return *checkpoint;
}

#endif
