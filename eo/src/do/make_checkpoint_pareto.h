// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_checkpoint_pareto.h
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

#ifndef _make_checkpoint_pareto_h
#define _make_checkpoint_pareto_h

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#ifdef HAVE_SSTREAM
#include <sstream>
#else
#include <strstream>
#endif

#include "EO.h"
#include "eoParetoFitness.h"
#include "eoEvalFuncCounter.h"
#include "utils/checkpointing"
#include "utils/selectors.h"

// at the moment, in utils/make_help.cpp
// this should become some eoUtils.cpp with corresponding eoUtils.h
bool testDirRes(std::string _dirName, bool _erase);
/////////////////// The checkpoint and other I/O //////////////

/** Of course, Fitness needs to be an eoParetoFitness!!!
 */
template <class EOT>
eoCheckPoint<EOT>& do_make_checkpoint_pareto(eoParser& _parser, eoState& _state, eoEvalFuncCounter<EOT>& _eval, eoContinue<EOT>& _continue)
{
  // first, create a checkpoint from the eoContinue - and store in _state
  eoCheckPoint<EOT> & checkpoint = 
        _state.storeFunctor(new eoCheckPoint<EOT>(_continue));

  /////// get number of obectives from Fitness - not very elegant
  typedef typename EOT::Fitness Fit;
  Fit fit;
  unsigned nObj = fit.size();

  ///////////////////
  // Counters
  //////////////////
  // is nb Eval to be used as counter?
  bool useEval = _parser.createParam(true, "useEval", "Use nb of eval. as counter (vs nb of gen.)", '\0', "Output").value();

    // Create anyway a generation-counter parameter WARNING: not stored anywhere!!!
    eoValueParam<unsigned> *generationCounter = new eoValueParam<unsigned>(0, "Gen.");
    // Create an incrementor (sub-class of eoUpdater).
    eoIncrementor<unsigned> & increment = 
      _state.storeFunctor(new eoIncrementor<unsigned>(generationCounter->value()) );
    // Add it to the checkpoint, 
    checkpoint.add(increment);

    // dir for DISK output
    std::string & dirName =  _parser.getORcreateParam(std::string("Res"), "resDir", "Directory to store DISK outputs", '\0', "Output - Disk").value();
    // shoudl we empty it if exists
    eoValueParam<bool>& eraseParam = _parser.createParam(true, "eraseDir", "erase files in dirName if any", '\0', "Output - Disk");
    bool dirOK = false;		   // not tested yet

    /////////////////////////////////////////
    // now some statistics on the population:
    /////////////////////////////////////////
    /**
     * existing stats for Pareto as of today, Jan. 31. 2002
     *
     * eoSortedPopStat   : whole population - type std::string (!!)
     */

  eoValueParam<eoParamParamType>& fPlotParam = _parser.createParam(
      eoParamParamType("1(0,1)"), "frontFileFrequency",
      "File save frequency in objective spaces (std::pairs of comma-separated objectives " \
      "in 1 single parentheses std::pair)",
      '\0', "Output - Disk");

#if !defined(NO_GNUPLOT)
  bool boolGnuplot = _parser.createParam(false, "plotFront",
                                         "Objective plots (requires corresponding files " \
                                         "- see frontFileFrequency",
                                         '\0', "Output - Graphical").value();
#endif

  eoParamParamType & fPlot = fPlotParam.value(); // std::pair<std::string,std::vector<std::string> >
  unsigned frequency = atoi(fPlot.first.c_str());
  if (frequency)		   // something to plot
    {
      unsigned nbPlot = fPlot.second.size(); 
      if ( nbPlot % 2 )		   // odd!
	throw std::runtime_error("Odd number of front description in make_checkpoint_pareto");

      // only create the necessary stats
      std::vector<bool> bStat(nObj, false); // track of who's already there
      std::vector<eoMOFitnessStat<EOT>* > theStats(nObj);

      // first create the stats
      for (unsigned i=0; i<nbPlot; i+=2)
	{
	  unsigned obj1 = atoi(fPlot.second[i].c_str());
	  unsigned obj2 = atoi(fPlot.second[i+1].c_str());
	  eoMOFitnessStat<EOT>* fStat;
	  if (!bStat[obj1]) {		   // not already there: create it
#ifdef HAVE_SSTREAM
	      std::ostringstream os;
	      os << "Obj. " << obj1 << std::ends; 
	      fStat = new eoMOFitnessStat<EOT>(obj1, os.str().c_str());
#else
	      char s[1024];
	      std::ostrstream os(s, 1022);
	      os << "Obj. " << obj1 << std::ends; 
	      fStat = new eoMOFitnessStat<EOT>(obj1, s);
#endif
	      _state.storeFunctor(fStat);
	      bStat[obj1]=true;
	      theStats[obj1]=fStat;
	      checkpoint.add(*fStat);
          }
	  if (!bStat[obj2]) {		   // not already there: create it
#ifdef HAVE_SSTREAM
	      std::ostringstream os;
	      os << "Obj. " << obj2 << std::ends; 
	      fStat = new eoMOFitnessStat<EOT>(obj2, os.str().c_str());
#else
	      char s[1024];
	      std::ostrstream os2(s, 1022);
	      os2 << "Obj. " << obj2 << std::ends; 
	      fStat = new eoMOFitnessStat<EOT>(obj2, s);
#endif
	      _state.storeFunctor(fStat);
	      bStat[obj2]=true;
	      theStats[obj2]=fStat;
	      checkpoint.add(*fStat);
          }

	  // then the fileSnapshots
#ifdef HAVE_SSTREAM
          std::ostringstream os;
	  os << "Front." << obj1 << "." << obj2 << "." << std::ends; 
	  eoFileSnapshot& snapshot = _state.storeFunctor(
              new eoFileSnapshot(dirName, frequency, os.str().c_str()));
#else
	  char s3[1024];
	  std::ostrstream os3(s3, 1022);
	  os3 << "Front." << obj1 << "." << obj2 << "." << std::ends; 
	  eoFileSnapshot & snapshot = _state.storeFunctor(
              new eoFileSnapshot(dirName, frequency, s3 ) );
#endif
	  checkpoint.add(snapshot);
      
	  snapshot.add(*theStats[obj1]);
	  snapshot.add(*theStats[obj2]);

	  // and create the gnuplotter from the fileSnapshot
#if !defined(NO_GNUPLOT)
	  if (boolGnuplot) 
	    {
	      eoGnuplot1DSnapshot & plotSnapshot = _state.storeFunctor(new
		    eoGnuplot1DSnapshot(snapshot));
	      plotSnapshot.pointSize =3;
	      checkpoint.add(plotSnapshot);
	    }
#endif
	}
    }
    // Dump of the whole population
    //-----------------------------
    bool printPop = _parser.createParam(false, "printPop", "Print sorted pop. every gen.", '\0', "Output").value();
    eoSortedPopStat<EOT> * popStat;
    if ( printPop ) // we do want pop dump
      {
	std::cout << "On cree printpop\n";
	popStat = & _state.storeFunctor(new eoSortedPopStat<EOT>);
	// add it to the checkpoint
	checkpoint.add(*popStat);
      }

    ///////////////
    // The monitors
    ///////////////
    // do we want an eoStdoutMonitor?
    bool needStdoutMonitor = printPop ;	// only this one at the moment

    // The Stdout monitor will print parameters to the screen ...  
    if ( needStdoutMonitor ) 
      {
	eoStdoutMonitor & monitor = _state.storeFunctor(new eoStdoutMonitor(false));

	// when called by the checkpoint (i.e. at every generation)
	checkpoint.add(monitor);

	// the monitor will output a series of parameters: add them
	monitor.add(*generationCounter);
	if (useEval) // we want nb of evaluations
	  monitor.add(_eval);
	if ( printPop)
	  monitor.add(*popStat);
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
    eoValueParam<unsigned>& saveTimeIntervalParam = _parser.createParam(unsigned(0), "saveTimeInterval", "Save every T seconds (0 or absent = never)", '\0',"Persistence" );
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

    // and that's it for the (control and) output
    return checkpoint;
}

#endif
