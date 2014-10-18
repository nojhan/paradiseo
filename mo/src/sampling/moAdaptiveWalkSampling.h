/*
  <moAdaptiveWalkSampling.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sebastien Verel, Arnaud Liefooghe, Jeremie Humeau

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.

  ParadisEO WebSite : http://paradiseo.gforge.inria.fr
  Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef moAdaptiveWalkSampling_h
#define moAdaptiveWalkSampling_h

#include <eoInit.h>
#include <eval/moEval.h>
#include <eoEvalFunc.h>
#include <continuator/moCheckpoint.h>
#include <perturb/moLocalSearchInit.h>
#include <algo/moRandomSearch.h>
#include <algo/moFirstImprHC.h>
#include <continuator/moSolutionStat.h>
#include <continuator/moMinusOneCounterStat.h>
#include <continuator/moStatFromStat.h>
#include <sampling/moSampling.h>
#include <eval/moEvalCounter.h>
#include <eoEvalFuncCounter.h>
#include <continuator/moValueStat.h>

/**
 * To compute the length and final solution of an adaptive walk:
 *   Perform a first improvement Hill-climber based on the neighborhood (adaptive walk),
 *   The adaptive walk is repeated several times (defined by a parameter)
 *
 *   Statistics are:
 *    - the length of the adaptive walk
 *    - the number of neighbor evaluaitons
 *    - the final solution which are local optimum
 *
 */
template <class Neighbor>
class moAdaptiveWalkSampling : public moSampling<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;

    using moSampling<Neighbor>::localSearch;

    /**
     * Constructor
     *
     * @param _init initialisation method of the solution
     * @param _neighborhood neighborhood giving neighbor in random order
     * @param _fullEval a full evaluation function
     * @param _eval an incremental evaluation of neighbors
     * @param _nbAdaptWalk Number of adaptive walks (minimum value = 2)
     */
    moAdaptiveWalkSampling(eoInit<EOT> & _init,
                          moNeighborhood<Neighbor> & _neighborhood,
                          eoEvalFunc<EOT>& _fullEval,
                          moEval<Neighbor>& _eval,
                          unsigned int _nbAdaptWalk) :
            moSampling<Neighbor>(initHC, * new moRandomSearch<Neighbor>(initHC, _fullEval, _nbAdaptWalk), copyStat),
	    neighborEvalCount(_eval),
	    nEvalStat(neighborEvalCount, true),
            copyStat(lengthStat),  // copy is used to report the statistic of the first walk
            copyEvalStat(nEvalStat),
            checkpoint(trueCont),
            hc(_neighborhood, _fullEval, neighborEvalCount, checkpoint),
            initHC(_init, hc)
    {
        // to count the number of step in the HC
        checkpoint.add(lengthStat);

	// to count the number of evaluations
        checkpoint.add(nEvalStat);
	
        // add the solution into statistics
        this->add(copyEvalStat);
        this->add(solStat);
    }

    /**
     * Destructor
     */
    ~moAdaptiveWalkSampling() {
        // delete the pointer on the local search which has been constructed in the constructor
        delete localSearch;
    }

protected:
  /* count the number of evaluations */
  moEvalCounter<Neighbor> neighborEvalCount;
  moValueStat<EOT, unsigned long> nEvalStat;
  moStatFromStat<EOT, double> copyEvalStat;

    moSolutionStat<EOT> solStat;
    moMinusOneCounterStat<EOT> lengthStat;
    moTrueContinuator<Neighbor> trueCont;
    moStatFromStat<EOT, unsigned int> copyStat;
    moCheckpoint<Neighbor> checkpoint;
    moFirstImprHC<Neighbor> hc;
    moLocalSearchInit<Neighbor> initHC;

};


#endif
