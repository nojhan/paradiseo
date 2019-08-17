/*
  <moTS.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  ue,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

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

#ifndef _moTS_h
#define _moTS_h

#include "moLocalSearch.h"
#include "../explorer/moTSexplorer.h"
#include "../memory/moNeighborVectorTabuList.h"
#include "../memory/moIntensification.h"
#include "../memory/moDummyIntensification.h"
#include "../memory/moDiversification.h"
#include "../memory/moDummyDiversification.h"
#include "../memory/moAspiration.h"
#include "../memory/moBestImprAspiration.h"
#include "../continuator/moTimeContinuator.h"
#include "../eval/moEval.h"
#include <paradiseo/eo/eoEvalFunc.h>

/**
 * Tabu Search
 */
template<class Neighbor>
class moTS: public moLocalSearch<Neighbor>
{
public:

  typedef typename Neighbor::EOT EOT;
  typedef moNeighborhood<Neighbor> Neighborhood ;

  /**
   * Basic constructor for a tabu search
   * @param _neighborhood the neighborhood
   * @param _fullEval the full evaluation function
   * @param _eval neighbor's evaluation function
   * @param _time the time limit for stopping criteria
   * @param _tabuListSize the size of the tabu list
   */
  moTS(Neighborhood& _neighborhood,
       eoEvalFunc<EOT>& _fullEval,
       moEval<Neighbor>& _eval,
       unsigned int _time,
       unsigned int _tabuListSize
       ):
    moLocalSearch<Neighbor>(explorer, timeCont, _fullEval),
    timeCont(_time),
    tabuList(_tabuListSize,0),
    explorer(_neighborhood, _eval, defaultNeighborComp, defaultSolNeighborComp, tabuList, dummyIntensification, dummyDiversification, defaultAspiration)
  {}

  /**
   * Simple constructor for a tabu search
   * @param _neighborhood the neighborhood
   * @param _fullEval the full evaluation function
   * @param _eval neighbor's evaluation function
   * @param _time the time limit for stopping criteria
   * @param _tabuList the tabu list
   */
  moTS(Neighborhood& _neighborhood,
       eoEvalFunc<EOT>& _fullEval,
       moEval<Neighbor>& _eval,
       unsigned int _time,
       moTabuList<Neighbor>& _tabuList):
    moLocalSearch<Neighbor>(explorer, timeCont, _fullEval),
    timeCont(_time),
    tabuList(0,0),
    explorer(_neighborhood, _eval, defaultNeighborComp, defaultSolNeighborComp, _tabuList, dummyIntensification, dummyDiversification, defaultAspiration)
  {}

  /**
   * General constructor for a tabu search
   * @param _neighborhood the neighborhood
   * @param _fullEval the full evaluation function
   * @param _eval neighbor's evaluation function
   * @param _cont an external continuator
   * @param _tabuList the tabu list
   * @param _aspiration the aspiration Criteria
   */
  moTS(Neighborhood& _neighborhood,
       eoEvalFunc<EOT>& _fullEval,
       moEval<Neighbor>& _eval,
       moContinuator<Neighbor>& _cont,
       moTabuList<Neighbor>& _tabuList,
       moAspiration<Neighbor>& _aspiration):
    moLocalSearch<Neighbor>(explorer, _cont, _fullEval),
    timeCont(0),
    tabuList(0,0),
    explorer(_neighborhood, _eval, defaultNeighborComp, defaultSolNeighborComp, _tabuList, dummyIntensification, dummyDiversification, _aspiration)
  {}

  /**
   * General constructor for a tabu search
   * @param _neighborhood the neighborhood
   * @param _fullEval the full evaluation function
   * @param _eval neighbor's evaluation function
   * @param _neighborComp a comparator between 2 neighbors
   * @param _solNeighborComp a solution vs neighbor comparator
   * @param _cont an external continuator
   * @param _tabuList the tabu list
   * @param _intensification the intensification strategy
   * @param _diversification the diversification strategy
   * @param _aspiration the aspiration Criteria
   */
  moTS(Neighborhood& _neighborhood,
       eoEvalFunc<EOT>& _fullEval,
       moEval<Neighbor>& _eval,
       moNeighborComparator<Neighbor>& _neighborComp,
       moSolNeighborComparator<Neighbor>& _solNeighborComp,
       moContinuator<Neighbor>& _cont,
       moTabuList<Neighbor>& _tabuList,
       moIntensification<Neighbor>& _intensification,
       moDiversification<Neighbor>& _diversification,
       moAspiration<Neighbor>& _aspiration):
    moLocalSearch<Neighbor>(explorer, _cont, _fullEval),
    timeCont(0),
    tabuList(0,0),
    explorer(_neighborhood, _eval, _neighborComp, _solNeighborComp, _tabuList, _intensification, _diversification, _aspiration)
  {}

  /*
   * To get the explorer and then to be abble to get the best solution so far
   * @return the TS explorer
   */
  moTSexplorer<Neighbor>& getExplorer() {
    return explorer;
  }
  
private:
  moTimeContinuator<Neighbor> timeCont;
  moNeighborComparator<Neighbor> defaultNeighborComp;
  moSolNeighborComparator<Neighbor> defaultSolNeighborComp;
  moNeighborVectorTabuList<Neighbor> tabuList;
  moDummyIntensification<Neighbor> dummyIntensification;
  moDummyDiversification<Neighbor> dummyDiversification;
  moBestImprAspiration<Neighbor> defaultAspiration;
  moTSexplorer<Neighbor> explorer;
};
#endif
