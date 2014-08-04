/*
  <moHillClimberSampling.h>
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

#ifndef moHillClimberSampling_h
#define moHillClimberSampling_h

#include "../../eo/eoInit.h"
#include "../eval/moEval.h"
#include "../../eo/eoEvalFunc.h"
#include "../continuator/moCheckpoint.h"
#include "../perturb/moLocalSearchInit.h"
#include "../algo/moRandomSearch.h"
#include "../algo/moSimpleHC.h"
#include "../continuator/moSolutionStat.h"
#include "../continuator/moMinusOneCounterStat.h"
#include "../continuator/moStatFromStat.h"
#include "moSampling.h"

/**
 * To compute the length and final solution of an adaptive walk:
 *   Perform a simple Hill-climber based on the neighborhood (gradiant walk, the whole neighborhood is visited),
 *   The lengths of HC are collected and the final solution which are local optima
 *   The adaptive walk is repeated several times
 *
 */
template <class Neighbor>
class moHillClimberSampling : public moSampling<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;

    using moSampling<Neighbor>::localSearch;

    /**
     * Constructor
     * @param _init initialisation method of the solution
     * @param _neighborhood neighborhood giving neighbor in random order
     * @param _fullEval a full evaluation function
     * @param _eval an incremental evaluation of neighbors
     * @param _nbAdaptWalk Number of adaptive walks
     */
    moHillClimberSampling(eoInit<EOT> & _init,
                          moNeighborhood<Neighbor> & _neighborhood,
                          eoEvalFunc<EOT>& _fullEval,
                          moEval<Neighbor>& _eval,
                          unsigned int _nbAdaptWalk) :
            moSampling<Neighbor>(initHC, * new moRandomSearch<Neighbor>(initHC, _fullEval, _nbAdaptWalk), copyStat),
            copyStat(lengthStat),
            checkpoint(trueCont),
            hc(_neighborhood, _fullEval, _eval, checkpoint),
            initHC(_init, hc)
    {
        // to count the number of step in the HC
        checkpoint.add(lengthStat);

        // add the solution into statistics
        this->add(solStat);
    }

    /**
     * Destructor
     */
    ~moHillClimberSampling() {
        // delete the pointer on the local search which has been constructed in the constructor
        delete localSearch;
    }

protected:
    moSolutionStat<EOT> solStat;
    moMinusOneCounterStat<EOT> lengthStat;
    moTrueContinuator<Neighbor> trueCont;
    moStatFromStat<EOT, unsigned int> copyStat;
    moCheckpoint<Neighbor> checkpoint;
    moSimpleHC<Neighbor> hc;
    moLocalSearchInit<Neighbor> initHC;
};


#endif
