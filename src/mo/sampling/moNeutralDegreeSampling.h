/*
  <moNeutralDegreeSampling.h>
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

#ifndef moNeutralDegreeSampling_h
#define moNeutralDegreeSampling_h

#include "../../eo/eoInit.h"
#include "../neighborhood/moNeighborhood.h"
#include "../eval/moEval.h"
#include "../../eo/eoEvalFunc.h"
#include "../algo/moRandomSearch.h"
#include "../continuator/moFitnessStat.h"
#include "../continuator/moNeighborhoodStat.h"
#include "../continuator/moNeutralDegreeNeighborStat.h"
#include "moSampling.h"

/**
 * To compute the neutral degree:
 *   Sample the fitness of random solution in the search space (1er information)
 *   and sample the neutral degree (2nd information), i.e. the number of neighbor solutions with the same fitness value
 *   The values are collected during the random search
 *
 */
template <class Neighbor>
class moNeutralDegreeSampling : public moSampling<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;

    using moSampling<Neighbor>::localSearch;

    /**
     * Constructor
     * @param _init initialisation method of the solution
     * @param _neighborhood neighborhood to compute the neutral degree
     * @param _fullEval Fitness function, full evaluation function
     * @param _eval neighbor evaluation, incremental evaluation function
     * @param _nbSol Number of solutions in the sample
     */
    moNeutralDegreeSampling(eoInit<EOT> & _init,
                            moNeighborhood<Neighbor> & _neighborhood,
                            eoEvalFunc<EOT>& _fullEval,
                            moEval<Neighbor>& _eval,
                            unsigned int _nbSol) :
            moSampling<Neighbor>(_init, * new moRandomSearch<Neighbor>(_init, _fullEval, _nbSol), fitnessStat),
            neighborhoodStat(_neighborhood, _eval),
            ndStat(neighborhoodStat)
    {
        this->add(neighborhoodStat, false);
        this->add(ndStat);
    }

    /**
     * Constructor with comparators
     * @param _init initialisation method of the solution
     * @param _neighborhood neighborhood to compute the neutral degree
     * @param _fullEval Fitness function, full evaluation function
     * @param _eval neighbor evaluation, incremental evaluation function
     * @param _neighborComparator a neighbor Comparator
     * @param _solNeighborComparator a comparator between a solution and a neighbor
     * @param _nbSol Number of solutions in the sample
     */
    moNeutralDegreeSampling(eoInit<EOT> & _init,
                            moNeighborhood<Neighbor> & _neighborhood,
                            eoEvalFunc<EOT>& _fullEval,
                            moEval<Neighbor>& _eval,
                            moNeighborComparator<Neighbor>& _neighborComparator,
                            moSolNeighborComparator<Neighbor>& _solNeighborComparator,
                            unsigned int _nbSol) :
            moSampling<Neighbor>(_init, * new moRandomSearch<Neighbor>(_init, _fullEval, _nbSol), fitnessStat),
            neighborhoodStat(_neighborhood, _eval, _neighborComparator, _solNeighborComparator),
            ndStat(neighborhoodStat)
    {
        this->add(neighborhoodStat, false);
        this->add(ndStat);
    }

    /**
     * default destructor
     */
    ~moNeutralDegreeSampling() {
        // delete the pointer on the local search which has been constructed in the constructor
        delete localSearch;
    }

protected:
    moFitnessStat<EOT> fitnessStat;
    moNeighborhoodStat< Neighbor > neighborhoodStat;
    moNeutralDegreeNeighborStat< Neighbor > ndStat;

};


#endif
