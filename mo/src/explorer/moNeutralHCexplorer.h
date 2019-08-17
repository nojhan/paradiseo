/*
  <moNeutralHCexplorer.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

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

#ifndef _moNeutralHCexplorer_h
#define _moNeutralHCexplorer_h

#include "moRandomBestHCexplorer.h"
#include "../comparator/moNeighborComparator.h"
#include "../comparator/moSolNeighborComparator.h"
#include "../neighborhood/moNeighborhood.h"

/**
 * Explorer for a neutral Hill-climbing
 */
template< class Neighbor >
class moNeutralHCexplorer : public moRandomBestHCexplorer<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;
    typedef moNeighborhood<Neighbor> Neighborhood ;

    using moNeighborhoodExplorer<Neighbor>::neighborhood;
    using moRandomBestHCexplorer<Neighbor>::solNeighborComparator;
    using moRandomBestHCexplorer<Neighbor>::isAccept;
    using moRandomBestHCexplorer<Neighbor>::bestVector;
    using moRandomBestHCexplorer<Neighbor>::initParam;
    using moRandomBestHCexplorer<Neighbor>::updateParam;

    /**
     * Constructor
     * @param _neighborhood the neighborhood
     * @param _eval the evaluation function
     * @param _neighborComparator a neighbor comparator
     * @param _solNeighborComparator a solution vs neighbor comparator
     * @param _nbStep maximum step to do
     */
    moNeutralHCexplorer(Neighborhood& _neighborhood,
                        moEval<Neighbor>& _eval,
                        moNeighborComparator<Neighbor>& _neighborComparator,
                        moSolNeighborComparator<Neighbor>& _solNeighborComparator,
                        unsigned _nbStep) :
            moRandomBestHCexplorer<Neighbor>(_neighborhood, _eval, _neighborComparator, _solNeighborComparator),nbStep(_nbStep) {
        //Some cycle is possible with equals fitness solutions if the neighborhood is not random
    }

    /**
     * Destructor
     */
    ~moNeutralHCexplorer() {
    }

    /**
     *  initial number of step
     * @param _solution the current solution
     */
    virtual void initParam(EOT & _solution) {
        moRandomBestHCexplorer<Neighbor>::initParam(_solution);

        step = 0;
    };

    /**
     * one more step
     * @param _solution the current solution
     */
    virtual void updateParam(EOT & _solution) {
        moRandomBestHCexplorer<Neighbor>::updateParam(_solution);

        step++;
    };

    /**
     * continue if there is a neighbor and it is remains some steps to do
     * @param _solution the solution
     * @return true there is some steps to do
     */
    virtual bool isContinue(EOT & _solution) {
        return (step < nbStep)  && isAccept ;
    };

    /**
     * accept test if an ameliorated or an equal neighbor was be found
     * @param _solution the solution
     * @return true if the best neighbor ameliorate the fitness or is equals
     */
    virtual bool accept(EOT & _solution) {
        if (neighborhood.hasNeighbor(_solution))
            isAccept = solNeighborComparator(_solution, bestVector[0]) || solNeighborComparator.equals(_solution, bestVector[0]) ;
        return isAccept;
    };

private:
    // current number of step
    unsigned int step;

    // maximum number of steps to do
    unsigned int nbStep;

};


#endif
