/*
  <moMetropolisHastingExplorer.h>
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

#ifndef _moMetropolisHastingExplorer_h
#define _moMetropolisHastingExplorer_h

#include <cstdlib>

#include <explorer/moNeighborhoodExplorer.h>
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
#include <neighborhood/moNeighborhood.h>

#include <utils/eoRNG.h>

/**
 * Explorer for the Metropolis-Hasting Sampling.
 * Only the symetric case is considered when Q(x,y) = Q(y,x)
 * Fitness must be > 0
 */
template< class Neighbor >
class moMetropolisHastingExplorer : public moNeighborhoodExplorer<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;
    typedef moNeighborhood<Neighbor> Neighborhood ;

    using moNeighborhoodExplorer<Neighbor>::neighborhood;
    using moNeighborhoodExplorer<Neighbor>::eval;
    using moNeighborhoodExplorer<Neighbor>::selectedNeighbor;

    /**
     * Constructor
     * @param _neighborhood the neighborhood
     * @param _eval the evaluation function
     * @param _neighborComparator a neighbor comparator
     * @param _solNeighborComparator a solution vs neighbor comparator
     * @param _nbStep maximum number of step to do
     */
    moMetropolisHastingExplorer(Neighborhood& _neighborhood, moEval<Neighbor>& _eval, moNeighborComparator<Neighbor>& _neighborComparator, moSolNeighborComparator<Neighbor>& _solNeighborComparator, unsigned int _nbStep) : moNeighborhoodExplorer<Neighbor>(_neighborhood, _eval), neighborComparator(_neighborComparator), solNeighborComparator(_solNeighborComparator), nbStep(_nbStep) {
        isAccept = false;
        if (!neighborhood.isRandom()) {
            std::clog << "moMetropolisHastingExplorer::Warning -> the neighborhood used is not random" << std::endl;
        }
    }

    /**
     * Destructor
     */
    ~moMetropolisHastingExplorer() {
    }

    /**
     * initialization of the number of step to be done
     * @param _solution unused solution
     */
    virtual void initParam(EOT & /*_solution*/) {
        step     = 0;
        isAccept = true;
    };

    /**
     * increase the number of step
     * @param _solution unused solution
     */
    virtual void updateParam(EOT & /*_solution*/) {
        step++;
    };

    /**
     * terminate: NOTHING TO DO
     * @param _solution unused solution
     */
    virtual void terminate(EOT & /*_solution*/) {};

    /**
     * Explore the neighborhood of a solution
     * @param _solution
     */
    virtual void operator()(EOT & _solution) {
        //Test if _solution has a Neighbor
        if (neighborhood.hasNeighbor(_solution)) {
            //init the first neighbor
            neighborhood.init(_solution, selectedNeighbor);

            //eval the _solution moved with the neighbor and stock the result in the neighbor
            eval(_solution, selectedNeighbor);
        }
        else {
            //if _solution hasn't neighbor,
            isAccept=false;
        }
    };

    /**
     * continue if there is a neighbor and it is remainds some steps to do
     * @param _solution the solution
     * @return true there is some steps to do
     */
    virtual bool isContinue(EOT & /*_solution*/) {
        return (step < nbStep) ;
    };

    /**
     * accept test if an ameliorated neighbor was found
     * @param _solution the solution
     * @return true if the best neighbor ameliorate the fitness
     */
    virtual bool accept(EOT & _solution) {
        double alpha=0.0;
        if (neighborhood.hasNeighbor(_solution)) {
	  if (solNeighborComparator(_solution, selectedNeighbor))
                isAccept = true;
            else {
                if (_solution.fitness() != 0) {
                    if ( (double)selectedNeighbor.fitness() < (double)_solution.fitness()) // maximizing
                        alpha = (double) selectedNeighbor.fitness() / (double) _solution.fitness();
                    else //minimizing
                        alpha = (double) _solution.fitness() / (double) selectedNeighbor.fitness();
                    isAccept = (rng.uniform() < alpha) ;
                }
                else {
                    if ( (double) selectedNeighbor.fitness() < (double) _solution.fitness()) // maximizing
                        isAccept = true;
                    else
                        isAccept = false;
                }
            }
        }
        return isAccept;
    };

private:
    // comparator between solution and neighbor or between neighbors
    moNeighborComparator<Neighbor>& neighborComparator;
    moSolNeighborComparator<Neighbor>& solNeighborComparator;

    // current number of step
    unsigned int step;

    // maximum number of steps to do
    unsigned int nbStep;

    // true if the move is accepted
    bool isAccept ;
};


#endif
