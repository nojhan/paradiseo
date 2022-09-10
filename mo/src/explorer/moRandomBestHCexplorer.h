/*
  <moRandomBestHCexplorer.h>
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

#ifndef _moRandomBestHCexplorer_h
#define _moRandomBestHCexplorer_h

#include <explorer/moNeighborhoodExplorer.h>
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
#include <neighborhood/moNeighborhood.h>
#include <vector>
#include <utils/eoRNG.h>

/**
 * Explorer for Hill-Climbing
 * which choose randomly one of the best solution in the neighborhood at each iteration
 */
template< class Neighbor >
class moRandomBestHCexplorer : public moNeighborhoodExplorer<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;
    typedef moNeighborhood<Neighbor> Neighborhood ;

    using moNeighborhoodExplorer<Neighbor>::neighborhood;
    using moNeighborhoodExplorer<Neighbor>::eval;
    using moNeighborhoodExplorer<Neighbor>::currentNeighbor;
    using moNeighborhoodExplorer<Neighbor>::selectedNeighbor;

    /**
     * Constructor
     * @param _neighborhood the neighborhood
     * @param _eval the evaluation function
     * @param _neighborComparator a neighbor comparator
     * @param _solNeighborComparator solution vs neighbor comparator
     */
    moRandomBestHCexplorer(Neighborhood& _neighborhood,
                           moEval<Neighbor>& _eval,
                           moNeighborComparator<Neighbor>& _neighborComparator,
                           moSolNeighborComparator<Neighbor>& _solNeighborComparator) :
            moNeighborhoodExplorer<Neighbor>(_neighborhood, _eval),
            neighborComparator(_neighborComparator),
            solNeighborComparator(_solNeighborComparator) {
        isAccept = false;
    }

    /**
     * Destructor
     */
    ~moRandomBestHCexplorer() {
    }

    /**
     * empty the vector of best solutions
     * @param _solution unused solution
     */
    virtual void initParam(EOT & /*_solution*/) {
        // delete all the best solutions
        bestVector.clear();
    };

    /**
     * empty the vector of best solutions
     * @param _solution unused solution
     */
    virtual void updateParam(EOT & /*_solution*/) {
        // delete all the best solutions
        bestVector.clear();
    };

    /**
     * terminate: NOTHING TO DO
     * @param _solution unused solution
     */
    virtual void terminate(EOT & /*_solution*/) {};

    /**
     * Explore the neighborhood of a solution
     * @param _solution the current solution
     */
    virtual void operator()(EOT & _solution) {

        //Test if _solution has a Neighbor
        if (neighborhood.hasNeighbor(_solution)) {
            //init the first neighbor
            neighborhood.init(_solution, currentNeighbor);

            //eval the _solution moved with the neighbor and stock the result in the neighbor
            eval(_solution, currentNeighbor);

            //initialize the best neighbor
            bestVector.push_back(currentNeighbor);

            //test all others neighbors
            while (neighborhood.cont(_solution)) {
                //next neighbor
                neighborhood.next(_solution, currentNeighbor);

                //eval
                eval(_solution, currentNeighbor);

                //if we found a better neighbor, update the best
                if (neighborComparator(bestVector[0], currentNeighbor)) {
                    bestVector.clear();
                    bestVector.push_back(currentNeighbor);
                }
                else if (neighborComparator.equals(currentNeighbor, bestVector[0])) //if the current is equals to previous best solutions then update vector of the best solution
                    bestVector.push_back(currentNeighbor);
            }

	    // choose randomly one of the best solutions
	    unsigned int i = rng.random(bestVector.size());

	    // the selected Neighbor
	    selectedNeighbor = bestVector[i];
        }
        else {
            //if _solution hasn't neighbor,
            isAccept=false;
        }
    };

    /**
     * continue if a move is accepted
     * @param _solution the solution
     * @return true if an ameliorated neighbor was be found
     */
    virtual bool isContinue(EOT & /*_solution*/) {
        return isAccept ;
    };

    /**
     * move the solution with the best neighbor
     * @param _solution the solution to move
     */
  /*
    virtual void move(EOT & _solution) {
        // choose randomly one of the best solutions
        unsigned int i = rng.random(bestVector.size());

        //move the solution
        bestVector[i].move(_solution);

        //update its fitness
        _solution.fitness(bestVector[i].fitness());
    };
  */

    /**
     * accept test if an amelirated neighbor was found
     * @param _solution the solution
     * @return true if the best neighbor ameliorate the fitness
     */
    virtual bool accept(EOT & _solution) {
        if (neighborhood.hasNeighbor(_solution))
            isAccept = solNeighborComparator(_solution, selectedNeighbor) ;
        return isAccept;
    };

protected:
    // comparator between solution and neighbor or between neighbors
    moNeighborComparator<Neighbor>& neighborComparator;
    moSolNeighborComparator<Neighbor>& solNeighborComparator;

    // the best solutions in the neighborhood
    std::vector<Neighbor> bestVector;

    // true if the move is accepted
    bool isAccept ;
};


#endif
