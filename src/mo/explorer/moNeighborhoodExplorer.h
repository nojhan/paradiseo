/*
  <moNeighborhoodExplorer.h>
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

#ifndef _neighborhoodExplorer_h
#define _neighborhoodExplorer_h

//EO inclusion
#include "../../eo/eoFunctor.h"

#include "../neighborhood/moNeighborhood.h"
#include "../eval/moEval.h"
#include "../neighborhood/moDummyNeighborhood.h"
#include "../eval/moDummyEval.h"

/**
 * Explore the neighborhood according to the local search algorithm 
 *
 * During this exploration,
 *   the parameters are updated
 *   one neighbor is selected
 *   a comparason with the solution is made to acccept or not this new neighbor
 * 
 * The current neighbor (currentNeigbor) is the neighbor under consideration during the search (in operator()(EOT &))
 *
 * The selected neighbor (selectedNeighbor) is the neighbor selected in method operator()(EOT &). 
 * If this neighbor is accepted, then the solution is moved on this neighbor (in move(EOT &))
 *
 */
template< class Neighbor >
class moNeighborhoodExplorer : public eoUF<typename Neighbor::EOT&, void>
{
public:
    typedef moNeighborhood<Neighbor> Neighborhood;
    typedef typename Neighbor::EOT EOT;
    typedef typename EOT::Fitness Fitness ;

    moNeighborhoodExplorer():neighborhood(dummyNeighborhood), eval(dummyEval), isMoved(false) {}

    /**
     * Constructor with a Neighborhood and evaluation function
     * @param _neighborhood the neighborhood
     * @param _eval the evaluation function
     */
    moNeighborhoodExplorer(Neighborhood& _neighborhood, moEval<Neighbor>& _eval):neighborhood(_neighborhood), eval(_eval), isMoved(false) {}

    /**
     * Init Search parameters
     * @param _solution the solution to explore
     */
    virtual void initParam (EOT& _solution) = 0 ;

    /**
     * Update Search parameters
     * @param _solution the solution to explore
     */
    virtual void updateParam (EOT& _solution) = 0 ;

    /**
     * Test if the exploration continue or not
     * @param _solution the solution to explore
     * @return true if the exploration continue, else return false
     */
    virtual bool isContinue(EOT& _solution) = 0 ;

    /**
     * Move a solution on the selected neighbor
     * This method can be re-defined according to the metaheuritic
     *
     * @param _solution the solution to explore
     */
    virtual void move(EOT& _solution) {
      // move the solution
      selectedNeighbor.move(_solution);

      // update the fitness
      _solution.fitness(selectedNeighbor.fitness());
    }

    /**
     * Test if a solution is accepted
     * @param _solution the solution to explore
     * @return true if the solution is accepted, else return false
     */
    virtual bool accept(EOT& _solution) = 0 ;

    /**
     * Terminate the search
     * @param _solution the solution to explore
     */
    virtual void terminate(EOT& _solution) = 0 ;

    /**
     * Getter of "isMoved"
     * @return true if move is applied
     */
    bool moveApplied() {
        return isMoved;
    }

    /**
     * Setter of "isMoved"
     * @param _isMoved
     */
    void moveApplied(bool _isMoved) {
        isMoved=_isMoved;
    }

  /**
   * Getter of the current neighbor
   * @return current neighbor
   */
  Neighbor & getCurrentNeighbor() {
    return currentNeighbor;
  }
  
  /**
   * Getter of the selected neighbor
   * @return selected neighbor
   */
  Neighbor & getSelectedNeighbor() {
    return selectedNeighbor;
  }
  
protected:
  // default class for the empty constructor
    moDummyNeighborhood<Neighbor> dummyNeighborhood;

  // default class for the empty constructor
    moDummyEval<Neighbor> dummyEval;

  // the neighborhood which is explored
    Neighborhood & neighborhood;

  // evaluation of a neighbor
    moEval<Neighbor>& eval;

  // flag : true a the neighbor is accepted and the movement is made
    bool isMoved;

  // the current neighbor of the exploration : common features of algorithm
    Neighbor currentNeighbor;

  // the selected neighbor after the exploration of the neighborhood
    Neighbor selectedNeighbor;
};

#endif
