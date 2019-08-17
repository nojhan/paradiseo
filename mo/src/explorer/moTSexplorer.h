/*
  <moTSexplorer.h>
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
#ifndef _moTSexplorer_h
#define _moTSexplorer_h

#include "moNeighborhoodExplorer.h"
#include "../comparator/moNeighborComparator.h"
#include "../comparator/moSolNeighborComparator.h"
#include "../memory/moAspiration.h"
#include "../memory/moTabuList.h"
#include "../memory/moIntensification.h"
#include "../memory/moDiversification.h"
#include "../neighborhood/moNeighborhood.h"

/**
 * Explorer for a Tabu Search
 */
template< class Neighbor >
class moTSexplorer : public moNeighborhoodExplorer<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;
    typedef moNeighborhood<Neighbor> Neighborhood ;

    using moNeighborhoodExplorer<Neighbor>::currentNeighbor;
    using moNeighborhoodExplorer<Neighbor>::selectedNeighbor;

    /**
     * Constructor
     * @param _neighborhood the neighborhood
     * @param _eval the evaluation function
     * @param _neighborComparator a neighbor comparator
     * @param _solNeighborComparator a comparator between a solution and a neighbor
     * @param _tabuList the tabu list
     * @param _intensification the intensification box
     * @param _diversification the diversification box
     * @param _aspiration the aspiration criteria
     */
    moTSexplorer(Neighborhood& _neighborhood,
                 moEval<Neighbor>& _eval,
                 moNeighborComparator<Neighbor>& _neighborComparator,
                 moSolNeighborComparator<Neighbor>& _solNeighborComparator,
                 moTabuList<Neighbor> & _tabuList,
                 moIntensification<Neighbor> & _intensification,
                 moDiversification<Neighbor> & _diversification,
                 moAspiration<Neighbor> & _aspiration
                ) :
            moNeighborhoodExplorer<Neighbor>(_neighborhood, _eval), neighborComparator(_neighborComparator), solNeighborComparator(_solNeighborComparator),
            tabuList(_tabuList), intensification(_intensification), diversification(_diversification), aspiration(_aspiration)
    {
        isAccept = false;
    }

    /**
     * Destructor
     */
    ~moTSexplorer() {
    }

    /**
     * init tabu list, intensification box, diversification box and aspiration criteria
     * @param _solution
     */
    virtual void initParam(EOT& _solution)
    {
        tabuList.init(_solution);
        intensification.init(_solution);
        diversification.init(_solution);
        aspiration.init(_solution);
        bestSoFar=_solution;
    };


    /**
     * update params of tabu list, intensification box, diversification box and aspiration criteria
     * @param _solution
      */
    virtual void updateParam(EOT& _solution)
    {
        if ((*this).moveApplied()) {
            tabuList.add(_solution, selectedNeighbor);
            intensification.add(_solution, selectedNeighbor);
            diversification.add(_solution, selectedNeighbor);
            if (_solution.fitness() > bestSoFar.fitness())
                bestSoFar = _solution;
        }
        tabuList.update(_solution, selectedNeighbor);
        intensification.update(_solution, selectedNeighbor);
        diversification.update(_solution, selectedNeighbor);
        aspiration.update(_solution, selectedNeighbor);
    };


    /**
     * terminate : _solution becomes the best so far
     */
    virtual void terminate(EOT & _solution) {
        _solution= bestSoFar;
    };


    /**
     * Explore the neighborhood of a solution
     * @param _solution
     */
    virtual void operator()(EOT & _solution)
    {
        bool found=false;
        intensification(_solution);
        diversification(_solution);
        if (neighborhood.hasNeighbor(_solution))
        {
            //init the current neighbor
            neighborhood.init(_solution, currentNeighbor);
            //eval the current neighbor
            eval(_solution, currentNeighbor);
            
            //Find the first non-tabu element
            if ( (!tabuList.check(_solution, currentNeighbor)) || aspiration(_solution, currentNeighbor) ) {
                // set selectedNeighbor
                selectedNeighbor = currentNeighbor;
                found=true;
            }
            while (neighborhood.cont(_solution) && !found) {
                //next neighbor
                neighborhood.next(_solution, currentNeighbor);
                //eval
                eval(_solution, currentNeighbor);

                if ( (!tabuList.check(_solution, currentNeighbor)) || aspiration(_solution, currentNeighbor) ) {
                    // set selectedNeighbor
                    selectedNeighbor = currentNeighbor;
                    found=true;
                }
            }
            //Explore the neighborhood
            if (found) {
                isAccept=true;
                while (neighborhood.cont(_solution)) {
                    //next neighbor
                    neighborhood.next(_solution, currentNeighbor);
                    //eval
                    eval(_solution, currentNeighbor);
                    //check if the current is better than the best and is not tabu or if it is aspirat (by the aspiration criteria of course)
                    if ( (!tabuList.check(_solution, currentNeighbor) || aspiration(_solution, currentNeighbor)) && neighborComparator(selectedNeighbor, currentNeighbor)) {
                        // set selectedNeighbor
                        selectedNeighbor = currentNeighbor;
                    }
                }
            }
            else {
                isAccept=false;
            }
        }
        else {
            isAccept=false;
        }
        
    };


    /**
     * always continue
     * @param _solution the solution
     * @return true
     */
    virtual bool isContinue(EOT & _solution) {
        return true;
    };

    /**
     * accept test if an ameliorated neighbor was found
     * @param _solution the solution
     * @return true if the best neighbor ameliorate the fitness
     */
    virtual bool accept(EOT & _solution) {
        return isAccept;
    };


  /**
   * Give the current best found so far
   * @return the best solution so far
   */
  const EOT& getBest() {
    return bestSoFar;
  };

protected:

    using moNeighborhoodExplorer<Neighbor>::neighborhood;
    using moNeighborhoodExplorer<Neighbor>::eval;

    // comparator between solution and neighbor or between neighbors
    moNeighborComparator<Neighbor>& neighborComparator;
    moSolNeighborComparator<Neighbor>& solNeighborComparator;

    // Tabu components
    moTabuList<Neighbor> & tabuList;
    moIntensification<Neighbor> & intensification;
    moDiversification<Neighbor> & diversification;
    moAspiration<Neighbor> & aspiration;

    // Best neighbor
  //    Neighbor best;

    //Best so far Solution
    EOT bestSoFar;

    // true if the move is accepted
    bool isAccept ;

};


#endif
