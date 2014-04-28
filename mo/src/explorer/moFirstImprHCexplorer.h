/*
  <moFirstImprHCexplorer.h>
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

#ifndef _moFirstImprHCexplorer_h
#define _moFirstImprHCexplorer_h

#include <explorer/moNeighborhoodExplorer.h>
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
#include <neighborhood/moNeighborhood.h>

/**
 * Explorer for a first improvement heuristic
 */
template< class Neighbor>
class moFirstImprHCexplorer : public moNeighborhoodExplorer<Neighbor>
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
     * @param _solNeighborComparator a solution vs neighbor comparator
     */
    moFirstImprHCexplorer(Neighborhood& _neighborhood, moEval<Neighbor>& _eval, moNeighborComparator<Neighbor>& _neighborComparator, moSolNeighborComparator<Neighbor>& _solNeighborComparator) : moNeighborhoodExplorer<Neighbor>(_neighborhood, _eval), neighborComparator(_neighborComparator), solNeighborComparator(_solNeighborComparator) {
        isAccept = false;
	stop = true;
    }

    /**
     * Destructor
     */
    ~moFirstImprHCexplorer() {
    }

    /**
     * to never stop the hill climbing
     * 
     */
    virtual void alwaysContinue() {
      stop = false;
    }

    /**
     * initParam: NOTHING TO DO
     * @param _solution unused solution
     */
    virtual void initParam(EOT & _solution) {};

    /**
     * updateParam: NOTHING TO DO
     * @param _solution unused solution
     */
    virtual void updateParam(EOT & _solution) {};

    /**
     * terminate: NOTHING TO DO
     * @param _solution unused solution
     */
    virtual void terminate(EOT & _solution) {};

    /**
     * Explore the neighborhood of a solution until an ameliorated neighbor is found
     * @param _solution
     */
    virtual void operator()(EOT & _solution) {

        //Test if _solution has a Neighbor
        if (neighborhood.hasNeighbor(_solution)) {
            //init the first neighbor
            neighborhood.init(_solution, currentNeighbor);

            //eval the _solution moved with the neighbor and stock the result in the neighbor
            eval(_solution, currentNeighbor);

            //test all others neighbors
            while (! solNeighborComparator(_solution, currentNeighbor) && neighborhood.cont(_solution)) {
                //next neighbor
                neighborhood.next(_solution, currentNeighbor);
                //eval
                eval(_solution, currentNeighbor);
            }
	    
	    // the selected neighbor
	    selectedNeighbor = currentNeighbor;
        }
        else {
            //if _solution hasn't neighbor,
            isAccept=false;
        }
    };

    /**
     * continue if a move is accepted, or according to the flag 'stop'
     * @param _solution the solution
     * @return true if an ameliorated neighbor was found
     */
    virtual bool isContinue(EOT & _solution) {
      if (stop)
        return isAccept ;
      else
	return true;
    };

    /**
     * accept test if an ameliorated neighbor was found
     * @param _solution the solution
     * @return true if the best neighbor ameliorate the fitness
     */
    virtual bool accept(EOT & _solution) {
        if (neighborhood.hasNeighbor(_solution)) {
            isAccept = solNeighborComparator(_solution, selectedNeighbor) ;
        }
        return isAccept;
    };

private:
    // comparator between solution and neighbor or between neighbors
    moNeighborComparator<Neighbor>& neighborComparator;
    moSolNeighborComparator<Neighbor>& solNeighborComparator;

    // true if the move is accepted
    bool isAccept ;

  // if true the HC stop when to improving solution is found
  // if false : never stop, always continue (external continuator)
  bool stop ;
};


#endif
