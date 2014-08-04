/*
  <moNeighborBestStat.h>
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

#ifndef moNeighborBestStat_h
#define moNeighborBestStat_h

#include "moStat.h"

#include "../explorer/moNeighborhoodExplorer.h"
#include "../comparator/moNeighborComparator.h"
#include "../comparator/moSolNeighborComparator.h"
#include "../neighborhood/moNeighborhood.h"

/**
 * Compute the fitness of the best solution among k neighbor or all neighbors
 */
template< class Neighbor >
class moNeighborBestStat : public moStat<typename Neighbor::EOT, typename Neighbor::EOT::Fitness>
{
public :
    typedef typename Neighbor::EOT EOT ;
    typedef moNeighborhood<Neighbor> Neighborhood ;
    typedef typename EOT::Fitness Fitness ;

    using moStat< EOT, Fitness >::value;

    /**
     * Constructor
     * @param _neighborhood a neighborhood
     * @param _eval an evaluation function
     * @param _neighborComparator a neighbor Comparator
     * @param _solNeighborComparator a comparator between a solution and a neighbor
     * @param _k number of neighbors visited
     */
    moNeighborBestStat(Neighborhood& _neighborhood, moEval<Neighbor>& _eval, moNeighborComparator<Neighbor>& _neighborComparator, moSolNeighborComparator<Neighbor>& _solNeighborComparator, unsigned int _k = 0):
            moStat<EOT, Fitness>(true, "neighborhood"),
            neighborhood(_neighborhood), eval(_eval),
            neighborComparator(_neighborComparator),
            solNeighborComparator(_solNeighborComparator),
            kmax(_k)
    {}

    /**
     * Default Constructor
     * where the comparators are basic, there only compare the fitness values
     *
     * @param _neighborhood a neighborhood
     * @param _eval an evaluation function
     * @param _k number of neighbors visited (default all)
     */
    moNeighborBestStat(Neighborhood& _neighborhood, moEval<Neighbor>& _eval, unsigned _k = 0):
            moStat<EOT, Fitness>(Fitness(), "best"),
            neighborhood(_neighborhood), eval(_eval),
            neighborComparator(defaultNeighborComp),
            solNeighborComparator(defaultSolNeighborComp),
            kmax(_k)
    {}

    /**
     * Compute classical statistics of the first solution's neighborhood
     * @param _solution the first solution
     */
    virtual void init(EOT & _solution) {
        operator()(_solution);
    }

    /**
     * Compute the best fitness amoung all neighbors or k neighbors
     * @param _solution the corresponding solution
     */
    virtual void operator()(EOT & _solution) {
        Neighbor current ;
        Neighbor best ;

        if (neighborhood.hasNeighbor(_solution)) {
            //init the first neighbor
            neighborhood.init(_solution, current);

            //eval the _solution moved with the neighbor and stock the result in the neighbor
            eval(_solution, current);

            //initialize the best neighbor
            best   = current;

            // number of visited neighbors
            unsigned int k = 1;

            //test all others neighbors
            while ( ( (kmax == 0) || (k < kmax) ) && neighborhood.cont(_solution)) {
                //next neighbor
                neighborhood.next(_solution, current);
                //eval
                eval(_solution, current);

                //if we found a better neighbor, update the best
                if (neighborComparator(best, current))
                    best = current;

                k++;
            }

            value() = best.fitness();
        }
        else {
            //if _solution hasn't neighbor,
            value() = Fitness();
        }
    }

    /**
     * @return the class name
     */
    virtual std::string className(void) const {
        return "moNeighborBestStat";
    }

private:
    // to explore the neighborhood
    Neighborhood& neighborhood ;
    moEval<Neighbor>& eval;

    // comparator betwenn solution and neighbor or between neighbors
    moNeighborComparator<Neighbor>& neighborComparator;
    moSolNeighborComparator<Neighbor>& solNeighborComparator;

    // default comparators
    // compare the fitness values of neighbors
    moNeighborComparator<Neighbor> defaultNeighborComp;
    // compare the fitness values of the solution and the neighbor
    moSolNeighborComparator<Neighbor> defaultSolNeighborComp;

    // number of neighbor to explore
    unsigned int kmax;
};

#endif
