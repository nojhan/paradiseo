/*
  <moNeighborhoodStat.h>
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

#ifndef moNeighborhoodStat_h
#define moNeighborhoodStat_h

#include <continuator/moStat.h>

#include <explorer/moNeighborhoodExplorer.h>
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
#include <neighborhood/moNeighborhood.h>

/**
 * All possible statitic on the neighborhood fitness
 * to combine with other specific statistic to print them
 */
template< class Neighbor >
class moNeighborhoodStat : public moStat<typename Neighbor::EOT, bool>
{
public :
    typedef typename Neighbor::EOT EOT ;
    typedef moNeighborhood<Neighbor> Neighborhood ;
    typedef typename EOT::Fitness Fitness ;

    using moStat< EOT, bool >::value;

    /**
     * Constructor
     * @param _neighborhood a neighborhood
     * @param _eval an evaluation function
     * @param _neighborComparator a neighbor Comparator
     * @param _solNeighborComparator a comparator between a solution and a neighbor
     */
    moNeighborhoodStat(Neighborhood& _neighborhood, moEval<Neighbor>& _eval, moNeighborComparator<Neighbor>& _neighborComparator, moSolNeighborComparator<Neighbor>& _solNeighborComparator):
            moStat<EOT, bool>(true, "neighborhood"),
            neighborhood(_neighborhood), eval(_eval),
            neighborComparator(_neighborComparator),
            solNeighborComparator(_solNeighborComparator)
    {}

    /**
     * Default Constructor
     * where the comparators are basic, there only compare the fitness values
     *
     * @param _neighborhood a neighborhood
     * @param _eval an evaluation function
     */
    moNeighborhoodStat(Neighborhood& _neighborhood, moEval<Neighbor>& _eval):
            moStat<EOT, bool>(true, "neighborhood"),
            neighborhood(_neighborhood), eval(_eval),
            neighborComparator(defaultNeighborComp),
            solNeighborComparator(defaultSolNeighborComp)
    {}

    /**
     * Compute classical statistics of the first solution's neighborhood
     * @param _solution the first solution
     */
    virtual void init(EOT & _solution) {
        operator()(_solution);
    }

    /**
     * Compute classical statistics of a solution's neighborhood
     * @param _solution the corresponding solution
     */
    virtual void operator()(EOT & _solution) {
        Neighbor current ;
        Neighbor best ;
        Neighbor lowest ;

        if (neighborhood.hasNeighbor(_solution)) {
            //init the first neighbor
            neighborhood.init(_solution, current);

            //eval the _solution moved with the neighbor and stock the result in the neighbor
            eval(_solution, current);

            // init the statistics
            value() = true;

            mean = current.fitness();
            sd   = mean * mean;
            nb      = 1;
            nbInf   = 0;
            nbEqual = 0;
            nbSup   = 0;

            if (solNeighborComparator.equals(_solution, current))
                nbEqual++;
            else if (solNeighborComparator(_solution, current))
                nbSup++;
            else
                nbInf++;

            //initialize the best neighbor
            best   = current;
            lowest = current;

            //test all others neighbors
            while (neighborhood.cont(_solution)) {
                //next neighbor
                neighborhood.next(_solution, current);
                //eval
                eval(_solution, current);

                mean += current.fitness();
                sd += current.fitness() * current.fitness();
                nb++;

                if (solNeighborComparator.equals(_solution, current))
                    nbEqual++;
                else if (solNeighborComparator(_solution, current))
                    nbSup++;
                else
                    nbInf++;

                //if we found a better neighbor, update the best
                if (neighborComparator(best, current))
                    best = current;

                if (neighborComparator(current, lowest))
                    lowest = current;
            }

            max = best.fitness();
            min = lowest.fitness();

            mean /= nb;
            if (nb > 1)
                sd = sqrt( (sd - nb * mean * mean) / (nb - 1.0) );
            else
                sd = 0.0;
        }
        else {
            //if _solution hasn't neighbor,
            value() = false;
        }
    }

    /**
     * @return the worst fitness value found in the neighborhood
     */
    Fitness getMin() {
        return min;
    }

    /**
     * @return the best fitness value found in the neighborhood
     */
    Fitness getMax() {
        return max;
    }

    /**
     * @return the mean fitness value of the neighborhood
     */
    double getMean() {
        return mean;
    }

    /**
     * @return the standard deviation value of the neighborhood
     */
    double getSD() {
        return sd;
    }

    /**
     * @return the size of the neighborhood
     */
    unsigned getSize() {
        return nb;
    }

    /**
     * @return the number of neighbors having a better fitness than the current solution
     */
    unsigned getNbSup() {
        return nbSup;
    }

    /**
     * @return the number of neighbors having the same fitness than the current solution
     */
    unsigned getNbEqual() {
        return nbEqual;
    }

    /**
     * @return the number of neighbors having a worst fitness than the current solution
     */
    unsigned getNbInf() {
        return nbInf;
    }

    /**
     * @return the class name
     */
    virtual std::string className(void) const {
        return "moNeighborhoodStat";
    }

protected:

    //the neighborhood
    Neighborhood& neighborhood ;
    moEval<Neighbor>& eval;

    // comparator between solution and neighbor or between neighbors
    moNeighborComparator<Neighbor>& neighborComparator;
    moSolNeighborComparator<Neighbor>& solNeighborComparator;

    // default comparators
    // compare the fitness values of neighbors
    moNeighborComparator<Neighbor> defaultNeighborComp;
    // compare the fitness values of the solution and the neighbor
    moSolNeighborComparator<Neighbor> defaultSolNeighborComp;

    // the stastics of the fitness
    Fitness max, min;
    //mean and standard deviation
    double mean, sd ;

    // number of neighbors in the neighborhood;
    unsigned int nb;

    // number of neighbors with lower, equal and higher fitness
    unsigned int nbInf, nbEqual, nbSup ;
};

#endif
