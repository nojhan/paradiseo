/*
  <moPopXoverRndNeighborhood.h>
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

#ifndef _moPopXoverRndNeighborhood_h
#define _moPopXoverRndNeighborhood_h

#include <neighborhood/moRndNeighborhood.h>
#include <eoOp.h>
#include <utils/eoRNG.h>

/**
 * A Xover neighborhood with random crossover between random solutions
 */
template< class Neighbor >
class moPopXoverRndNeighborhood : public moRndNeighborhood<Neighbor>
{
public:

  /**
   * Define type of a solution corresponding to Neighbor
   */
  typedef typename Neighbor::EOT EOT;

  /**
   * Define type of a sub-solution which compose the population
   */
  typedef typename EOT::SUBEOT SUBEOT;

  /**
   * Constructor
   * @param _crossover the crossover operator
   * @param _maxNeighbor maximum neighbors to explore in the neighborhood
   */
  moPopXoverRndNeighborhood(eoQuadOp<SUBEOT> & _crossover, unsigned int _maxNeighbor = 0): moRndNeighborhood<Neighbor>(), crossover(_crossover), maxNeighbor(_maxNeighbor) {}

  /**
   * Test if it exist a neighbor
   * @param _solution the solution to explore
   * @return true if the neighborhood was not empty: the opulation soze is at least 2
   */
  virtual bool hasNeighbor(EOT& _solution) {
    return _solution.size() > 1;
  }
  
  /**
   * Initialization of the neighborhood: 
   * apply a ones the crossover operato between random solutions
   * @param _solution the solution to explore (population of solutions)
   * @param _neighbor the first neighbor
   */
  virtual void init(EOT & _solution, Neighbor & _neighbor) {
    next(_solution, _neighbor);

    nbNeighbor = 0;
  }
  
  /**
   * Give the next neighbor
   * apply a ones the crossover operato between random solutions
   * @param _solution the solution to explore (population of solutions)
   * @param _neighbor the next neighbor which is "random"
   */
  virtual void next(EOT & _solution, Neighbor & _neighbor) {
    unsigned int popSize = _solution.size();

    // random solutions in the population
    unsigned int i1, i2;

    i1 = rng.random(popSize);
    i2 = rng.random(popSize);

    while (i2 == i1)
      i2 = rng.random(popSize);

    _neighbor.setIndexes(i1, i2);

    // copy the solutions
    _neighbor.solution1() = _solution[i1];
    _neighbor.solution2() = _solution[i2];

    // apply the crossover
    crossover(_neighbor.solution1(), _neighbor.solution2());

    // increase the number of neighbor explored
    nbNeighbor++;
  }
  
  /**
   * Test if all neighbors are explored or not,if false, there is no neighbor left to explore
   * @param _solution the solution to explore
   * @return true if there is again a neighbor to explore: population size larger or equals than 2
   */
  virtual bool cont(EOT & _solution) {
    if (maxNeighbor != 0)
      return (_solution.size() > 1) && (nbNeighbor < maxNeighbor);
    else
      return _solution.size() > 1;
  }
  
  /**
   * Return the class Name
   * @return the class name as a std::string
   */
  virtual std::string className() const {
    return "moPopXoverRndNeighborhood";
  }
  
private:
  eoQuadOp<SUBEOT> & crossover;
  unsigned int maxNeighbor;
  unsigned int nbNeighbor;
};

#endif
