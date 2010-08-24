/*
  <moPopBitsRndNeighborhood.h>
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

#ifndef _moPopBitsRndNeighborhood_h
#define _moPopBitsRndNeighborhood_h

#include <neighborhood/moRndNeighborhood.h>
#include <utils/eoRNG.h>

/**
 * A Bits neighborhood with one random bit flip on random solutions
 */
template< class Neighbor >
class moPopBitsRndNeighborhood : public moRndNeighborhood<Neighbor>
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
   * @param _rate mutation rate
   */
  moPopBitsRndNeighborhood(double _rate): moRndNeighborhood<Neighbor>(), mutationRate(_rate) {}

  /**
   * Test if it exist a neighbor
   * @param _solution the solution to explore
   * @return true if the neighborhood was not empty: the population size is at least 1
   */
  virtual bool hasNeighbor(EOT& _solution) {
    return _solution.size() > 0;
  }
  
  /**
   * Initialization of the neighborhood: 
   * apply one bit flip on several solutions according to the mutation rate
   * @param _solution the solution to explore (population of solutions)
   * @param _neighbor the first neighbor
   */
  virtual void init(EOT & _solution, Neighbor & _neighbor) {
    unsigned int popSize = _solution.size();
    unsigned int length  = _solution[0].size();

    _neighbor.mutate.resize(popSize);
    _neighbor.bits.resize(popSize);
    _neighbor.fitSol.resize(popSize);
    
    for(unsigned int i = 0; i < popSize; i++) {
      _neighbor.mutate[i] = rng.uniform() < mutationRate;

      if (_neighbor.mutate[i]) 
	_neighbor.bits[i] = rng.random(length);
    }
  }
  
  /**
   * Give the next neighbor
   * apply one bit flip on several solutions according to the mutation rate
   * @param _solution the solution to explore (population of solutions)
   * @param _neighbor the next neighbor which is "random"
   */
  virtual void next(EOT & _solution, Neighbor & _neighbor) {
    unsigned int popSize = _solution.size();
    unsigned int length  = _solution[0].size();

    for(unsigned int i = 0; i < popSize; i++) {
      _neighbor.mutate[i] = rng.uniform() < mutationRate;

      if (_neighbor.mutate[i]) 
	_neighbor.bits[i] = rng.random(length);
    }
  }
  
  /**
   * Test if all neighbors are explored or not,if false, there is no neighbor left to explore
   * @param _solution the solution to explore
   * @return true if there is again a neighbor to explore: population size larger or equals than 1
   */
  virtual bool cont(EOT & _solution) {
    return _solution.size() > 0;
  }
  
  /**
   * Return the class Name
   * @return the class name as a std::string
   */
  virtual std::string className() const {
    return "moPopBitsRndNeighborhood";
  }

private:
  //mutation rate
  double mutationRate;
};

#endif
