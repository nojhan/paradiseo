/*
  <moBitFlipNeighborhood.h>
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

#ifndef _moBitFlipNeighborhood_h
#define _moBitFlipNeighborhood_h

#include <neighborhood/moNeighborhood.h>
#include <utils/eoRNG.h>
#include <vector>

/**
 * A neighborhood for bit string solutions
 * where several bits could be flipped
 * with a given rate
 *
 * The neighborhood is explored in a random order
 * the number of visited neighbors is a parameter
 */
template< class Neighbor >
class moBitFlipNeighborhood : public moNeighborhood<Neighbor>
{
public:

  /**
   * Define type of a solution corresponding to Neighbor
   */
  typedef typename Neighbor::EOT EOT;

  /**
   * Constructor
   *
   * @param _rate bit flip rate (per bit)
   * @param _length bit string length
   * @param _sampleSize  number of neighbor to sample in the neighborhood, if 0 all the neighborhood is sampled
   */
  moBitFlipNeighborhood(double _rate, unsigned _length, unsigned _sampleSize): moNeighborhood<Neighbor>(), rate(_rate), length(_length), sampleSize(_sampleSize) {
    nNeighbors = 0;
  }

  /**
   * Test if it exist a neighbor
   * @param _solution the solution to explore
   * @return true if the neighborhood was not empty (bit string larger than 0)
   */
  virtual bool hasNeighbor(EOT& _solution) {
    return _solution.size() > 0;
  }
  
  /**
   * one random neighbor
   *
   * @param _solution the solution to explore 
   * @param _neighbor the first neighbor
   */
  virtual void randomNeighbor(EOT & _solution, Neighbor & _neighbor) {
    // number of flipped bits
    _neighbor.nBits = 0;

    for(unsigned int i = 0; i < _solution.size(); i++) {
      if (rng.flip(rate)) {
	if (_neighbor.nBits < _neighbor.bits.size())
	  _neighbor.bits[_neighbor.nBits] = i;
	else
	  _neighbor.bits.push_back(i);	  
	_neighbor.nBits++;
      }
    }

    // reduce the size if necessary
    if (_neighbor.nBits < _neighbor.bits.size())
      _neighbor.bits.resize(_neighbor.nBits);
  }
  
  /**
   * Initialization of the neighborhood: 
   * one random neighbor 
   *
   * @param _solution the solution to explore 
   * @param _neighbor the first neighbor
   */
  virtual void init(EOT & _solution, Neighbor & _neighbor) {
    randomNeighbor(_solution, _neighbor);

    nNeighbors = 1;
  }
  
  /**
   * Give the next neighbor
   * apply several bit flips on the solution
   * @param _solution the solution to explore (population of solutions)
   * @param _neighbor the next neighbor which in order of distance
   */
  virtual void next(EOT & _solution, Neighbor & _neighbor) {
    randomNeighbor(_solution, _neighbor);

    nNeighbors++;
  }
  
  /**
   * Test if all neighbors are explored or not,if false, there is no neighbor left to explore
   * @param _solution the solution to explore
   * @return true if there is again a neighbor to explore: population size larger or equals than 1
   */
  virtual bool cont(EOT & _solution) {
    return nNeighbors < sampleSize ;
  }
  
  /**
   * Return the class Name
   * @return the class name as a std::string
   */
  virtual std::string className() const {
    return "moBitFlipNeighborhood";
  }

protected:
  // bit flip rate
  double rate;

  // length of the bit string
  unsigned int length;

  // maximum number of visited neighbor i.e. number of neighbor to sample in the neighborhood
  unsigned int sampleSize;

  // number of visited neighbors
  unsigned nNeighbors;
};

#endif
