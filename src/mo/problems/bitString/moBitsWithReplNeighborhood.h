/*
  <moBitsWithReplNeighborhood.h>
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

#ifndef _moBitsWithReplNeighborhood_h
#define _moBitsWithReplNeighborhood_h

#include "../../neighborhood/moNeighborhood.h"
#include "moBitsNeighborhood.h"
#include "../../../eo/utils/eoRNG.h"
#include <vector>

/**
 * A neighborhood for bit string solutions
 * where several bits could be flipped
 * under a given Hamming distance
 *
 * The neighborhood is explored in a random order
 * Each neighbors is visited once time 
 * and the number of visited neighbors is a parameter
 */
template< class Neighbor >
class moBitsWithReplNeighborhood : public moBitsNeighborhood<Neighbor>
{
  using moBitsNeighborhood<Neighbor>::neighborhoodSize;
  using moBitsNeighborhood<Neighbor>::length;
  using moBitsNeighborhood<Neighbor>::nBits;
  using moBitsNeighborhood<Neighbor>::numberOfNeighbors;

public:

  /**
   * Define type of a solution corresponding to Neighbor
   */
  typedef typename Neighbor::EOT EOT;

  /**
   * Constructor
   *
   * @param _length bit string length
   * @param _nBits maximum number of bits to flip (radius of the neighborhood)
   * @param _sampleSize  number of neighbor to sample in the neighborhood, if 0 all the neighborhood is sampled
   * @param _exactDistance when true, only neighbor with exactly k bits flip are considered, other neighbor <= Hamming distance k
   */
  moBitsWithReplNeighborhood(unsigned _length, unsigned _nBits, unsigned _sampleSize, bool _exactDistance = false): moBitsNeighborhood<Neighbor>(_length, _nBits, _exactDistance), sampleSize(_sampleSize), exactDistance(_exactDistance) {
    if (sampleSize > neighborhoodSize || sampleSize == 0)
      sampleSize = neighborhoodSize;

    indexVector.resize(length);

    for(unsigned int i = 0; i < length; i++)
      indexVector[i] = i;

    if (!exactDistance) {
      nSize.resize(nBits);
      nSize[0] = numberOfNeighbors(1); 
      for(unsigned d = 2; d <= nBits; d++)
	nSize[d - 1] = nSize[d - 2] + numberOfNeighbors(d); 
    }

    nNeighbors = 0;
  }

  /**
   * one random neighbor at Hamming distance _n
   *
   * @param _solution the solution to explore 
   * @param _neighbor the first neighbor
   * @param _n Hamming distance of the neighbor
   */
  virtual void randomNeighbor(EOT & _solution, Neighbor & _neighbor, unsigned _n) {
    _neighbor.bits.resize(_n);
    _neighbor.nBits = _n;

    unsigned i;
    unsigned b;
    unsigned tmp;

    for(unsigned k = 0; k < _n; k++) {
      i = rng.random(length - k);
      b = indexVector[i];

      _neighbor.bits[k] = b;

      indexVector[i]              = indexVector[length - k - 1];
      indexVector[length - k - 1] = b;
    }
  }
  
  /**
   * one random neighbor at maximal Hamming distance _n
   *
   * @param _solution the solution to explore 
   * @param _neighbor the first neighbor
   */
  virtual void randomNeighbor(EOT & _solution, Neighbor & _neighbor) {
    if (exactDistance) 
      randomNeighbor(_solution, _neighbor, nBits);
    else {
      // equiprobability between neighbors at maximal Hamming distance nBits
      unsigned n = rng.random(neighborhoodSize);

      unsigned d = 1;
      while (n < nSize[d - 1]) d++;

      randomNeighbor(_solution, _neighbor, d);
    }
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
    return "moBitsWithReplNeighborhood";
  }

protected:
  // vector of possible bits
  std::vector<unsigned int> indexVector;

  // maximum number of visited neighbor i.e. number of neighbor to sample in the neighborhood
  unsigned int sampleSize;

  // number of visited neighbors
  unsigned nNeighbors;

  // when true, only neighbors at Hamming distance nBits
  bool exactDistance;

  // the number of neighbors below the given distance
  std::vector<unsigned int> nSize;
};

#endif
