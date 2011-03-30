/*
  <moBitsWithoutReplNeighborhood.h>
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

#ifndef _moBitsWithoutReplNeighborhood_h
#define _moBitsWithoutReplNeighborhood_h

#include <neighborhood/moNeighborhood.h>
#include <problems/bitString/moBitsNeighborhood.h>
#include <utils/eoRNG.h>
#include <vector>

/**
 * A neighborhood for bit string solutions
 * where several bits could be flipped
 * under a given Hamming distance
 *
 * The neighborhood is explred in a random order
 * and the number of neighbors is fixed by a number
 */
template< class Neighbor >
class moBitsWithoutReplNeighborhood : public moBitsNeighborhood<Neighbor>
{
  using moBitsNeighborhood<Neighbor>::neighborhoodSize;
  using moBitsNeighborhood<Neighbor>::setNeighbor;
  using moBitsNeighborhood<Neighbor>::key;
  using moBitsNeighborhood<Neighbor>::nBits;

public:

  /**
   * Define type of a solution corresponding to Neighbor
   */
  typedef typename Neighbor::EOT EOT;

  /**
   * Constructor
   * @param _length bit string length
   * @param _nBits maximum number of bits to flip (radius of the neighborhood)
   */
  moBitsWithoutReplNeighborhood(unsigned _length, unsigned _nBits, unsigned _sampleSize): moBitsNeighborhood<Neighbor>(_length, _nBits), sampleSize(_sampleSize) {
    if (sampleSize > neighborhoodSize || sampleSize == 0)
      sampleSize = neighborhoodSize;

    indexVector.resize(neighborhoodSize);

    for(unsigned int i = 0; i < neighborhoodSize; i++)
      indexVector[i] = i;
  }

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
   * apply several bit flips on the solution
   * @param _solution the solution to explore 
   * @param _neighbor the first neighbor
   */
  virtual void init(EOT & _solution, Neighbor & _neighbor) {
    maxIndex = neighborhoodSize ;

    unsigned i = rng.random(maxIndex);
    key = indexVector[i];

    unsigned tmp              = indexVector[i];
    indexVector[i]            = indexVector[maxIndex - 1];
    indexVector[maxIndex - 1] = tmp;
    maxIndex--;

    _neighbor.bits.resize(nBits);
    setNeighbor(key, _neighbor);
  }
  
  /**
   * Give the next neighbor
   * apply several bit flips on the solution
   * @param _solution the solution to explore (population of solutions)
   * @param _neighbor the next neighbor which in order of distance
   */
  virtual void next(EOT & _solution, Neighbor & _neighbor) {
    unsigned i = rng.random(maxIndex);
    key = indexVector[i];

    unsigned tmp              = indexVector[i];
    indexVector[i]            = indexVector[maxIndex - 1];
    indexVector[maxIndex - 1] = tmp;
    maxIndex--;

    setNeighbor(key, _neighbor);
  }
  
  /**
   * Test if all neighbors are explored or not,if false, there is no neighbor left to explore
   * @param _solution the solution to explore
   * @return true if there is again a neighbor to explore: population size larger or equals than 1
   */
  virtual bool cont(EOT & _solution) {
    return neighborhoodSize - maxIndex < sampleSize ;
  }
  
  /**
   * Return the class Name
   * @return the class name as a std::string
   */
  virtual std::string className() const {
    return "moBitsWithoutReplNeighborhood";
  }

  unsigned int index() {
    return key;
  }

protected:
  // number of remainded neighbors to sample
  unsigned int maxIndex;

  // vector of possible index
  std::vector<unsigned int> indexVector;

  // maximum number of visited neighbor i.e. number of neighbor to sample in the neighborhood
  unsigned int sampleSize;
};

#endif
