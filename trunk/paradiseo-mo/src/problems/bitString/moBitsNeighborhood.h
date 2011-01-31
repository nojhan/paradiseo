/*
  <moBitsNeighborhood.h>
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

#ifndef _moBitsNeighborhood_h
#define _moBitsNeighborhood_h

#include <neighborhood/moNeighborhood.h>
#include <utils/eoRNG.h>
#include <vector>

/**
 * A neighborhood for bit string solutions
 * where several bits could be flipped
 * in a given Hamming distance
 */
template< class Neighbor >
class moBitsNeighborhood : public moNeighborhood<Neighbor>
{
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
  moBitsNeighborhood(unsigned _length, unsigned _nBits): moNeighborhood<Neighbor>(), length(_length), nBits(_nBits) {
    int j;
    bool last;

    neighborhoodSize = 0;

    for(int d = 1; d <= nBits; d++) {
      vector<unsigned int> bits(d);

      // the first one for this Hamming distance
      for(unsigned i = 0; i < d; i++)
	bits[i] = i;

      neighborsVec.push_back(bits);

      /*
      std::cout << "first --" << std::endl;
      for(unsigned i = 0; i < d; i++)
	std::cout << bits[i] ;
      std::cout << std::endl;
      */

      // the others ones
      last = false;

      while(!last) {
	j = d - 1;

	if (bits[j] < length - 1) {
	  bits[j]++;
	  neighborsVec.push_back(bits);
	} else {
	  j--;

	  while ( (j >= 0) && (bits[j] + 1 == bits[j+1]) ) 
	    j--;

	  if (j < 0) {
	    last = true;
	  } else {
	    bits[j]++;

	    for(unsigned i = j+1; i < d; i++)
	      bits[i] = bits[i-1] + 1;

	    neighborsVec.push_back(bits);
	  }
	}

	/*
	std::cout << "fin de while" << std::endl; 
	for(unsigned i = 0; i < d; i++)
	  std::cout << bits[i] << " ";
	std::cout << std::endl;
	*/
      }
    }

    neighborhoodSize = neighborsVec.size();
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
    key = 0;

    _neighbor.bits.resize(nBits);

    setNeighbor(0, _neighbor);
  }
  
  /**
   * Give the next neighbor
   * apply several bit flips on the solution
   * @param _solution the solution to explore (population of solutions)
   * @param _neighbor the next neighbor which in order of distance
   */
  virtual void next(EOT & _solution, Neighbor & _neighbor) {
    key++;

    setNeighbor(key, _neighbor);
  }
  
  /**
   * Test if all neighbors are explored or not,if false, there is no neighbor left to explore
   * @param _solution the solution to explore
   * @return true if there is again a neighbor to explore: population size larger or equals than 1
   */
  virtual bool cont(EOT & _solution) {
    return key < neighborhoodSize - 1;
  }
  
  /**
   * Return the class Name
   * @return the class name as a std::string
   */
  virtual std::string className() const {
    return "moBitsNeighborhood";
  }

  unsigned int index() {
    return key;
  }

protected:
  // length of the bit strings
  unsigned int length;

  // radius of the neighborhood
  unsigned int nBits;

  // size of the neighborhood
  unsigned int neighborhoodSize;

  // list of neighbors 
  vector< vector<unsigned int> > neighborsVec;

  // key of the neighbor which is currently explored
  unsigned int key;


  /**
   * Set the neighbor to the correct neighbor
   * @param _key index in neighborVec of the neighbor to set
   * @param _neighbor neighbor to set
   */
  virtual void setNeighbor(unsigned _key, Neighbor & _neighbor) {
    _neighbor.nBits = neighborsVec[_key].size();

    for(unsigned i = 0; i < _neighbor.nBits; i++)
      _neighbor.bits[i] = neighborsVec[_key][i];
  } 

};

#endif
