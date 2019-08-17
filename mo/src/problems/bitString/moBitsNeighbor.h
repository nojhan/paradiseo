/*
  <moBitsNeighbor.h>
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

#ifndef _moBitsNeighbor_h
#define _moBitsNeighbor_h

#include "../../neighborhood/moNeighbor.h"
#include <vector>

/**
 * Neighbor to flip several bits
 * of a solution of type eoBit
 */
template< class EOT, class Fitness=typename EOT::Fitness >
class moBitsNeighbor : virtual public moNeighbor<EOT, Fitness>, public moBackableNeighbor<EOT>
{
public:

  // position of bits which are flipped
  std::vector<unsigned int> bits;

  // number of bits to flip
  unsigned int nBits;

  /**
   * Default Constructor
   */
  moBitsNeighbor() : moNeighbor<EOT, Fitness>() {}

  /**
   * Copy Constructor
   * @param _source the neighbor to copy
   */
  moBitsNeighbor(const moBitsNeighbor& _source) : moNeighbor<EOT, Fitness>(_source) {
    bits.resize(_source.bits.size());

    nBits = _source.nBits;

    for(unsigned i = 0; i < bits.size(); i++)
      this->bits[i] = _source.bits[i] ;
  }

  /**
   * Assignment operator
   * @param _source the source neighbor
   */
  moBitsNeighbor<EOT, Fitness> & operator=(const moBitsNeighbor<EOT, Fitness> & _source) {
    moNeighbor<EOT, Fitness>::operator=(_source);

    if (bits.size() != _source.bits.size())
      bits.resize(_source.bits.size());

    nBits = _source.nBits;

    for(unsigned i = 0; i < bits.size(); i++)
      this->bits[i] = _source.bits[i] ;

    return *this ;
  }

  /**
   * Return the class Name
   * @return the class name as a std::string
   */
  virtual std::string className() const {
    return "moBitsNeighbor";
  }
  
  /**
   * flipped the bits according to the bits vector
   * @param _solution the solution to move
   */
  virtual void move(EOT& _solution) {
    for(unsigned i = 0; i < nBits; i++)
      _solution[ this->bits[i] ] = !_solution[ this->bits[i] ];
  }

  /**
   * flipped the bits according to the bits vector
   * @param _solution the solution to move back
   */
  virtual void moveBack(EOT& _solution) {
    for(unsigned i = 0; i < nBits; i++)
      _solution[ this->bits[i] ] = !_solution[ this->bits[i] ];
  }

  /**
   * @param _neighbor a neighbor
   * @return if _neighbor and this one are equals
   */
  virtual bool equals(moBitsNeighbor<EOT,Fitness> & _neighbor) {
    if (nBits != _neighbor.nBits)
      return false;
    else {
      unsigned int i = 0;

      while ((i < nBits) && (bits[i] == _neighbor.bits[i])) i++;

      if (i < nBits)
	return false;
      else
	return true;
    }
  }

  /**
   * Write object. Called printOn since it prints the object _on_ a stream.
   * @param _os A std::ostream.
   */
  virtual void printOn(std::ostream& _os) const {
    _os << this->fitness() << " " << nBits ;
    for(unsigned int i = 0; i < nBits; i++)
      _os << " "  << bits[i] ;
  }

};

#endif
