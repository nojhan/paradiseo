/*
  <moPopBitsNeighbor.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sebastien Verel, Arnaud Liefooghe, Jeremie Humeau

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  ue,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

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

#ifndef _moPopBitsNeighbor_h
#define _moPopBitsNeighbor_h

#include <eoPop.h>
#include <problems/bitString/moPopSol.h>
#include <neighborhood/moNeighbor.h>

/**
 * Bits neighbor: apply a bit flip on several solution in the set-solution
 */
template< class EOT >
class moPopBitsNeighbor : public moNeighbor<EOT>
{
public:
  typedef typename EOT::SUBEOT SUBEOT ;
  typedef typename SUBEOT::Fitness SUBFitness;

  using moNeighbor<EOT>::fitness;

  /**
   * Empty Constructor
   */
  moPopBitsNeighbor() : moNeighbor<EOT>() {
  }

  /**
   * Copy Constructor
   * @param _neighbor to copy
   */
  moPopBitsNeighbor(const moPopBitsNeighbor<EOT>& _neighbor) {
    fitness(_neighbor.fitness());

    mutate.resize( _neighbor.mutate.size() );

    for(unsigned int i = 0; i < mutate.size(); i++)
      mutate[i] = _neighbor.mutate[i];

    bits.resize( _neighbor.bits.size() );

    for(unsigned int i = 0; i < bits.size(); i++)
      bits[i] = _neighbor.bits[i];

    fitSol.resize( _neighbor.fitSol.size() );

    for(unsigned int i = 0; i < fitSol.size(); i++)
      fitSol[i] = _neighbor.fitSol[i];
  }

  /**
   * Assignment operator
   * @param _neighbor the neighbor to assign
   * @return a neighbor equal to the other
   */
  virtual moPopBitsNeighbor<EOT>& operator=(const moPopBitsNeighbor<EOT>& _neighbor) {
    fitness(_neighbor.fitness());

    mutate.resize( _neighbor.mutate.size() );

    for(unsigned int i = 0; i < mutate.size(); i++)
      mutate[i] = _neighbor.mutate[i];

    bits.resize( _neighbor.bits.size() );

    for(unsigned int i = 0; i < bits.size(); i++)
      bits[i] = _neighbor.bits[i];

    fitSol.resize( _neighbor.fitSol.size() );

    for(unsigned int i = 0; i < fitSol.size(); i++)
      fitSol[i] = _neighbor.fitSol[i];

    return (*this);
  }

  /**
   * Move the solution according to the information of this neighbor
   * @param _solution the solution to move
   */
  virtual void move(EOT & _solution) {
    if (_solution.size() > 0) {

      for(unsigned i = 0; i < mutate.size(); i++)
	if (mutate[i]) {
	  _solution[i][ bits[i] ] = !_solution[i][ bits[i] ] ;
	  _solution[i].fitness( fitSol[i] );
	}

      _solution.invalidate();
    }
  }

  /**
   * return the class name
   * @return the class name as a std::string
   */
  virtual std::string className() const {
    return "moPopBitsNeighbor";
  }

  /**
   * Read object.\
   * Calls base class, just in case that one had something to do.
   * The read and print methods should be compatible and have the same format.
   * In principle, format is "plain": they just print a number
   * @param _is a std::istream.
   * @throw runtime_std::exception If a valid object can't be read.
   */
  virtual void readFrom(std::istream& _is) {
    std::string fitness_str;
    int pos = _is.tellg();
    _is >> fitness_str;
    if (fitness_str == "INVALID") {
      throw std::runtime_error("invalid fitness");
    }
    else {
      typename EOT::Fitness repFit ;
      _is.seekg(pos);
      _is >> repFit;

      fitness(repFit);

      unsigned int s ;
      _is >> s;

      mutate.resize(s);
      bits.resize(s);
      fitSol.resize(s);

      bool m;
      unsigned int b;
      SUBFitness f;

      for(unsigned i = 0; i < s; i++) {
	_is >> m;
	_is >> b;
	_is >> f;

	mutate[i] = m;
	bits[i]   = b;
	fitSol[i] = f;
      }
    }
  }

  /**
   * Write object. Called printOn since it prints the object _on_ a stream.
   * @param _os A std::ostream.
   */
  virtual void printOn(std::ostream& _os) const {
    _os << fitness() ;

    _os << ' ' << mutate.size();

    for(unsigned int i = 0; i < mutate.size(); i++) 
      _os << ' ' << mutate[i] << ' ' << bits[i] << ' ' << fitSol[i];

    _os << std::endl;
  }

  // Information on the bitflip on each solution: true=bit flip
  vector<bool> mutate;

  // Information on the bit which is flipped
  vector<unsigned int> bits;

  // fitness of the mutated solutions
  vector<SUBFitness> fitSol;
};

#endif
