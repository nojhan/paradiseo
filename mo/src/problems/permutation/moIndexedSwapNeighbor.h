/*
<moIndexedSwapNeighbor.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

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

#ifndef _moIndexedSwapNeighbor_h
#define _moIndexedSwapNeighbor_h

#include "../../neighborhood/moBackableNeighbor.h"
#include "../../neighborhood/moIndexNeighbor.h"

/**
 * Indexed Swap Neighbor: the position of the swap are computed according to the index
 */
template <class EOT, class Fitness=typename EOT::Fitness>
class moIndexedSwapNeighbor: public moBackableNeighbor<EOT, Fitness>, public moIndexNeighbor<EOT, Fitness>
{
public:

  using moBackableNeighbor<EOT, Fitness>::fitness;
  using moIndexNeighbor<EOT, Fitness>::key;
  using moIndexNeighbor<EOT, Fitness>::index;
  
  /**
   * Default Constructor
   */
  moIndexedSwapNeighbor() : moIndexNeighbor<EOT, Fitness>() {
  }

  /**
   * Copy Constructor
   * @param _n the neighbor to copy
   */
  moIndexedSwapNeighbor(const moIndexedSwapNeighbor<EOT, Fitness> & _n) : moIndexNeighbor<EOT, Fitness>(_n)
  {
    indices.first = _n.first();
    indices.second = _n.second();
  }

  /**
   * Assignment operator
   * @param _source the source neighbor
   */
  moIndexedSwapNeighbor<EOT, Fitness> & operator=(const moIndexedSwapNeighbor<EOT, Fitness> & _source) {
    moIndexNeighbor<EOT, Fitness>::operator=(_source);
    indices.first = _source.first();
    indices.second = _source.second();
    return *this;
  }

 /**
   * Apply the swap
   * @param _solution the solution to move
   */
  virtual void move(EOT& _solution) {
    // bof utiliser plutot le template du vector : to do
//    EOT tmp(1);
//    
//    tmp[0] = _solution[indices.first];
//    _solution[indices.first] = _solution[indices.second];
//    _solution[indices.second] = tmp[0];
      std::swap(_solution[indices.first], _solution[indices.second]);
    _solution.invalidate();
  }
  
  /**
   * Return the class Name
   * @return the class name as a std::string
   */
  virtual std::string className() const {
      return "moIndexedSwapNeighbor";
  }

  /**
   * apply the swap to restore the solution (use by moFullEvalByModif)
   * @param _solution the solution to move back
   */
  virtual void moveBack(EOT& _solution) {
    move(_solution);
  }
  
  /**
   * Setter 
   * The "parameters" of the neighbor is a function of key and the current solution
   * for example, for variable length solution
   *
   * @param _solution solution from which the neighborhood is visited
   * @param _key index of the IndexNeighbor
   */
  virtual void index(EOT & _solution, unsigned int _key) {
    index( _key );
    
    unsigned int n = (unsigned int) ( (1 + sqrt(1 + 8 * _key)) / 2);

    indices.first  = _key - (n - 1) * n / 2;
    indices.second = _solution.size() - 1  - (n - 1 - indices.first);
  }

  /**
   * Setter to fix the two indexes to swap
   * @param _solution solution from which the neighborhood is visited
   * @param _first first index
   * @param _second second index
   */
  void set(EOT & _solution, unsigned int _first, unsigned int _second) {
    indices.first = _first;
    indices.second = _second;

    // set the index
    unsigned n = _solution.size()  + _first - _second;
    index( _first + n * (n - 1) / 2 );
  }

  /**
   * Getter of the firt location
   * @return first indice
   */
  unsigned int first() const {
    return indices.first;
  }

  /**
   * Getter of the second location
   * @return second indice
   */
  unsigned int second() const {
    return indices.second;
  }

protected:
  std::pair<unsigned int, unsigned int> indices;    

};

#endif
