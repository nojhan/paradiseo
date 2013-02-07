/*
  <moShiftNeighbor.h>
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

#ifndef _moShiftNeighbor_h
#define _moShiftNeighbor_h

#include <neighborhood/moBackableNeighbor.h>
#include <neighborhood/moIndexNeighbor.h>

/**
 * Indexed Shift Neighbor
 * Other name : insertion operator
 */
template <class EOT, class Fitness=typename EOT::Fitness>
class moShiftNeighbor: public moBackableNeighbor<EOT, Fitness>, public moIndexNeighbor<EOT>
{
public:

  using moBackableNeighbor<EOT>::fitness;
  using moIndexNeighbor<EOT, Fitness>::key;
  using moIndexNeighbor<EOT, Fitness>::index;
  
  /**
   * Apply move on a solution regarding a key
   * @param _solution the solution to move
   */
  virtual void move(EOT & _solution) {
    insertion(_solution, indices.first, indices.second);

    _sol.invalidate();
  }

  /**
   * apply the correct insertion to restore the solution (use by moFullEvalByModif)
   * @param _solution the solution to move back
   */
  virtual void moveBack(EOT& _solution) {
    if (indices.first < indices.second)
      insertion(_solution, indices.second - 1, indices.first);
    else
      insertion(_solution, indices.second, indices.first + 1);
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

    indices.first  = _key % _solution.size() ;
    indices.second = _key / _solution.size() ;

    if (indices.first <= indices.second)
      indices.second += 2;

    // =============== To kill : 
    // int step;
    // int val = _key;
    // int tmpSize = _solution.size() * (_solution.size() - 1) / 2;

    // // moves from left to right
    // if (val <= tmpSize) {
    //   step = _solution.size() - 1;
    //   indices.first = 0;
    //   while ((val - step) > 0) {
    // 	val = val - step;
    // 	step--;
    // 	indices.first++;
    //   }
    //   indices.second = indices.first + val + 1;
    // }
    // // moves from right to left (equivalent moves are avoided)
    // else {  /* val > tmpSize */
    //   val = val - tmpSize;
    //   step = _solution.size() - 2;
    //   indices.second = 0;
    //   while ((val - step) > 0) {
    // 	val = val - step;
    // 	step--;
    // 	indices.second++;
    //   }
    //   indices.first = indices.second + val + 1;
    // }
  }

  /**
   * Setter to fix the two indexes to swap
   * @param _solution solution from which the neighborhood is visited
   * @param _first first index
   * @param _second second index
   */
  void set(EOT & _solution, unsigned int _first, unsigned int _second) {
    indices.first  = _first;
    indices.second = _second;

    // set the index
    if (_first < _second) {
      index( (_second - 2) * _solution.size() + _first );
    } else {
      index( _second * _solution.size() + _first );
    }
  }

  /**
   * Getter of the firt location
   * @return first indice
   */
  unsigned int first() {
    return indices.first;
  }

  /**
   * Getter of the second location
   * @return second indice
   */
  unsigned int second() {
    return indices.second;
  }

  void print() {
    std::cout << key << ": [" << indices.first << ", " << indices.second << "] -> " << (*this).fitness() << std::endl;
  }

private:
  std::pair<unsigned int, unsigned int> indices;    

  /**
   * Apply insertion move on a solution regarding a key
   * @param _sol the solution to move
   * @param _first first position
   * @param _second second position
   */
  void insertion(EOT & _sol, unsigned int _first, unsigned int _second) {
    unsigned int tmp ;

    // keep the first component to change
    tmp = _sol[_first];
    // shift
    if (_first < _second) {
      for (unsigned int i = _first; i < _second - 1; i++)
	_sol[i] = _sol[i+1];
      // shift the first component
      _sol[_second-1] = tmp;
    }
    else {  /* first > second*/
      for (unsigned int i = _first; i > _second; i--)
	_sol[i] = _sol[i-1];
      // shift the first component
      _sol[_second] = tmp;
    }
  }


};

#endif
