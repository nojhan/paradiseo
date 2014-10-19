/*
  <moRndIndexedVectorTabuList.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sebastien Verel

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

#ifndef _moRndIndexedVectorTabuList_h
#define _moRndIndexedVectorTabuList_h

#include "moIndexedVectorTabuList.h"
#include "../../eo/utils/eoRndGenerators.h"


/**
 * Tabu List of indexed neighbors save in a vector
 * each neighbor can not used during howlong + rnd(howlongRnd) iterations
 * see paper: 
 * Zhipeng Lu, Fred Glover, Jin-Kao Hao. "A Hybrid Metaheuristic Approach to Solving the UBQP Problem". European Journal of Operational Research, 2010.
 */
template<class Neighbor >
class moRndIndexedVectorTabuList : public moIndexedVectorTabuList<Neighbor>
{
public:
  typedef typename Neighbor::EOT EOT;

  //tabu list
  using moIndexedVectorTabuList<Neighbor>::tabuList;
  //maximum size of the tabu list
  using moIndexedVectorTabuList<Neighbor>::maxSize;
  //how many iteration a move is tabu
  using moIndexedVectorTabuList<Neighbor>::howlong;

  /**
   * Constructor
   * @param _maxSize maximum size of the tabu list
   * @param _howlong how many minimal iteration a move is tabu
   * @param _howlongRnd how many additional iterations a move is tabu (random between [0 , _howlongRnd [ )
   */
  moRndIndexedVectorTabuList(unsigned int _maxSize, unsigned int _howlong, unsigned int _howlongRnd) : moIndexedVectorTabuList<Neighbor>(_maxSize, _howlong), howlongRnd(_howlongRnd) {
  }

  /**
   * add a new neighbor in the tabuList
   * @param _sol unused solution
   * @param _neighbor the current neighbor
   */
  virtual void add(EOT & _sol, Neighbor & _neighbor) {
    if (_neighbor.index() < maxSize) 
      tabuList[_neighbor.index()] = howlong + rng.uniform(howlongRnd) ;
  }


protected:
  // the random part of the forbidden time
  unsigned int howlongRnd;
};

#endif
