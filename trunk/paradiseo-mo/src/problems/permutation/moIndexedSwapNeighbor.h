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

#include <neighborhood/moBackableNeighbor.h>
#include <neighborhood/moIndexNeighbor.h>

/**
 * Indexed Swap Neighbor: the position of the swap are computed according to the index
 */
template <class EOT, class Fitness=typename EOT::Fitness>
class moIndexedSwapNeighbor: public moBackableNeighbor<EOT, Fitness>, public moIndexNeighbor<EOT>
{
public:
  //  using moIndexNeighbor<EOT>::EOT;

  using moBackableNeighbor<EOT>::fitness;
  using moIndexNeighbor<EOT>::key;
  
  /**
   * Apply the swap
   * @param _solution the solution to move
   */
  virtual void move(EOT& _solution) {
    unsigned int tmp;
    unsigned i, j;

    this->getIndices(_solution.size(), i, j);

    tmp          = _solution[i];
    _solution[i] = _solution[j];
    _solution[j] = tmp;

    _solution.invalidate();
  }
  
  /**
   * apply the swap to restore the solution (use by moFullEvalByModif)
   * @param _solution the solution to move back
   */
  virtual void moveBack(EOT& _solution) {
    unsigned int tmp;
    unsigned i, j;
    this->getIndices(_solution.size(), i, j);

    tmp          = _solution[i];
    _solution[i] = _solution[j];
    _solution[j] = tmp;

    _solution.invalidate();
  }
  
  /**
   * Get the two indexes of the swap
   * @param N size of the permutation
   * @param _first first index
   * @param _second second index
   */
  void getIndices(unsigned N, unsigned int & _first, unsigned int & _second) {
    unsigned int n = (unsigned int) ( (1 + sqrt(1 + 8 * key)) / 2);

    _first  = key - (n - 1) * n / 2;
    _second = N - 1  - (n - 1 - _first);
  }
    

};

#endif
