/*
<moUBQPSimpleIncrEval.h>
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

#ifndef _moUBQPSimpleIncrEval_H
#define _moUBQPSimpleIncrEval_H

#include "../eval/moEval.h"
#include "../../../problems/eval/ubqpEval.h"

/**
 * Incremental evaluation Function for the UBQPSimple problem
 */
template< class Neighbor >
class moUBQPSimpleIncrEval : public moEval<Neighbor>
{
public:
  typedef typename Neighbor::EOT EOT;
  
  /*
   * default constructor
   * @param _ubqpEval full evaluation of the UBQP problem
   */
  moUBQPSimpleIncrEval(UbqpEval<EOT> & _ubqpEval) {
    n = _ubqpEval.getNbVar();
    Q = _ubqpEval.getQ();
  }

  /*
   * Incremental evaluation of the neighbor for the UBQP problem (linear time complexity)
   * @param _solution the solution to move (bit string)
   * @param _neighbor the neighbor to consider (of type moBitNeighbor)
   */
  virtual void operator()(EOT & _solution, Neighbor & _neighbor) {
    unsigned int i = _neighbor.index();
    unsigned int j;

    int d = Q[i][i]; 

    for(j = 0; j < i; j++)
      if (_solution[j] == 1)
	d += Q[i][j];

    for(j = i+1; j < n; j++)
      if (_solution[j] == 1)
	d += Q[j][i];

    if (_solution[i] == 0)
      _neighbor.fitness(_solution.fitness() + d);
    else
      _neighbor.fitness(_solution.fitness() - d);

  }

  /*
   * to get the matrix Q
   *
   * @return matrix Q
   */
  int** getQ() {
    return Q;
  }

  /*
   * to get the number of variable (bit string length)
   *
   * @return bit string length
   */
  int getNbVar() {
    return n;
  }

private:
  // number of variables
  int n;
  
  // matrix Q (supposed to be in lower triangular form)
  int ** Q;
    
};

#endif

