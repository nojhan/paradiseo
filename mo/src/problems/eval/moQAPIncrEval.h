/*
<moQAPIncrEval.h>
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

#ifndef _moQAPIncrEval_H
#define _moQAPIncrEval_H

#include "../../eval/moEval.h"
#include "../../../problems/eval/qapEval.h"

/**
 * Incremental evaluation Function for the QAP problem
 *
 * ElemType is the type of elements in the matrix. This type must be signed and not unsigned.
 */
template< class Neighbor, typename ElemType = long int >
class moQAPIncrEval : public moEval<Neighbor>
{
public:
  typedef typename Neighbor::EOT EOT;
  
  /*
   * default constructor
   * @param _qapEval full evaluation of the QAP problem
   */
  moQAPIncrEval(QAPeval<EOT, ElemType> & _qapEval) {
    n = _qapEval.getNbVar();
    A = _qapEval.getA();
    B = _qapEval.getB();
  }

  /*
   * incremental evaluation of the neighbor for the QAP problem
   * @param _solution the solution to move (permutation)
   * @param _neighbor the neighbor to consider (of type moSwapNeigbor)
   */
  virtual void operator()(EOT & _solution, Neighbor & _neighbor) {
    ElemType d;  
    int k;
    
    unsigned i = _neighbor.first();
    unsigned j = _neighbor.second();

    //    _neighbor.getIndices(n, i, j);
    d = (A[i][i]-A[j][j])*(B[_solution[j]][_solution[j]]-B[_solution[i]][_solution[i]]) +
      (A[i][j]-A[j][i])*(B[_solution[j]][_solution[i]]-B[_solution[i]][_solution[j]]);
    
    for (k = 0; k < n; k++) 
      if (k != i && k != j)
	d = d + (A[k][i]-A[k][j])*(B[_solution[k]][_solution[j]]-B[_solution[k]][_solution[i]]) +
	  (A[i][k]-A[j][k])*(B[_solution[j]][_solution[k]]-B[_solution[i]][_solution[k]]);
    
    _neighbor.fitness(_solution.fitness() + d);
  }

private:
  // number of variables
  int n;
  
  // matrix A
  ElemType ** A;
  
  // matrix B
  ElemType ** B;
  
};

#endif

