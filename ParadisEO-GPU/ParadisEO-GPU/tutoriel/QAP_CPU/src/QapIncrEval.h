/*
  <QapIncrEval.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Karima Boufaras, Thé Van LUONG

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

#ifndef __QapIncrEval
#define __QapIncrEval

#include <eval/moEval.h>

/**
 * Incremental Evaluation of QAP
 */

template <class Neighbor>
class QapIncrEval : public moEval<Neighbor>{

 public:

  typedef typename moEval<Neighbor>::EOT EOT;
  typedef typename moEval<Neighbor>::Fitness Fitness;
 
  /**
   * Constructor
   */

  QapIncrEval(){}  

  /**
   * Destructor
   */

  ~QapIncrEval(){}

  /**
   * Functor for incremental evaluation of the solution
   * @param _sol the solution 
   * @param _neighbor the neighbor of solution to evaluate
   */

  void operator() (EOT & _sol, Neighbor & _neighbor){

    unsigned int cost=0;
    unsigned i,j;
    _neighbor.getIndices(n,i,j);
    cost = _sol.fitness() +compute_delta(_sol,i,j);
  }

  /**
   * Specific to the QAP incremental evaluation (part of algorithmic)
   * @param _sol the solution to evaluate
   * @param _i the first position of swap
   * @param _j the second position of swap
   */
 
  unsigned int compute_delta(EOT & _sol,unsigned i,unsigned j)
  {
    int d; 
    int k;
    
    d = (a[i*n+i]-a[j*n+j])*(b[_sol[j]*n+_sol[j]]-b[_sol[i]*n+_sol[i]]) +
      (a[i*n+j]-a[j*n+i])*(b[_sol[j]*n+_sol[i]]-b[_sol[i]*n+_sol[j]]);
    for (k = 0; k < n; k = k + 1) 
      if (k!=i && k!=j)
	d = d + (a[k*n+i]-a[k*n+j])*(b[_sol[k]*n+_sol[j]]-b[_sol[k]*n+_sol[i]]) +
	  (a[i*n+k]-a[j*n+k])*(b[_sol[j]*n+_sol[k]]-b[_sol[i]*n+_sol[k]]);
    return(d);
  }
}; 
#endif
