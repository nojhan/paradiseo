/*
  <PPPIncrEval.h>
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

#ifndef __PPPIncrEval_H
#define __PPPIncrEval_H

#include <eval/moCudaEvalFunc.h>

/**
 * Incremental Evaluation of PPP
 */

template<class Neighbor>
class PPPIncrEval: public moCudaEvalFunc<Neighbor> {

 public:

  typedef typename Neighbor::EOT EOT;
  typedef typename EOT::Fitness Fitness;

  /**
   * Constructor
   */

  PPPIncrEval() {
  }

  /**
   * Destructor
   */

  ~PPPIncrEval() {
  }

  /**
   * functor of  incremental evaluation of the solution(function inline can be called from host or device)
   * @param _sol the solution to evaluate
   * @param _fitness the fitness of the current solution
   * @param _index the set of information to compute fitness neighbor
   */

  inline __host__ __device__ Fitness operator() (EOT & _sol,Fitness _fitness, unsigned int *_index) {

   int H[Nd+1];
   int S[Md];
   int tmp_1=0;
   int tmp_2=0;
        
    for (int i=0; i<Md; i++){
      S[i]=0;
      for (int j=0; j<Nd; j++){
          if(_sol[j])
            S[i]=S[i]+dev_a[i*Nd+j];
          else
            S[i]=S[i]-dev_a[i*Nd+j];
      }

      tmp_1=tmp_1+abs(S[i])-S[i];  
      if(S[i]>=0)
	H[S[i]]=H[S[i]]+1;
    }

    for (int j=0; j<=Nd; j++){
      tmp_2=tmp_2+abs(dev_h[j]-H[j]);
    } 

    return ca*tmp_1+cb*tmp_2;
  
  }  

};

#endif

