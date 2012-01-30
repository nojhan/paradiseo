/*
  <t-moGPUBitVector.cu>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

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

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <GPUType/moGPUBitVector.h>
#include  <problems/eval/moGPUEvalOneMax.h>
#include <eo>

typedef moGPUBitVector<eoMaximizingFitness> Solution;


int main() {


  std::cout << "[t-moGPUBitVector] => START" << std::endl;

  moGPUEvalOneMax<Solution> eval;
  
  //test default constructor
  Solution _sol;
   
  //test constructor 
  Solution sol1(5);
  
  //test copy constructor
  sol1.fitness(10);  
  Solution sol(sol1);
  assert(sol.size()==5);
  assert(sol.fitness()==10);
  for(int i=0;i<5;i++)
    assert(sol[i]==sol1[i]);
   
  //test random vector of  bool
  for(int i=0;i<5;i++)
    assert((sol[i]==0)||(sol[i]==1));
    
  //test oneMax eval function
  eval(sol);
  eoMaximizingFitness sum=0;
  for(int i=0;i<5;i++)
    sum=sum+sol[i];
  assert(sol.fitness()==sum);
    
  //test size getter
  assert(_sol.size()==5);
  assert(sol1.size()==5);
  
  //test size setter
  sol1.setSize(10); 
  assert(sol1.size()==10);
  for(int i=0;i<5;i++)
    assert(sol[i]==sol1[i]);
  for(int i=5;i<10;i++)
    assert((sol1[i]==0)||(sol1[i]==1));


  //test constructor of constant vector
  Solution sol2(4,1);
  assert(sol2.size()==4);
  for(int i=0;i<4;i++)
    assert(sol2[i]==1);
  eval(sol2);
  assert(sol2.fitness()==4);

  //test accessor to the vector of bool
  sol2[3]=0; 
  assert(sol2[3]==0);
  eval(sol2);
  assert(sol2.fitness()==3);

  //test assignement operator
  sol2=sol;
  assert(sol.size()==sol2.size());
  assert(sol.fitness()==sol2.fitness());
  for(int i=0;i<5;i++)
    assert(sol[i]==sol2[i]);
     
  std::cout << "[t-moGPUBitVector] => OK" << std::endl;

  return EXIT_SUCCESS;
}

