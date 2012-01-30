/*
  <t-moGPUIntVector.cu>
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
#include <GPUType/moGPUIntVector.h>
#include <eo>

typedef moGPUIntVector<eoMaximizingFitness> Solution;

int main() {


  std::cout << "[t-moGPUIntVector] => START" << std::endl;

  //test Default constructor
  Solution _sol;
  assert(_sol.size()==0);
  
  //test constructor
  Solution sol1(5);
  Solution sol2(3);
  
  //test copy constructor
  sol1.fitness(50);  
  Solution sol(sol1);
  assert(sol.size()==5);
  assert(sol.fitness()==50);
  for(int i=0;i<5;i++)
    assert(sol[i]==sol1[i]);
  
  //test Integer vector (create)& getter
  int tmp;
  for(int i=0;i<5;i++){
    tmp=sol1[i]%2;
    assert((tmp==0)||(tmp==1));
  }
  //test size getter
  assert(sol1.size()==5); 
  assert(sol2.size()==3); 
    
  //test size setter
  sol1.setSize(10); 
  assert(sol1.size()==10);
  for(int i=0;i<5;i++)
    assert(sol[i]==sol1[i]);
  for(int i=5;i<10;i++){
    tmp=sol1[i]%2;
    assert((tmp==0)||(tmp==1));
  }
  
  //test assignement operator
  sol2.fitness(20);
  sol1=sol2;
  assert(sol1.size()==3);
  assert(sol1.fitness()==20); 
  for(int i=0;i<3;i++)
    assert(sol1[i]==sol2[i]);

    
  std::cout << "[t-moGPUIntVector] => OK" << std::endl;

  return EXIT_SUCCESS;
}

