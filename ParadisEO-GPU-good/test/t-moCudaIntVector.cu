/*
  <t-moCudaIntVector.cu>
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

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <cudaType/moCudaIntVector.h>
#include <eo>

typedef moCudaIntVector<eoMaximizingFitness> Solution;


int main() {


  std::cout << "[t-moCudaIntVector] => START" << std::endl;

  //verif constructor
  Solution sol;
  Solution sol1(5);
  Solution sol2(3);

  assert(sol.size()==1); 
  assert(sol1.size()==5); 
  assert(sol2.size()==3); 

  //verif discret vector
  for(int i=0;i<5;i++)
    assert((sol1[i]>=0)||(sol1[i]<5));
  
  //verif discret vector
  for(int i=0;i<3;i++)
    assert((sol2[i]>=0)||(sol2[i]<3));
    
  std::cout << "[t-moCudaIntVector] => OK" << std::endl;

  return EXIT_SUCCESS;
}

