/*
  <t-moGPUBitNeighbor.cu>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

  Karima Boufaras,  Thé Van LUONG

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
#include <neighborhood/moGPUBitNeighbor.h>

typedef moGPUBitVector<eoMaximizingFitness> Solution;
typedef moGPUBitNeighbor<eoMaximizingFitness> Neighbor;


int main() {


  std::cout << "[t-moGPUBitNeighbor] => START" << std::endl;
    
    //init sol
    Solution sol(3);
    sol[0]=1;
    sol[1]=0;
    sol[2]=1;

    //test default Constructor
    Neighbor test1;
    assert(test1.index()==0);

    //test index setter 
    test1.index(6);
    //test index getter
    assert(test1.index()==6);
    
    //test fitness setter 
    test1.fitness(2);
    //test fitness getter
    assert(test1.fitness()==2);
    
    //test Cpy constructor
    Neighbor test2(test1);
    assert(test2.index()==6);
    assert(test2.fitness()==2);

    //test assignement operator
    test1.fitness(8);
    test1.index(2);
    test2=test1;
    assert(test2.fitness()==8);
    assert(test2.index()==2);

    //test move
    test2.move(sol);
    assert(!sol[2]);

    //test moveBack
    test2.moveBack(sol);
    assert(sol[2]);

    test1.printOn(std::cout);
    test2.printOn(std::cout);

    assert(test1.className()=="moGPUBitNeighbor");

  std::cout << "[t-moGPUBitNeighbor] => OK" << std::endl;

  return EXIT_SUCCESS;
}