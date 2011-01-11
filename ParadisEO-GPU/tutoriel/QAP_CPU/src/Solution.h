#ifndef _SOLUTION_H_
#define _SOLUTION_H_

#include <eo> // PARADISEO

class Solution:public EO <eoMinimizingFitness>{ // PARADISEO

 public:

  int * solution;

  Solution(int _taille){
    solution=new int[_taille];
  }

  ~Solution(){
    delete[] solution;
  }
};

#endif
