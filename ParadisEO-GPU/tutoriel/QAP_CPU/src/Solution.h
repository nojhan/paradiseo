#ifndef _SOLUTION_H_
#define _SOLUTION_H_

#include <eo>

class Solution:public EO <eoMinimizingFitness>{ 

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
