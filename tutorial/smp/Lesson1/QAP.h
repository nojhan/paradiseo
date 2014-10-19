/*
<QAP.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Alexandre Quemy, Thibault Lasnier - INSA Rouen

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

#ifndef QAP_H
#define QAP_H

#include <paradiseo/eo.h>

/* global variables */
extern int n; // size
extern int** a; // matrix A
extern int** b; // matrix B

class Problem : public EO<eoMinimizingFitness> { 

public:   
 
  int* solution; 

  Problem () {
    solution = new int[n];
    create();
  } 

  Problem (const Problem & _problem){ // copy constructor
    solution = new int[n];
    for (int i = 0; i < n ; i++){
      solution[i] = _problem.solution[i];
    }
    if (!_problem.invalid()) // if the solution has already been evaluated
      fitness(_problem.fitness()); // copy the fitness 
  }

  ~Problem(){ // destructor
    delete[] solution;
  }

  void operator= (const Problem & _problem){ // copy assignment operator
    for (int i = 0; i < n ; i++){
      solution[i] = _problem.solution[i];
    }
    fitness(_problem.fitness()); // copy the fitness 
  }  
  
  int& operator[] (unsigned i)
  {
  	return solution[i];	
  }

 
  void create(){ // create and initialize a solution
    int random, temp;
    for (int i=0; i< n; i++) 
      solution[i]=i;
    
    // we want a random permutation so we shuffle
    for (int i = 0; i < n; i++){
      random = rand()%(n-i) + i;
      temp = solution[i];
      solution[i] = solution[random];
      solution[random] = temp;
    }
  }

  int evaluate() { // evaluate the solution
    int cost=0;
    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
	cost += a[i][j] * b[solution[i]][solution[j]]; 
    
    return cost;
  }


  void printSolution() {
   for (int i = 0; i < n ; i++)
     std::cout << solution[i] << " " ;
 
   std::cout << std::endl;
  }
 
 
};
#endif
