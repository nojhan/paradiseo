/*
<QAPGA.h>
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

#ifndef _QAPGA_h
#define _QAPGA_h

extern int n; // size

class ProblemInit : public eoInit<Problem>
{
public:

  void operator()(Problem & _problem)
  {
    _problem.create();
  }
};

class ProblemEvalFunc : public eoEvalFunc<Problem>
{
public:

  void operator()(Problem & _problem)
  {
    _problem.fitness(_problem.evaluate());
      
  }
};

class ProblemXover : public eoQuadOp<Problem> {
public:

  /* The two parameters in input are the parents.
     The first parameter is also the output ie the child 
  */
  bool operator()(Problem & _problem1, Problem & _problem2)
  {
    int i;
    int random, temp;
    int unassigned_positions[n];
    int remaining_items[n];
    int j = 0;
				
    /* 1) find the items assigned in different positions for the 2 parents */
    for (i = 0 ; i < n ; i++){
      if (_problem1.solution[i] != _problem2.solution[i]){
	unassigned_positions[j] = i;
	remaining_items[j] = _problem1.solution[i];
	j++;
      }
    }
    
    /* 2) shuffle the remaining items to ensure that remaining items 
       will be assigned at random positions */
    for (i = 0; i < j; i++){
      random = rand()%(j-i) + i;
      temp = remaining_items[i];
      remaining_items[i] = remaining_items[random];
      remaining_items[random] = temp;
    }
						    					   
    /* 3) copy the shuffled remaining items at unassigned positions */
    for (i = 0; i < j ; i++)
      _problem1.solution[unassigned_positions[i]] = remaining_items[i];

    // crossover in our case is always possible
    return true; 
  }
};

class ProblemSwapMutation: public eoMonOp<Problem> {
public:
 
  bool operator()(Problem& _problem)  {
    int i,j;
    int temp;

    // generate two different indices
    i=rand()%n;
    do 
      j = rand()%n; 
    while (i == j);  
		   
    // swap
    temp = _problem.solution[i];
    _problem.solution[i] = _problem.solution[j];
    _problem.solution[j] = temp;

    return true;
    
  }
};




#endif
