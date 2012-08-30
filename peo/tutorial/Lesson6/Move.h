/*
  <Move.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille Nord Europe, 2006-2009
  (C) OPAC Team, LIFL, 2002-2009

  The Van LUONG,  (The-Van.Luong@inria.fr)
  Mahmoud FATENE, (mahmoud.fatene@inria.fr)

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

#ifndef MOVE_H
#define MOVE_H

#include <oldmo.h>


class ProblemEval : public eoEvalFunc <Problem> {
public:

  void operator() (Problem & _problem){
    //cout << "eoEvalFunc()"<< endl;
    _problem.fitness(_problem.evaluate());
    
  }

};

class Move : public moMove <Problem>, public std :: pair <unsigned, unsigned> {
public :
  void operator () (Problem & _problem) {
    // transformer une solution en une autre connaissant le meilleur mouvement
    //cout << "MOVE() " << endl;
    int temp = _problem.solution[first];
    _problem.solution[first] = _problem.solution[second];
    _problem.solution[second] = temp; 
  
    
}
  
};


class MoveInit : public moMoveInit <Move> {
public:

  void operator () (Move & _move, const Problem& _problem) {
    //cout << "MoveInit " << endl;
    
    _move.first = 0;
    _move.second = 1;

    
  }
};


class MoveNext : public moNextMove <Move>{
public:
  bool operator() (Move & _move, const Problem & _problem) {
    //cout << "MoveNext" << endl;
    //cout << _move.first << " , " << _move.second << endl;

    if (_move.first < n-2){
      if (_move.second < n-1){
	
	_move.second++;
      }
      else {
	_move.first++;
	_move.second = _move.first + 1;

      }
      
     
      return true;
	
    }

  
      return false;
      
  }
};

class MoveIncrEval : public moMoveIncrEval <Move>
{
public:

  int compute_delta(int n, int** a, int** b,
			     int* p, int i, int j)
    {
	int d; int k;
	d = (a[i][i]-a[j][j])*(b[p[j]][p[j]]-b[p[i]][p[i]]) +
	    (a[i][j]-a[j][i])*(b[p[j]][p[i]]-b[p[i]][p[j]]);
	for (k = 0; k < n; k = k + 1) if (k!=i && k!=j)
					  d = d + (a[k][i]-a[k][j])*(b[p[k]][p[j]]-b[p[k]][p[i]]) +
					      (a[i][k]-a[j][k])*(b[p[j]][p[k]]-b[p[i]][p[k]]);
	return(d);
    }


  eoMinimizingFitness operator() (const Move & _move, const Problem & _problem){
    
    double cost;
    cost=0;
 
    // for calculing delta difference
    int* p = new int[n];
    for (int i = 0 ; i < n ; i++)
      p[i] = _problem.solution[i];

 
    
    cost = _problem.fitness() + compute_delta(n,a,b,p,_move.first, _move.second);
    

    
    delete[] p;
    
    return cost;
  }

};




class MoveRand : public moRandMove <Move>{
public:

  void operator() (Move & _move){
    _move.first = rand()%n;
    do{
      _move.second = rand()%n;
      
    } while (_move.first == _move.second);
  }

};


//for ILS 


/*
class Perturbation : public eoMonOp <Problem>{
public:

  bool operator() (Problem & _problem){
    int r1,r2,temp;
    int mu = 1 + rand()%n;
    for (int k = 1 ; k <= mu; k++){ 
      
      r1 = rand()%n;
      
      for(;;){
	r2 = rand()%n;
	if (r1 != r2)
	  break;
      }
      
      temp = _problem.solution[r1];
      _problem.solution[r1] = _problem.solution[r2];
      _problem.solution[r2] = temp;
    
    }
    _problem.invalidate();
    
  }


};
*/
#endif
