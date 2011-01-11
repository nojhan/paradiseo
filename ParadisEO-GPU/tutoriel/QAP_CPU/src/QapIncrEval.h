#ifndef __QapIncrEval
#define __QapIncrEval

#include <eval/moEval.h>
template <class Neighbor>
class QapIncrEval : public moEval<Neighbor>{

 public:

  typedef typename moEval<Neighbor>::EOT EOT;
  typedef typename moEval<Neighbor>::Fitness Fitness;
 

  QapIncrEval(){}

  

  ~QapIncrEval(){}

  void operator() (EOT & _sol, Neighbor & _neighbor){

    int cost=0;
    unsigned i,j;
    int neighborhoodsize = (n * (n - 1)) / 2;
   _neighbor.getIndices(i,j);
    cost = _sol.fitness() +compute_delta(n,a,b,_sol,i,j);
    _neighbor.fitness(cost);
  }

  // specific to the QAP incremental evaluation (part of algorithmic)
  int compute_delta(int n, int** a, int** b,EOT & _sol,int i,int j)
  {
    int d; int k;
    d = (a[i][i]-a[j][j])*(b[_sol[j]][_sol[j]]-b[_sol[i]][_sol[i]]) +
      (a[i][j]-a[j][i])*(b[_sol[j]][_sol[i]]-b[_sol[i]][_sol[j]]);
    for (k = 0; k < n; k = k + 1) 
      if (k!=i && k!=j)
	d = d + (a[k][i]-a[k][j])*(b[_sol[k]][_sol[j]]-b[_sol[k]][_sol[i]]) +
	  (a[i][k]-a[j][k])*(b[_sol[j]][_sol[k]]-b[_sol[i]][_sol[k]]);
    return(d);
  }
}; 
#endif
