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
   _neighbor.getIndices(n,i,j);
    cost = _sol.fitness() +compute_delta(n,a,b,_sol,i,j);
    _neighbor.fitness(cost);
  }

  // specific to the QAP incremental evaluation (part of algorithmic)
  int compute_delta(int n, int* a, int* b,EOT & _sol,int i,int j)
  {
    int d; int k;
    d = (a[i*n+i]-a[j*n+j])*(b[_sol[j]*n+_sol[j]]-b[_sol[i]*n+_sol[i]]) +
      (a[i*n+j]-a[j*n+i])*(b[_sol[j]*n+_sol[i]]-b[_sol[i]*n+_sol[j]]);
    for (k = 0; k < n; k = k + 1) 
      if (k!=i && k!=j)
	d = d + (a[k*n+i]-a[k*n+j])*(b[_sol[k]*n+_sol[j]]-b[_sol[k]*n+_sol[i]]) +
	  (a[i*n+k]-a[j*n+k])*(b[_sol[j]*n+_sol[k]]-b[_sol[i]*n+_sol[k]]);
    return(d);
  }


}; 
#endif
