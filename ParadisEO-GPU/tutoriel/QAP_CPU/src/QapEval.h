#ifndef __QapEval
#define __QapEval

template <class EOT>
class QapEval : public eoEvalFunc<EOT>
{

 public:

  QapEval(){}

  ~QapEval(){}

  void operator() (EOT & _sol) {
    int cost=0;
    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
	cost += a[i*n+j] * b[_sol[i]*n+_sol[j]]; 
    _sol.fitness(cost);
  }
    
};

#endif
