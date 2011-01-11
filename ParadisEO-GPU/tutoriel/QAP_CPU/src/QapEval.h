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
      std::cout<<_sol[i]<<" ";
    std::cout<<std::endl;
    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
	cost += a[i][j] * b[_sol[i]][_sol[j]]; 
    _sol.fitness(cost);
  }
    
};

#endif
