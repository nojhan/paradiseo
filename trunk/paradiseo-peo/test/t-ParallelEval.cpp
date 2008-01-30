// Test : parallel evaluation
#include <peo>
#include <es.h>
typedef eoReal<double> Indi;
double f (const Indi & _indi)
{
  double sum=_indi[0]+_indi[1];
  return (-sum);
}
struct Algorithm 
{
	Algorithm( peoPopEval < Indi > & _eval): eval( _eval ){}
	void operator()(eoPop < Indi > & _pop) 
	{
		eoPop < Indi > empty_pop;
        eval(empty_pop, _pop);	
	}
	peoPopEval < Indi > & eval;
};

int main (int __argc, char *__argv[])
{
  peo :: init( __argc, __argv );
  if (getNodeRank()==1)
  	std::cout<<"\n\nTest : parallel evaluation\n\n";
  rng.reseed (10);
  peoEvalFunc<Indi, double, const Indi& > plainEval(f);
  peoPopEval< Indi > eval(plainEval);
  eoUniformGenerator < double >uGen (0, 1);
  eoInitFixedLength < Indi > random (2, uGen);
  eoPop < Indi > pop(20, random);
  Algorithm algo ( eval );
  peoWrapper parallelAlgo( algo, pop);
  eval.setOwner(parallelAlgo);
  peo :: run();
  peo :: finalize();
  if (getNodeRank()==1)
  {
      pop.sort();
      std::cout<<pop;
  }
}
