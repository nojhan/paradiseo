// Test : parallel transform
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
	Algorithm( eoEvalFunc < Indi > & _eval, eoSelect < Indi > & _select, peoTransform < Indi > & _transform): 
		loopEval(_eval),
        eval(loopEval),
		selectTransform( _select, _transform),
		breed(selectTransform) {}
		
	void operator()(eoPop < Indi > & _pop) 
	{
		eoPop < Indi > offspring, empty_pop;
        eval(empty_pop, _pop);
        eval(empty_pop, offspring);
        std::cout<<"\n\nBefore :\n"<<offspring;
        breed(_pop, offspring);
        eval(empty_pop, offspring);
        std::cout<<"\n\nAfter :\n"<<offspring;
	}
	eoPopLoopEval < Indi > loopEval;
    eoPopEvalFunc < Indi > & eval;
	eoSelectTransform < Indi > selectTransform;
    eoBreed < Indi > & breed;
};

int main (int __argc, char *__argv[])
{
  peo :: init( __argc, __argv );
  if (getNodeRank()==1)
  	std::cout<<"\n\nTest : parallel transform\n\n";
  rng.reseed (10);
  eoEvalFuncPtr < Indi > plainEval(f);
  eoEvalFuncCounter < Indi > eval(plainEval);
  eoUniformGenerator < double >uGen (0, 1);
  eoInitFixedLength < Indi > random (2, uGen);
  eoPop < Indi > empty_pop,pop(6, random);
  eoRankingSelect < Indi > selectionStrategy;
  eoSelectNumber < Indi > select(selectionStrategy,6);
  eoSegmentCrossover < Indi > crossover;
  eoUniformMutation < Indi >  mutation(0.01);
  peoTransform<Indi> transform(crossover,0.8,mutation,0.3);
  Algorithm algo ( eval, select, transform );
  peoWrapper parallelAlgo( algo, pop);
  transform.setOwner(parallelAlgo);
  peo :: run();
  peo :: finalize();
}
