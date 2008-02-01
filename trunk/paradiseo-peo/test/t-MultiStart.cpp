// Test : multistart

#include <peo>

struct Algorithm 
{
	void operator()(double & _d) 
	{
		_d = _d * _d;	
	}
};

int main (int __argc, char * * __argv)
{

  peo :: init (__argc, __argv);
  if (getNodeRank()==1)
  	std::cout<<"\n\nTest : multistart\n\n";
  std::vector < double > v;
  if (getNodeRank()==1)
  	std::cout<<"\n\nBefore :";
  for(unsigned i = 0; i< 10; i++)
  {
  	v.push_back(i);
 	if (getNodeRank()==1)
  		std::cout<<"\n"<<v[i];
  }
  Algorithm algo;
  peoMultiStart < double > initParallel (algo);
  peoWrapper parallelAlgo (initParallel, v);
  initParallel.setOwner(parallelAlgo);
  peo :: run( );
  peo :: finalize( );
  if (getNodeRank()==1)
  {
  	std::cout<<"\n\nAfter :\n";
  	for(unsigned i = 0; i< 10; i++)
		std::cout<<v[i]<<"\n";
  }
}
