//-----------------------------------------------------------------------------
// t-eoRingTopology.cpp
//-----------------------------------------------------------------------------


#include <eo>

typedef eoRealParticle < double >Indi;

//Evaluation function
double f (const Indi & _indi)
{
    double sum = 0;
    for (unsigned i = 0; i < _indi.size (); i++)
      sum += pow(_indi[i],2);
    return (-sum);
}

int main_function(int argc, char **argv)
{
	//Parameters
    const unsigned int VEC_SIZE = 2;
    const unsigned int POP_SIZE = 10;
    const unsigned int NEIGHBORHOOD_SIZE= 3;

    rng.reseed (33);
    eoEvalFuncPtr<Indi, double, const Indi& > plainEval(f);
    eoEvalFuncCounter < Indi > eval (plainEval);
    eoUniformGenerator < double >uGen (0., 5.);
    eoInitFixedLength < Indi > random (VEC_SIZE, uGen);
    eoUniformGenerator < double >sGen (-1., 1.);
    eoVelocityInitFixedLength < Indi > veloRandom (VEC_SIZE, sGen);
    eoFirstIsBestInit < Indi > localInit;
    eoPop < Indi > pop;
    pop.append (POP_SIZE, random);
    apply(eval, pop);
    apply < Indi > (veloRandom, pop);
    apply < Indi > (localInit, pop);
    eoRingTopology<Indi> topology(NEIGHBORHOOD_SIZE);
    topology.setup(pop);
    std::cout<<"\n\n\nPopulation :\n\n"<<pop;
    std::cout<<"\n\nNeighborhood :\n\n";
    topology.printOn();
    int k = NEIGHBORHOOD_SIZE/2;
    for(unsigned i=0;i<pop.size();i++)
    {
    	std::cout<<"\nBetween : ";
    	for(unsigned j=0;j<NEIGHBORHOOD_SIZE;j++)
    		std::cout<<"\n"<<pop[((pop.size()+i-k+j)%pop.size())];
    	std::cout<<"\nThe best is : \n"<<topology.best(i)<<"\n";
    }
    std::cout<<"\n\n";
	return 1;
}

int main(int argc, char **argv)
{
    try
    {
	main_function(argc, argv);
    }
    catch(std::exception& e)
    {
	std::cout << "Exception: " << e.what() << " in t-eoRingTopology" << std::endl;
    }

}
