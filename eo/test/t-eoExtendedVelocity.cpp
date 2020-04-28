//-----------------------------------------------------------------------------
// t-eoExtendedVelocity.cpp
//-----------------------------------------------------------------------------


#include <eo>

typedef eoRealParticle < double > Particle;

//Evaluation function
double f (const Particle & _particle)
{
    double sum = 0;
    for (unsigned i = 0; i < _particle.size (); i++)
      sum += pow(_particle[i],2);
    return (-sum);
}

int main_function(int /*argc*/, char **/*argv*/)
{
	const unsigned POP_SIZE = 6, VEC_SIZE = 2, NEIGHBORHOOD_SIZE=2;

	// the population:
    eoPop<Particle> pop;

    // Evaluation
    eoEvalFuncPtr<Particle, double, const Particle& > eval(  f );

    // position + velocity + best init
    eoUniformGenerator < double >uGen (-3, 3);
    eoInitFixedLength < Particle > random (VEC_SIZE, uGen);
    eoUniformGenerator < double >sGen (-2, 2);
    eoVelocityInitFixedLength < Particle > veloRandom (VEC_SIZE, sGen);
    eoFirstIsBestInit < Particle > localInit;
    pop.append (POP_SIZE, random);

    // topology
    eoLinearTopology<Particle> topology(NEIGHBORHOOD_SIZE);
    eoInitializer <Particle> init(eval,veloRandom,localInit,topology,pop);
    init();

    // velocity
    eoExtendedVelocity <Particle> velocity (topology,1,1,1,1);

    // the test itself
    for (unsigned int i = 0; i < POP_SIZE; i++)
    {
      std::cout << " Initial particle n°" << i << " velocity: " <<  std::endl;
      for (unsigned int j = 0; j < VEC_SIZE; j++)
    		std::cout << " v" << j << "=" << pop[i].velocities[j] << std::endl;
    }

    for (unsigned int i = 0; i < POP_SIZE; i++)
   	 velocity (pop[i],i);

   	for (unsigned int i = 0; i < POP_SIZE; i++)
    {
      std::cout << " Final particle n°" << i << " velocity: " <<  std::endl;
      for (unsigned int j = 0; j < VEC_SIZE; j++)
    		std::cout << " v" << j << "=" << pop[i].velocities[j] << std::endl;
    }
   	return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    try
    {
	main_function(argc, argv);
    }
    catch(std::exception& e)
    {
	std::cout << "Exception: " << e.what() <<  " in t-eoExtendedVelocity" << std::endl;
    }
    return EXIT_SUCCESS;
}
