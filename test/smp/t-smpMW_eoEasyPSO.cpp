//-----------------------------------------------------------------------------
// BinaryPSO.cpp
//-----------------------------------------------------------------------------
//*
// An instance of a VERY simple Real-coded binary Particle Swarm Optimization Algorithm
//
//-----------------------------------------------------------------------------
#include <stdexcept>
#include <iostream>
#include <sstream>

#include <paradiseo/eo.h>
#include <paradiseo/smp.h>

// Use functions from namespace std
using namespace std;
using namespace paradiseo::smp;

typedef eoMinimizingFitness FitT;
typedef eoBitParticle < FitT > Particle;

double binary_value (const Particle & _particle)
{
    double sum = 0;
    for (unsigned i = 0; i < _particle.size(); i++)
	sum +=_particle[i];
    return (sum);
}

int main(void)
{
 	const unsigned int SEED = 42;	// seed for random number generator

 	const unsigned int MAX_GEN=500;
    const unsigned int VEC_SIZE = 10;
    const unsigned int POP_SIZE = 20;
    const unsigned int NEIGHBORHOOD_SIZE= 3;

    const double VELOCITY_INIT_MIN= -1;
    const double VELOCITY_INIT_MAX= 1;

    const double VELOCITY_MIN= -1.5;
    const double VELOCITY_MAX= 1.5;

    const double INERTIA= 1;
    const double LEARNING_FACTOR1= 1.7;
    const double LEARNING_FACTOR2= 2.3;

  	rng.reseed(SEED);

    eoPop<Particle> pop;

    eoEvalFuncPtr<Particle, double, const Particle& > eval(binary_value);
    eoRingTopology<Particle> topology(NEIGHBORHOOD_SIZE);

    eoUniformGenerator<bool> uGen;
    eoInitFixedLength < Particle > random (VEC_SIZE, uGen);
	pop.append (POP_SIZE, random);

    eoUniformGenerator < double >sGen (VELOCITY_INIT_MIN, VELOCITY_INIT_MAX);
    eoVelocityInitFixedLength < Particle > veloRandom (VEC_SIZE, sGen);


    eoFirstIsBestInit < Particle > localInit;

    eoInitializer <Particle> fullInit(eval,veloRandom,localInit,topology,pop);

    fullInit();

  	pop.sort();

    cout << "INITIAL POPULATION:" << endl;
    for (unsigned i = 0; i < pop.size(); ++i)
	 cout <<  "\t best fit=" <<  pop[i] <<  endl;

    eoRealVectorBounds bnds(VEC_SIZE,VELOCITY_MIN,VELOCITY_MAX);

    eoStandardVelocity <Particle> velocity (topology,INERTIA,LEARNING_FACTOR1,LEARNING_FACTOR2,bnds);

    eoSigBinaryFlight <Particle> flight;

    eoGenContinue <Particle> genCont (MAX_GEN);
    
    try
    {
        MWModel<eoEasyPSO, Particle> pso(genCont, eval, velocity, flight);

        pso(pop);

        pop.sort();
	    cout << "FINAL POPULATION:" << endl;
      	for (unsigned i = 0; i < pop.size(); ++i)
      	{
	        cout <<  "\t best fit=" <<  pop[i] <<  endl;
	    }
	}
    catch(exception& e)
    {
        cout << "Exception: " << e.what() << '\n';
    }

    return 0;
}
//-----------------------------------------------------------------------------
