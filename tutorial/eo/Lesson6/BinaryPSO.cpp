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

// Use functions from namespace std
using namespace std;

//-----------------------------------------------------------------------------
typedef eoMinimizingFitness FitT;
typedef eoBitParticle < FitT > Particle;
//-----------------------------------------------------------------------------


// EVALFUNC
//-----------------------------------------------------------------------------
// Just a simple function that takes binary value of a chromosome and sets
// the fitness
double binary_value (const Particle & _particle)
{
    double sum = 0;
    for (unsigned i = 0; i < _particle.size(); i++)
	sum +=_particle[i];
    return (sum);
}



void main_function(int argc, char **argv)
{
// PARAMETRES
 // all parameters are hard-coded!
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


//////////////////////////
//  RANDOM SEED
//////////////////////////
  //reproducible random seed: if you don't change SEED above,
  // you'll aways get the same result, NOT a random run
  	rng.reseed(SEED);


/// SWARM
 	// population <=> swarm
    eoPop<Particle> pop;

/// EVALUATION
	// Evaluation: from a plain C++ fn to an EvalFunc Object
    eoEvalFuncPtr<Particle, double, const Particle& > eval(  binary_value );


///////////////
/// TOPOLOGY
//////////////
	// ring topology
    eoRingTopology<Particle> topology(NEIGHBORHOOD_SIZE);


/////////////////////
// INITIALIZATION
////////////////////
    // position initialization
    eoUniformGenerator<bool> uGen;
    eoInitFixedLength < Particle > random (VEC_SIZE, uGen);
	pop.append (POP_SIZE, random);

    // velocities initialization component
    eoUniformGenerator < double >sGen (VELOCITY_INIT_MIN, VELOCITY_INIT_MAX);
    eoVelocityInitFixedLength < Particle > veloRandom (VEC_SIZE, sGen);

    // first best position initialization component
    eoFirstIsBestInit < Particle > localInit;

	// Create an eoInitialier that:
	// 		- performs a first evaluation of the particles
	//  	- initializes the velocities
	//  	- the first best positions of each particle
	// 		- setups the topology
    eoInitializer <Particle> fullInit(eval,veloRandom,localInit,topology,pop);

   // Full initialization here to be able to print the initial population
   // Else: give the "init" component in the eoEasyPSO constructor
   fullInit();

/////////////
// OUTPUT
////////////
// sort pop before printing it!
  	pop.sort();

  	// Print (sorted) the initial population (raw printout)
    cout << "INITIAL POPULATION:" << endl;
    for (unsigned i = 0; i < pop.size(); ++i)
	 cout <<  "\t best fit=" <<  pop[i] <<  endl;


///////////////
/// VELOCITY
//////////////
    // Create the bounds for the velocity not go to far away
    eoRealVectorBounds bnds(VEC_SIZE,VELOCITY_MIN,VELOCITY_MAX);

    // the velocity itself that needs the topology and a few constants
    eoStandardVelocity <Particle> velocity (topology,INERTIA,LEARNING_FACTOR1,LEARNING_FACTOR2,bnds);


///////////////
/// FLIGHT
//////////////
    // Binary flight based on sigmoid function
    eoSigBinaryFlight <Particle> flight;


////////////////////////
/// STOPPING CRITERIA
///////////////////////
    // the algo will run for MAX_GEN iterations
    eoGenContinue <Particle> genCont (MAX_GEN);


// GENERATION
  /////////////////////////////////////////
  // the algorithm
  ////////////////////////////////////////
  // standard PSO requires
  // stopping criteria, evaluation,velocity, flight

    eoEasyPSO<Particle> pso(genCont, eval, velocity, flight);

    // Apply the algo to the swarm - that's it!
    pso(pop);

// OUTPUT
  // Print (sorted) intial population
	 pop.sort();
	 cout << "FINAL POPULATION:" << endl;
  	 for (unsigned i = 0; i < pop.size(); ++i)
	cout <<  "\t best fit=" <<  pop[i] <<  endl;

}


// A main that catches the exceptions

int main(int argc, char **argv)
{
    try
    {
	main_function(argc, argv);
    }
    catch(exception& e)
    {
	cout << "Exception: " << e.what() << '\n';
    }

    return 1;
}
//-----------------------------------------------------------------------------
