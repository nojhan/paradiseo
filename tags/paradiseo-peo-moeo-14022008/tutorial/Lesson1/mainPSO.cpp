/*
* <main.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, INRIA, 2007
*
* Clive Canape
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#include <peo>

typedef eoRealParticle < double >Indi;

//Evaluation function

double f (const Indi & _indi)
{
  // Rosenbrock function f(x) = 100*(x[1]-x[0]^2)^2+(1-x[0])^2
  // => optimal : f* = 0 , with x* =(1,1)

  double sum;
  sum=_indi[1]-pow(_indi[0],2);
  sum=100*pow(sum,2);
  sum+=pow((1-_indi[0]),2);
  return (-sum);
}

int main (int __argc, char *__argv[])
{


// Initialization of the parallel environment : thanks this instruction, ParadisEO-PEO can initialize himself
  peo :: init( __argc, __argv );

//Parameters

  const unsigned int VEC_SIZE = 2;  // Don't change this parameter when you are resolving the Rosenbrock function

  const unsigned int POP_SIZE = 20; // As with a sequential algorithm, you change the size of the population

  const unsigned int NEIGHBORHOOD_SIZE= 6; // This parameter define the neighborhoods in the PSO's topology

  const unsigned int MAX_GEN = 150; // Define the number of maximal generation

  const double INIT_POSITION_MIN = -2.0;  // For initialize x
  const double INIT_POSITION_MAX = 2.0;   // In the case of the Rosenbrock function : -2 < x[i] < 2
  const double INIT_VELOCITY_MIN = -1.;
  const double INIT_VELOCITY_MAX = 1.;
  const double weight = 1;
  const double C1 = 0.5;
  const double C2 = 2.;
  rng.reseed (time(0));

// Stopping
  eoGenContinue < Indi > genContPara (MAX_GEN);
  eoCombinedContinue <Indi> continuatorPara (genContPara);
  eoCheckPoint<Indi> checkpoint(continuatorPara);


// For a parallel evaluation
  peoEvalFunc<Indi, double, const Indi& > plainEval(f);
  peoPopEval< Indi > eval(plainEval);

// Initialization
  eoUniformGenerator < double >uGen (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > random (VEC_SIZE, uGen);

// Velocity
  eoUniformGenerator < double >sGen (INIT_VELOCITY_MIN, INIT_VELOCITY_MAX);
  eoVelocityInitFixedLength < Indi > veloRandom (VEC_SIZE, sGen);

// Initializing the best
  eoFirstIsBestInit < Indi > localInit;

// Flight
  eoRealVectorBounds bndsFlight(VEC_SIZE,INIT_POSITION_MIN,INIT_POSITION_MAX);
  eoStandardFlight < Indi > flight(bndsFlight);

// Creation of the population
  eoPop < Indi > pop;
  pop.append (POP_SIZE, random);

// Topology (ie Lesson 6 of ParadisEO-PEO)
  eoLinearTopology<Indi> topology(NEIGHBORHOOD_SIZE);
  eoRealVectorBounds bnds(VEC_SIZE,INIT_VELOCITY_MIN,INIT_VELOCITY_MAX);
  eoStandardVelocity < Indi > velocity (topology,weight,C1,C2,bnds);

// Initialization
  eoInitializer <Indi> init(eval,veloRandom,localInit,topology,pop);

//Parallel algorithm
  eoSyncEasyPSO <Indi> psa(init,checkpoint,eval, velocity, flight);
  peoWrapper parallelPSO( psa, pop);
  eval.setOwner(parallelPSO);

  peo :: run();
  peo :: finalize();
  if (getNodeRank()==1)
    {
      pop.sort();
      std::cout << "Final population :\n" << pop << std::endl;
    }
}
