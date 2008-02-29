/*
* <mainPSO.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, INRIA, 2008
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
  eoParser parser(__argc, __argv);
  unsigned int POP_SIZE = parser.createParam((unsigned int)(20), "popSize", "Population size",'P',"Param").value();
  unsigned int MAX_GEN = parser.createParam((unsigned int)(100), "maxGen", "Maximum number of generations",'G',"Param").value();
  unsigned int VEC_SIZE = parser.createParam((unsigned int)(2), "vecSize", "Vector size",'V',"Param").value();  
  double INIT_POSITION_MIN = parser.createParam(-2.0, "pMin", "Init position min",'N',"Param").value();
  double INIT_POSITION_MAX = parser.createParam(2.0, "pMax", "Init position max",'X',"Param").value();
  double INIT_VELOCITY_MIN = parser.createParam(-1.0, "vMin", "Init velocity min",'n',"Param").value();
  double INIT_VELOCITY_MAX = parser.createParam(1.0, "vMax", "Init velocity max",'x',"Param").value();
  double weight = parser.createParam(1.0, "weight", "Weight",'w',"Param").value();
  double C1 = parser.createParam(0.5, "c1", "C1",'1',"Param").value();
  double C2 = parser.createParam(2.0, "c2t", "C2",'2',"Param").value();
  unsigned int NEIGHBORHOOD_SIZE = parser.createParam((unsigned int)(6), "neighSize", "Neighborhood size",'H',"Param").value();
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

// Topology
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
