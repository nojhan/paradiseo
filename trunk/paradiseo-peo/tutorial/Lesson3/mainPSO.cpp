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

double f (const Indi & _indi)
{
  double sum;
  sum=_indi[1]-pow(_indi[0],2);
  sum=100*pow(sum,2);
  sum+=pow((1-_indi[0]),2);
  return (-sum);
}

int main (int __argc, char *__argv[])
{
  peo :: init( __argc, __argv );
  eoParser parser(__argc, __argv);
  unsigned int POP_SIZE = parser.createParam((unsigned int)(20), "popSize", "Population size",'P',"Param").value();
  unsigned int MAX_GEN = parser.createParam((unsigned int)(100), "maxGen", "Maximum number of generations",'G',"Param").value();
  unsigned int VEC_SIZE = parser.createParam((unsigned int)(2), "vecSize", "Vector size",'V',"Param").value();  
  double INIT_POSITION_MIN = parser.createParam(-2.0, "pMin", "Init position min",'N',"Param").value();
  double INIT_POSITION_MAX = parser.createParam(2.0, "pMax", "Init position max",'X',"Param").value();
  double INIT_VELOCITY_MIN = parser.createParam(-1.0, "vMin", "Init velocity min",'n',"Param").value();
  double INIT_VELOCITY_MAX = parser.createParam(1.0, "vMax", "Init velocity max",'x',"Param").value();
  double omega = parser.createParam(1.0, "weight", "Weight",'w',"Param").value();
  double C1 = parser.createParam(0.5, "c1", "C1",'1',"Param").value();
  double C2 = parser.createParam(2.0, "c2t", "C2",'2',"Param").value();
  unsigned int NEIGHBORHOOD_SIZE = parser.createParam((unsigned int)(6), "neighSize", "Neighborhood size",'H',"Param").value();
  unsigned int MIG_FREQ = parser.createParam((unsigned int)(10), "migFreq", "Migration frequency",'F',"Param").value();
  rng.reseed (time(0));

// Island model

  RingTopology topologyMig;

// First
  eoGenContinue < Indi > genContPara (MAX_GEN);
  eoCombinedContinue <Indi> continuatorPara (genContPara);
  eoCheckPoint<Indi> checkpoint(continuatorPara);
  peoEvalFunc<Indi, double, const Indi& > plainEval(f);
  peoPopEval< Indi > eval(plainEval);
  eoUniformGenerator < double >uGen (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > random (VEC_SIZE, uGen);
  eoUniformGenerator < double >sGen (INIT_VELOCITY_MIN, INIT_VELOCITY_MAX);
  eoVelocityInitFixedLength < Indi > veloRandom (VEC_SIZE, sGen);
  eoFirstIsBestInit < Indi > localInit;
  eoRealVectorBounds bndsFlight(VEC_SIZE,INIT_POSITION_MIN,INIT_POSITION_MAX);
  eoStandardFlight < Indi > flight(bndsFlight);
  eoPop < Indi > pop;
  pop.append (POP_SIZE, random);
  eoLinearTopology<Indi> topology(NEIGHBORHOOD_SIZE);
  eoRealVectorBounds bnds(VEC_SIZE,INIT_VELOCITY_MIN,INIT_VELOCITY_MAX);
  eoStandardVelocity < Indi > velocity (topology,omega,C1,C2,bnds);
  eoInitializer <Indi> init(eval,veloRandom,localInit,topology,pop);

// Island model

  eoPeriodicContinue< Indi > mig_cont( MIG_FREQ );
  peoPSOSelect<Indi> mig_selec(topology);
  peoWorstPositionReplacement<Indi> mig_replac;

// Specific implementation (peoData.h)

  eoContinuator<Indi> cont(mig_cont, pop);
  eoSelector <Indi, eoPop<Indi> > mig_select (mig_selec,1,pop);
  eoReplace <Indi, eoPop<Indi> > mig_replace (mig_replac,pop);


// Second

  eoGenContinue < Indi > genContPara2 (MAX_GEN);
  eoCombinedContinue <Indi> continuatorPara2 (genContPara2);
  eoCheckPoint<Indi> checkpoint2(continuatorPara2);
  peoEvalFunc<Indi, double, const Indi& > plainEval2(f);
  peoPopEval< Indi > eval2(plainEval2);
  eoUniformGenerator < double >uGen2 (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > random2 (VEC_SIZE, uGen2);
  eoUniformGenerator < double >sGen2 (INIT_VELOCITY_MIN, INIT_VELOCITY_MAX);
  eoVelocityInitFixedLength < Indi > veloRandom2 (VEC_SIZE, sGen2);
  eoFirstIsBestInit < Indi > localInit2;
  eoRealVectorBounds bndsFlight2(VEC_SIZE,INIT_POSITION_MIN,INIT_POSITION_MAX);
  eoStandardFlight < Indi > flight2(bndsFlight2);
  eoPop < Indi > pop2;
  pop2.append (POP_SIZE, random2);
  eoLinearTopology<Indi> topology2(NEIGHBORHOOD_SIZE);
  eoRealVectorBounds bnds2(VEC_SIZE,INIT_VELOCITY_MIN,INIT_VELOCITY_MAX);
  eoStandardVelocity < Indi > velocity2 (topology2,omega,C1,C2,bnds2);
  eoInitializer <Indi> init2(eval2,veloRandom2,localInit2,topology2,pop2);

// Island model

  eoPeriodicContinue< Indi > mig_cont2( MIG_FREQ );
  peoPSOSelect<Indi> mig_selec2(topology2);
  peoWorstPositionReplacement<Indi> mig_replac2;

// Specific implementation (peoData.h)

  eoContinuator<Indi> cont2(mig_cont2,pop2);
  eoSelector <Indi, eoPop<Indi> > mig_select2 (mig_selec2,1,pop2);
  eoReplace <Indi, eoPop<Indi> > mig_replace2 (mig_replac2,pop2);


// Asynchronous island

  peoAsyncIslandMig< eoPop<Indi>, eoPop<Indi> > mig(cont,mig_select, mig_replace, topologyMig);
  checkpoint.add( mig );
  peoAsyncIslandMig< eoPop<Indi>, eoPop<Indi> > mig2(cont2,mig_select2, mig_replace2, topologyMig);
  checkpoint2.add( mig2 );


// Parallel algorithm

  eoSyncEasyPSO <Indi> psa(init,checkpoint,eval, velocity, flight);
  peoWrapper parallelPSO( psa, pop);
  eval.setOwner(parallelPSO);
  mig.setOwner(parallelPSO);
  eoSyncEasyPSO <Indi> psa2(init2,checkpoint2,eval2, velocity2, flight2);
  peoWrapper parallelPSO2( psa2, pop2);
  eval2.setOwner(parallelPSO2);
  mig2.setOwner(parallelPSO2);
  peo :: run();
  peo :: finalize();
  if (getNodeRank()==1)
    {
      pop.sort();
      pop2.sort();
      std::cout << "Final population :\n" << pop << std::endl;
      std::cout << "Final population :\n" << pop2	 << std::endl;
    }
}
