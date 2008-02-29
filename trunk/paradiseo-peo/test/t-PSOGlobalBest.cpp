/*
* <t-PSOGlobalBest.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
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
* peoData to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/


// Test : PSO Global Best
#include <peo>
typedef eoRealParticle < double >Indi;
double f (const Indi & _indi)
{
  double sum=_indi[0]+_indi[1];
  return (sum);
}
int main (int __argc, char *__argv[])
{
  peo :: init( __argc, __argv );
  if (getNodeRank()==1)
  	std::cout<<"\n\nTest : PSO Global Best\n\n";
  rng.reseed (10);
  RingTopology topologyMig;
  eoGenContinue < Indi > genContPara (10);
  eoCombinedContinue <Indi> continuatorPara (genContPara);
  eoCheckPoint<Indi> checkpoint(continuatorPara);
  peoEvalFunc<Indi, double, const Indi& > plainEval(f);
  peoPopEval< Indi > eval(plainEval);
  eoUniformGenerator < double >uGen (0, 1.);
  eoInitFixedLength < Indi > random (2, uGen);
  eoUniformGenerator < double >sGen (-1., 1.);
  eoVelocityInitFixedLength < Indi > veloRandom (2, sGen);
  eoFirstIsBestInit < Indi > localInit;
  eoRealVectorBounds bndsFlight(2,0,1.);
  eoStandardFlight < Indi > flight(bndsFlight);
  eoPop < Indi > pop;
  pop.append (10, random);
  eoLinearTopology<Indi> topology(2);
  eoRealVectorBounds bnds(2,-1.,1.);
  eoStandardVelocity < Indi > velocity (topology,1,0.5,2.,bnds);
  eoInitializer <Indi> init(eval,veloRandom,localInit,topology,pop);
  eoPeriodicContinue< Indi > mig_cont( 2 );
  peoPSOSelect<Indi> mig_selec(topology);
  peoGlobalBestVelocity<Indi> mig_replac (2.,velocity);
  eoContinuator<Indi> cont(mig_cont, pop);
  eoSelector <Indi, eoPop<Indi> > mig_select (mig_selec,1,pop);
  eoReplace <Indi, eoPop<Indi> > mig_replace (mig_replac,pop);
  eoGenContinue < Indi > genContPara2 (10);
  eoCombinedContinue <Indi> continuatorPara2 (genContPara2);
  eoCheckPoint<Indi> checkpoint2(continuatorPara2);
  peoEvalFunc<Indi, double, const Indi& > plainEval2(f);
  peoPopEval< Indi > eval2(plainEval2);
  eoUniformGenerator < double >uGen2 (0, 1.);
  eoInitFixedLength < Indi > random2 (2, uGen2);
  eoUniformGenerator < double >sGen2 (-1., 1.);
  eoVelocityInitFixedLength < Indi > veloRandom2 (2, sGen2);
  eoFirstIsBestInit < Indi > localInit2;
  eoRealVectorBounds bndsFlight2(2,0,1.);
  eoStandardFlight < Indi > flight2(bndsFlight2);
  eoPop < Indi > pop2;
  pop2.append (10, random2);
  eoLinearTopology<Indi> topology2(2);
  eoRealVectorBounds bnds2(2,-1.,1.);
  eoStandardVelocity < Indi > velocity2 (topology2,1,0.5,2.,bnds2);
  eoInitializer <Indi> init2(eval2,veloRandom2,localInit2,topology2,pop2);
  eoPeriodicContinue< Indi > mig_cont2( 2 );
  peoPSOSelect<Indi> mig_selec2(topology2);
  peoGlobalBestVelocity<Indi> mig_replac2 (2.,velocity2);
  eoContinuator<Indi> cont2(mig_cont2,pop2);
  eoSelector <Indi, eoPop<Indi> > mig_select2 (mig_selec2,1,pop2);
  eoReplace <Indi, eoPop<Indi> > mig_replace2 (mig_replac2,pop2);
  peoAsyncIslandMig< eoPop<Indi>, eoPop<Indi> > mig(cont,mig_select, mig_replace, topologyMig);
  checkpoint.add( mig );
  peoAsyncIslandMig< eoPop<Indi>, eoPop<Indi> > mig2(cont2,mig_select2, mig_replace2, topologyMig);
  checkpoint2.add( mig2 );
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
