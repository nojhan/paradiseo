/*
 * Copyright (C) DOLPHIN Project-Team, INRIA Lille Nord-Europe, 2007-2008
 * (C) OPAC Team, LIFL, 2002-2008
 *
 * (c) Mostepha Redouane Khouadjia <mr.khouadjia@ed.univ-lille1.fr>, 2008
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

// Miscellaneous includes and declarations


#include<iostream>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <eo>
#include <peo>

#include"eoPsoDVRP.h"

#include"eoEventScheduler.h"

#include "eoPsoDVRPEvalFunc.h"

#include "eoPsoDVRPInit.h"

#include "eoPeoEasyPsoDVRP.h"

#include "eoPsoDVRPflight.h"

#include "eoPsoDVRPutils.h"

#include "eoParticleDvrp_mesg.h"

using namespace std;



double f (const eoParticleDVRP& _po) {

	return( computTourLength(_po.pRoutes));

     }


int main(int __argc, char *__argv[])
{

peo :: init( __argc, __argv );
eoParser parser(__argc, __argv);
unsigned int POP_SIZE = parser.createParam((unsigned int)(1), "popSize", "Population size",'P',"Param").value();
unsigned int MAX_GEN = parser.createParam((unsigned int)(100), "maxGen", "Maximum number of generations",'G',"Param").value();
string BENCHMARK = parser.createParam(string(),"fileOut", "Benchmark",'O',"Param").value();
double INERTIA = parser.createParam(1.0, "weight", "Weight",'w',"Param").value();
double LEARNING_FACTOR1 = parser.createParam(1.0, "c1", "C1",'1',"Param").value();
double LEARNING_FACTOR2 = parser.createParam(1.0, "c2", "C2",'2',"Param").value();
unsigned int MIG_FREQ = parser.createParam((unsigned int)(4), "migFreq", "Migration frequency",'F',"Param").value();
unsigned int MIG_SIZE = parser.createParam((unsigned int)(1), "migSize", "Migration size",'M',"Param").value();
unsigned int NBR_SLICES = parser.createParam((unsigned int) (25), "slices", "Nbr Slices",'N',"Param").value();
double TIME_CUT_OFF=parser.createParam(0.5, "cutOff", "Time of Cut Off",'T',"Param").value();
unsigned int SEED = parser.createParam((unsigned int)(1), "Seed", "Seed of Rng",'S',"Param").value();
rng.reseed (SEED);


///Settings

eoInitialSettings  InitialSettings (BENCHMARK, TIME_CUT_OFF, NBR_SLICES);

/// Island model
RingTopology topologyMig;

/// First
///INITIALIZE BEST
eoDVRBestPositionInit<eoParticleDVRP> initBest1;
///INITIALIZATION OF PARTICLE
eoNothingInit <eoParticleDVRP> empty1 ;
///ENCODING
eoPsoDVRPEncodeSwarm <eoParticleDVRP> encoding1;
/// MOVEMENT
eoPsoDVRPMove<eoParticleDVRP> move1;
/// SWARM
eoPop<eoParticleDVRP> swarm1;
/// TOPOLOGY
eoDVRPStarTopology<eoParticleDVRP>topology1;
/// VELOCITY
eoPsoDVRPvelocity<eoParticleDVRP> velocity1(topology1,INERTIA,LEARNING_FACTOR1, LEARNING_FACTOR2);
/// STOPPING CRITERIA
eoGenContinue<eoParticleDVRP> genContinuator1(MAX_GEN);
eoGenContinue<eoParticleDVRP> continuator1(NBR_SLICES);
eoCombinedContinue <eoParticleDVRP> continuatorPara1(continuator1);
eoCheckPoint<eoParticleDVRP>checkpoint1(continuatorPara1);
/// EVALUATION
peoEvalFunc<eoParticleDVRP, double, const eoParticleDVRP& > evalParticle1(f);
peoPopEval<eoParticleDVRP> eval1(evalParticle1);
eoPsoDVRPEvalFunc<eoParticleDVRP> evalDvrp1;
//peoPopEval<eoParticleDVRP>  eval1(evalDvrp1);


///THE MAIN CODE
swarm1.append(POP_SIZE,empty1);
/// Island model
eoPeriodicContinue<eoParticleDVRP>mig_cont1( MIG_FREQ );
peoPSOSelect<eoParticleDVRP> mig_selec1(topology1);
peoWorstPositionReplacement<eoParticleDVRP> mig_replac1;

/// Specific implementation (peoData.h)
eoContinuator<eoParticleDVRP>cont1(mig_cont1, swarm1);
eoSelector <eoParticleDVRP, eoPop<eoParticleDVRP> > mig_select1 (mig_selec1,MIG_SIZE,swarm1);
eoReplace <eoParticleDVRP, eoPop<eoParticleDVRP> > mig_replace1 (mig_replac1,swarm1);

///Second
///INITIALIZATION OF PARTICLE
eoNothingInit <eoParticleDVRP> empty2 ;
eoDVRBestPositionInit<eoParticleDVRP> initBest2;
///ENCODING
eoPsoDVRPEncodeSwarm <eoParticleDVRP> encoding2;
/// MOVEMENT
eoPsoDVRPMove<eoParticleDVRP> move2;
/// SWARM
eoPop<eoParticleDVRP> swarm2;
/// TOPOLOGY
eoDVRPStarTopology<eoParticleDVRP>topology2;
/// VELOCITY
eoPsoDVRPvelocity<eoParticleDVRP> velocity2(topology2,INERTIA,LEARNING_FACTOR1, LEARNING_FACTOR2);

eoGenContinue<eoParticleDVRP> genContinuator2(MAX_GEN);
eoGenContinue<eoParticleDVRP> continuator2(NBR_SLICES);
eoCombinedContinue <eoParticleDVRP> continuatorPara2(continuator2);
eoCheckPoint<eoParticleDVRP>checkpoint2(continuatorPara2);
/// EVALUATION
peoEvalFunc<eoParticleDVRP, double, const eoParticleDVRP& > evalParticle2(f);
peoPopEval<eoParticleDVRP> eval2(evalParticle2);
eoPsoDVRPEvalFunc<eoParticleDVRP> evalDvrp2;
//peoPopEval<eoParticleDVRP>  eval2(evalDvrp2);

swarm2.append(POP_SIZE,empty2);

/// Island model
eoPeriodicContinue<eoParticleDVRP>mig_cont2( MIG_FREQ );
peoPSOSelect<eoParticleDVRP> mig_selec2(topology2);
peoWorstPositionReplacement<eoParticleDVRP> mig_replac2;


/// Specific implementation (peoData.h)
eoContinuator<eoParticleDVRP>cont2(mig_cont2, swarm2);
eoSelector <eoParticleDVRP, eoPop<eoParticleDVRP> > mig_select2 (mig_selec2,MIG_SIZE,swarm2);
eoReplace <eoParticleDVRP, eoPop<eoParticleDVRP> > mig_replace2 (mig_replac2,swarm2);
/*
/// Asynchronous island
peoAsyncIslandMig< eoPop<eoParticleDVRP>, eoPop<eoParticleDVRP> > mig1(cont1,mig_select1, mig_replace1, topologyMig);
checkpoint1.add( mig1);
peoAsyncIslandMig< eoPop<eoParticleDVRP>, eoPop<eoParticleDVRP> > mig2(cont2,mig_select2, mig_replace2, topologyMig);
checkpoint2.add( mig2 );
*/

 //Synchronus island
peoSyncIslandMig<eoPop<eoParticleDVRP>, eoPop<eoParticleDVRP> > mig1(MIG_FREQ,mig_select1,mig_replace1,topologyMig);
checkpoint1.add(mig1);
peoSyncIslandMig<eoPop<eoParticleDVRP>, eoPop<eoParticleDVRP> > mig2(MIG_FREQ,mig_select2,mig_replace2,topologyMig);
checkpoint2.add(mig2);



/// Parallel algorithm
eoPeoEasyPsoDVRP<eoParticleDVRP> algo1(encoding1,evalDvrp1,eval1,velocity1,move1,checkpoint1,genContinuator1,initBest1);
peoWrapper parallelPSO1( algo1, swarm1);
eval1.setOwner(parallelPSO1);
mig1.setOwner(parallelPSO1);

eoPeoEasyPsoDVRP<eoParticleDVRP> algo2(encoding2,evalDvrp2,eval2,velocity2,move2,checkpoint2,genContinuator2,initBest2);
peoWrapper parallelPSO2( algo2, swarm2);
eval2.setOwner(parallelPSO2);
mig2.setOwner(parallelPSO2);


peo :: run();
peo :: finalize();

if (getNodeRank()==1)
{

   printBestParticle(topology1,SEED, cout);

  printBestParticle(topology2,SEED, cout);

 }

return EXIT_SUCCESS;

}

