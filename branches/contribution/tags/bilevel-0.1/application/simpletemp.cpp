/*
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2010
*
* Legillon Francois
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
//-----------------------------------------------------------------------------
//Implementation of CoBRA, a coevolutionary algorithm, to solve the single-objective bilevel MDVRP 
#include <eo>
#include <biVRPDistMat.h>
#include <VRP2Eval.h>
#include <VRP2.h>
#include <VRP2InitNN.h>
#include <eoSwapMutation.h>
#include <eoOrderXover.h>
#include <eoPop.h>
#include <eoReplacement.h>
#include <eoEasyEA.h>
#include <BEO.h>
#include <PLA.h>
#include <eoStochTournamentSelect.h>
#include <op/biVRPRBX.h>
#include <op/biVRPSBX.h>
#include <op/VRP2Repair.h>
#include <op/beoCoevoUpExchange.h>
#include <op/beoCoevoLowExchange.h>
#include <op/beoLeftieExchange.h>
#include <op/beoRightieExchange.h>
#include <op/beoBestExchange.h>
#include <op/beoRateContainerCoevoOp.h>
#include <beoDummyCoevoPop.h>
#include <eoSurviveAndDie.h>
#include <biVRP.h>
#include <biVRPLowEval.h>
#include <biVRPUpEval.h>
#include <beoBLAISArchivetemp.h>
#include <beoBLAISArchivetemp.cpp>
#include <beoEval.h>
#include <beoLowQuad.h>
#include <beoUpQuad.h>
#include <beoLowMut.h>
#include <beoUpMut.h>
#include <beoInit.h>
#include <PLARandInit.h>
#include <es/eoRealOp.h>
#include <biVRPInit.h>
#include <es/eoNormalMutation.h>
#include <PLAInitResp.h>
#include <biVRPPerf2Worth.h>
#include <beoLexContinue.h>
#include <beoCoevoAllPop.h>
#include <beoSelectCoevoPop.h>
#include <beoCoevoLinePop.h>
#include <beoSelectOneUp.h>
#include <beoSelectOneLow.h>
#include <selection/moeoDetTournamentSelect.h>
#include <comparator/moeoFitnessComparator.h>
#include <replacement/moeoGenerationalReplacement.h>
using namespace std;

int main(int argc, char* argv[])
{
	if (argc<3) {
		cout<< "donnez un nom d'instance et un numerio de seed"<<endl;
		return EXIT_SUCCESS;
	}
	eo::rng.reseed(atoi(argv[2]));
	string filename(argv[1]);
	biVRPDistMat mat;
	mat.load(filename);
	int popsize=100;
	
	moeoFitnessComparator<biVRP> comp;
	biVRPUpEval evalU(mat);
	biVRPLowEval evalL(mat);
	beoEval<biVRP> eval(evalL,evalU);
	eval.nbEval=0;
	moeoDetTournamentSelect<biVRP> selectCoevoL(comp);
	moeoDetTournamentSelect<biVRP> selectCoevoU(comp);

	/******
	 * ALGOU
	 ******/
	moeoDetTournamentSelect<biVRP> oneselectU(comp);
	eoSelectNumber<biVRP> selectU(oneselectU,popsize/3);
	eoSwapMutation<biVRP::U> mmutU(5);
	beoUpMut<biVRP> mutU(mmutU);

	biVRPSBX xover;
	biVRPRBX xover2;
	eoGenerationalReplacement<biVRP> replaceU;
	eoPeriodicContinue<biVRP> contalgoU(10);
	eoBinGenOp<biVRP> genop(xover);
	eoMonGenOp<biVRP> genop2(mutU);
	eoBinGenOp<biVRP> genop3(xover2);
	eoDetTournamentSelect<biVRP> selectalgoU;
	eoSequentialOp<biVRP> nouvelopU;
	eoGeneralBreeder<biVRP> breedalgoU(oneselectU,nouvelopU);
	nouvelopU.add(genop,1);
	nouvelopU.add(genop2,0.3);
	nouvelopU.add(genop3,0.5);
	eoEasyEA<biVRP> algoU(contalgoU,eval, breedalgoU, replaceU);

	/*******
	 * ALGOL
	 ******/

	eoDetTournamentSelect<biVRP> oneselectL;
	eoSelectNumber<biVRP> selectL(oneselectL,popsize/3);
	eoGenerationalReplacement<biVRP> replaceL;
	double sigma=0.5;
	double proba=0.1;
	eoNormalMutation<biVRP::L> mmutL(sigma,proba);
	eoRealUXover<biVRP::L> xoverMMM;
	beoLowQuad<biVRP> xoveralgoL(xoverMMM);
	beoLowMut<biVRP> mutL(mmutL);
	eoMonGenOp<biVRP> genopalgoU1(mutL);
	eoRankingSelect<biVRP> selectalgoL;
	eoSequentialOp<biVRP> opL;
	eoGeneralBreeder<biVRP> breedalgoL(selectalgoL,opL);
	opL.add(mutL,0.5);
	opL.add(xoveralgoL,1);
	eoPeriodicContinue<biVRP> contalgoL(10);
	PLARandInit initpla(mat);
	
	eoEasyEA<biVRP> algoL(contalgoL,eval, breedalgoL, replaceL);

	/******
	 * MAIN ALGO
	 *****/

	biVRPPerf2Worth perf2WorthU;
	biVRPPerf2Worth perf2WorthL;
	eoDetTournamentWorthSelect<biVRP> ranselectpoperU(perf2WorthU,2);
	eoDetTournamentWorthSelect<biVRP> ranselectpoperL(perf2WorthL,2);

	beoLexContinue<biVRP> contU(100,100);
	beoLexContinue<biVRP> contL(100,100);
	VRP2 grou;
	grou.init(mat);
	eoInitPermutation<VRP2> initpermut(grou.size());
	biVRPInit init(mat,initpermut);
	beoBestExchange<biVRP> coevo3;
	beoRateContainerCoevoOp<biVRP> coevo(coevo3,0.5);
	beoSelectOneUp<biVRP> su(ranselectpoperU);
	beoSelectOneLow<biVRP> sl(ranselectpoperL);
	eoSelectNumber<biVRP> selectPoperU(su,10);
	eoSelectNumber<biVRP> selectPoperL(sl,10);
	beoCoevoLinePop<biVRP> poper(coevo, 80);
	eoDeterministicSaDReplacement<biVRP> merge(0.8);
	
	beoBLAISArchivetemp<biVRP> blais(algoU,algoL,eval,poper,contL,contU,selectCoevoU,selectCoevoL,merge);
	eoPop<biVRP> pop(100,init);
	for (int i=0;i<100;i++) eval(pop[i]);
	blais(pop);
	std::cout<<"nbEval"<<eval.nbEval<<std::endl;
	
	return EXIT_SUCCESS;
}
