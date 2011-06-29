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
//Implementation of rationality calculator (weighted rationality)
#include <eo>
#include <moeo>
#include <biVRPDistMat.h>
#include <mbiVRPDistMat.h>
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
#include <op/beoBestExchange.h>
#include <eoSurviveAndDie.h>
#include <biVRP.h>
#include <biVRPLowEval.h>
#include <mbiVRPLowEval.h>
#include <biVRPUpEval.h>
#include <beoEval.h>
#include <beoLowQuad.h>
#include <beoUpQuad.h>
#include <beoLowMut.h>
#include <beoInit.h>
#include <PLARandInit.h>
#include <es/eoRealOp.h>
#include <biVRPInit.h>
#include <es/eoNormalMutation.h>
#include <PLAInitResp.h>
#include <beoLowComparator.h>
using namespace std;

int main(int argc, char* argv[])
{
	if (argc<3) {
		cout<< "donnez un nom d'instance et un fichier de solution a verifier"<<endl;
		return EXIT_SUCCESS;
	}


	mbiVRPDistMat mat;
	string filename(argv[1]);
	mat.load(filename);
	biVRPUpEval evalU(mat);
	mbiVRPLowEval evalL(mat);
	eoPop<biVRP> pop;
	beoEval<biVRP> eval(evalL,evalU);
	VRP2 vrp;
	vrp.init(mat);
	PLA pla;
	pla.init(mat);
	moeoParetoObjectiveVectorComparator<biVRP::L::ObjectiveVector> compL;
	beoLowComparator<biVRP::ObjectiveVector> comp(compL);
	moeoUnboundedArchive<biVRP> arc(comp);
	for (int x=2;x<argc;x++){
		string filenameSol(argv[x]);
		ifstream file(filenameSol.data());
		int vouvou=0;
		bool first=true;
		double bestscore;
		biVRP best;
		if (file.is_open())
			while (!file.eof()){
				std::vector<int> vrpvector;
				std::vector<double> plavector;
				for (int i=0;i<pla.size();i++){
					double top;
					file>>top;
					plavector.push_back(top);
				}
				for (int i=0;i<vrp.size();i++){
					int top;
					file>>top;
					vrpvector.push_back(top);
				}
				PLA neopla(plavector,mat);
				VRP2 neovrp(vrpvector,mat);
				biVRP bivrp(neovrp,neopla);
				eval(bivrp);
				arc(bivrp);
			}
	}
	/*******
	 * ALGOL
	 ******/

	eoEliteSequentialSelect<biVRP> oneselectL;
	eoSelectNumber<biVRP> selectL(oneselectL,10);
	eoDeterministicSaDReplacement<biVRP> replaceL(0.5);
	double sigma=0.5;
	double proba=0.1;
	eoNormalMutation<biVRP::L> mmutL(sigma,proba);
	eoRealUXover<biVRP::L> xoverMMM;
	beoLowQuad<biVRP> xoveralgoL(xoverMMM);
	beoLowMut<biVRP> mutL(mmutL);
	eoMonGenOp<biVRP> genopalgoU1(mutL);
	eoEliteSequentialSelect<biVRP> selectalgoL;
	eoSequentialOp<biVRP> opL;
	eoGeneralBreeder<biVRP> breedalgoL(selectalgoL,opL);
	opL.add(mutL,0.5);
	opL.add(xoveralgoL,1);
	eoPeriodicContinue<biVRP> contalgoL(500);
	PLARandInit initpla(mat);

	moeoNSGAII<biVRP> algoL(contalgoL,eval, opL);
	double eps=0.1;
	int res=0;
	for (int k=0; k<100 ; k++){
		pop=arc;
		algoL(pop);
		bool stop=false;
		for (int j=0;j<pop.size() && !stop;j++){
			for (int l=0;l<arc.size();l++){
				if (compL(arc[l].objectiveVector().low(),pop[j].objectiveVector().low())){
					res++;
					stop=true;
					break;
				}
			}
		}
	}
	std::cout<<res<<std::endl;
	return EXIT_SUCCESS;
}
	
