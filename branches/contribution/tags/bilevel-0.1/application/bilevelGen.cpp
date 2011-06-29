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
//bilevel mdvrp instance generator with single-objective follower problem
#include <utils/eoRNG.h>
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
#include <eoSelectFromWorth.h>
#include <op/VRP2RBX.h>
#include <op/VRP2SBX.h>
using namespace std;

int main(int argc, char* argv[])

{
	if (argc<4) {
		cout<< "donnez un nom d'instance MDVRP, un nombre d'usine et la demande totale des retailers"<<endl;
		return EXIT_SUCCESS;
	}

	string filename(argv[1]);
	int numberOfPlant=atoi(argv[2]);
	double totalDemand=atof(argv[3]);
	int type,vehiculeNumber,retailerNumber,depotNumber;




	std::ifstream file(filename.data());
	if (file.is_open()){
		eoRng &rng=eo::rng;
		file>>type;
		file>>vehiculeNumber;
		file>>retailerNumber;
		file>>depotNumber;
		cout<<8<<" "<<vehiculeNumber<<" "<<retailerNumber<<" "<<depotNumber<<" "<<numberOfPlant<<endl;
		for (int i=0;i<depotNumber;i++){
			int a,b;
			file>>a;
			file>>b;
			if(i==depotNumber-1)
				cout <<a<<" "<<b;
			else			
				cout <<a<<" "<<b<<endl;
		}
		char ch;

		while(!file.eof())
		{
			file.get(ch);
			if (!file.eof() || ch!='\n')
				cout << ch;
		}
		int idx=vehiculeNumber+retailerNumber+1;
		for (int i=0;i<numberOfPlant;i++){
			double cb=rng.uniform(0.5)+0.5;
			int x=rng.random(400)-200;
			int y=rng.random(400)-200;
			double offerMin = totalDemand/numberOfPlant;
			double avail=rng.random(totalDemand-offerMin)+offerMin;
			double cc=rng.uniform(3)+2.0;
			cout<<idx++<<" "<<x<<" "<<y<<" 0\t";
			cout<<avail<<" "<<cb<<" "<<cc<<endl;


		}

	}

}
