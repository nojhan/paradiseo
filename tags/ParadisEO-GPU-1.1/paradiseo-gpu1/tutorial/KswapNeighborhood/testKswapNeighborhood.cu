/*
  <testKswapNeighborhood.cu>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

  Karima Boufaras, Thé Van LUONG

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.

  ParadisEO WebSite : http://paradiseo.gforge.inria.fr
  Contact: paradiseo-help@lists.gforge.inria.fr
*/


// standard includes
#define HAVE_SSTREAM

#include <stdexcept>  // runtime_error
#include <iostream>   // cout
#include <sstream>  // ostrstream, istrstream
#include <fstream>
#include <string.h>

// the general include for eo
#include <eo>
#include <ga.h>

using namespace std;

//----------------------------------------------------------------------------

//Representation and initializer
#include <eoInt.h>
#include <eoInit.h>

// fitness function
#include <eval/queenEval.h>
#include <eval/moFullEvalByModif.h>
#include <eval/moFullEvalByCopy.h>

//Neighbors and Neighborhoods
#include <problems/permutation/moShiftNeighbor.h>
#include <problems/permutation/moSwapNeighbor.h>
#include <problems/permutation/moSwapNeighborhood.h>

#include <neighborhood/moGPUXSwapN.h>
#include <neighborhood/moGPUXChange.h>
#include <neighborhood/moGPUNeighborhoodSizeUtils.h>

#include <neighborhood/moRndWithReplNeighborhood.h>
#include <neighborhood/moRndWithoutReplNeighborhood.h>
#include <neighborhood/moOrderNeighborhood.h>

// Define types of the representation solution, different neighbors and neighborhoods
//-----------------------------------------------------------------------------
typedef eoInt<unsigned int> Queen; //Permutation (Queen's problem representation)

typedef moSwapNeighbor<Queen> swapNeighbor; //swap Neighbor
typedef moSwapNeighborhood<Queen> swapNeighborhood; //classical swap Neighborhood

typedef moGPUXSwapN<Queen> xSwapNeighbor; //X-Swap Neighbor
typedef moGPUXChange<xSwapNeighbor> xSwapNeighborhood; // x-Swap Neighborhood

void main_function(int argc, char **argv) {

	/* =========================================================
	 *
	 * Parameters
	 *
	 * ========================================================= */

	// First define a parser from the command-line arguments
	eoParser parser(argc, argv);

	// For each parameter, define Parameter, read it through the parser,
	// and assign the value to the variable

	eoValueParam<uint32_t>
			seedParam(time(0), "seed", "Random number seed", 'S');
	parser.processParam(seedParam);
	unsigned seed = seedParam.value();

	// description of genotype
	eoValueParam<unsigned int> vecSizeParam(6, "vecSize", "Genotype size", 'V');
	parser.processParam(vecSizeParam, "Representation");
	unsigned vecSize = vecSizeParam.value();

	// Swap number
	eoValueParam<unsigned int> xSwapParam(2, "xSwap", "swap number", 'X');
	parser.processParam(xSwapParam, "xSwap");
	unsigned xSwap = xSwapParam.value();

	// the name of the "status" file where all actual parameter values will be saved
	string str_status = parser.ProgramName() + ".status"; // default value
	eoValueParam<string> statusParam(str_status.c_str(), "status",
			"Status file");
	parser.processParam(statusParam, "Persistence");

	// do the following AFTER ALL PARAMETERS HAVE BEEN PROCESSED
	// i.e. in case you need parameters somewhere else, postpone these
	if (parser.userNeedsHelp()) {
		parser.printHelp(cout);
		exit(1);
	}
	if (statusParam.value() != "") {
		ofstream os(statusParam.value().c_str());
		os << parser;// and you can use that file as parameter file
	}

	/* =========================================================
	 *
	 * Random seed
	 *
	 * ========================================================= */

	//reproducible random seed: if you don't change SEED above,
	// you'll aways get the same result, NOT a random run
	rng.reseed(seed);

	/* =========================================================
	 *
	 * Eval fitness function
	 *
	 * ========================================================= */

	queenEval<Queen> fullEval;

	/* =========================================================
	 *
	 * Initializer of the solution
	 *
	 * ========================================================= */

	eoInitPermutation<Queen> init(vecSize);

	/* =========================================================
	 *
	 * evaluation operators of a neighbor solution
	 *
	 * ========================================================= */

	moFullEvalByModif<swapNeighbor> swapEval(fullEval);

	moFullEvalByModif<xSwapNeighbor> xSwapEval(fullEval);

	/* =========================================================
	 *
	 * Neighbors and Neighborhoods
	 *
	 * ========================================================= */

	swapNeighborhood swapNH;
	xSwapNeighborhood xSwapNH(sizeMapping(vecSize,xSwap), xSwap);

	swapNeighbor n1;
	xSwapNeighbor nk(xSwap);

	/* =========================================================
	 *
	 * Init and eval a Queen
	 *
	 * ========================================================= */

	Queen solution;

	init(solution);

	fullEval(solution);

	std::cout << "Initial Solution:" << std::endl;
	std::cout << solution << std::endl << std::endl;

	/* =========================================================
	 *
	 * Use classical Neighbor and Neighborhood (swap)
	 *
	 * ========================================================= */

	std::cout << "SWAP NEIGHBORHOOD" << std::endl;
	std::cout << "-----------------" << std::endl;
	std::cout << "Neighbors List: (Neighbor -> fitness)" << std::endl;

	swapNH.init(solution, n1);
	swapEval(solution, n1);
	n1.print();
	while (swapNH.cont(solution)) {
		swapNH.next(solution, n1);
		swapEval(solution, n1);
		n1.print();
	}

	/* =========================================================
	 *
	 * Use K-swap Neighbor and Neighborhood (swap)
	 *
	 * ========================================================= */

	std::cout << "K-SWAP NEIGHBORHOOD" << std::endl;
	std::cout << "-----------------" << std::endl;
	std::cout << "Neighbors List: (Neighbor -> fitness)" << std::endl;
	std::cout << solution << std::endl << std::endl;
	xSwapNH.init(solution, nk);
	xSwapEval(solution, nk);
	nk.print();
	while (xSwapNH.cont(solution)) {
		xSwapNH.next(solution, nk);
		xSwapEval(solution, nk);
		nk.print();
	}

}

// A main that catches the exceptions

int main(int argc, char **argv) {
	try {
		main_function(argc, argv);
	} catch (exception& e) {
		cout << "Exception: " << e.what() << '\n';
	}
	return 1;
}
