//-----------------------------------------------------------------------------
/** testKNeighborhood.cpp
 *
 * KB - 20/10/10
 *
 */
//-----------------------------------------------------------------------------

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

#include <neighborhood/moKswapNeighbor.h>
#include <neighborhood/moKswapNeighborhood.h>

#include <neighborhood/moRndWithReplNeighborhood.h>
#include <neighborhood/moRndWithoutReplNeighborhood.h>
#include <neighborhood/moOrderNeighborhood.h>

// Define types of the representation solution, different neighbors and neighborhoods
//-----------------------------------------------------------------------------
typedef eoInt<unsigned int> Queen; //Permutation (Queen's problem representation)

typedef moSwapNeighbor<Queen> swapNeighbor; //swap Neighbor
typedef moSwapNeighborhood<Queen> swapNeighborhood; //classical swap Neighborhood

typedef moKswapNeighbor<Queen> kswapNeighbor; //k-swap Neighbor
typedef moKswapNeighborhood<kswapNeighbor> kswapNeighborhood; // k- swap Neighborhood

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
	eoValueParam<unsigned int> KswapParam(1, "Kswap", "swap number", 'N');
	parser.processParam(KswapParam, "Kswap");
	unsigned Kswap = KswapParam.value();

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

	moFullEvalByModif<kswapNeighbor> kswapEval(fullEval);

	/* =========================================================
	 *
	 * Neighbors and Neighborhoods
	 *
	 * ========================================================= */

	swapNeighborhood swapNH;
	kswapNeighborhood kswapNH(vecSize, Kswap);

	swapNeighbor n1;
	kswapNeighbor nk(Kswap);

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

	kswapNH.init(solution, nk);
	kswapEval(solution, nk);
	nk.print();
	while (kswapNH.cont(solution)) {
		kswapNH.next(solution, nk);
		kswapEval(solution, nk);
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
