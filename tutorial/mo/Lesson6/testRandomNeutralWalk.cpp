//-----------------------------------------------------------------------------
/** testRandomNeutralWalk.cpp
 *
 * SV - 22/02/10
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
#include <paradiseo/eo.h>
#include <paradiseo/eo/ga.h>

using namespace std;

//-----------------------------------------------------------------------------
// fitness function
#include <paradiseo/problems/eval/royalRoadEval.h>
#include <paradiseo/eo/eoInt.h>
#include <paradiseo/mo/neighborhood/moRndWithoutReplNeighborhood.h>
#include <paradiseo/mo/problems/bitString/moBitNeighbor.h>

#include <paradiseo/mo/eval/moFullEvalByModif.h>
#include <paradiseo/mo/eval/moFullEvalByCopy.h>
#include <paradiseo/mo/comparator/moNeighborComparator.h>
#include <paradiseo/mo/comparator/moSolNeighborComparator.h>
#include <paradiseo/mo/continuator/moTrueContinuator.h>
#include <paradiseo/mo/algo/moLocalSearch.h>
#include <paradiseo/mo/explorer/moRandomNeutralWalkExplorer.h>

#include <paradiseo/mo/continuator/moCheckpoint.h>
#include <paradiseo/mo/continuator/moFitnessStat.h>
#include <paradiseo/eo/utils/eoDistance.h>
#include <paradiseo/mo/continuator/moDistanceStat.h>
#include <paradiseo/mo/neighborhood/moOrderNeighborhood.h>
#include <paradiseo/mo/continuator/moNeighborhoodStat.h>
#include <paradiseo/mo/continuator/moMinNeighborStat.h>
#include <paradiseo/mo/continuator/moMaxNeighborStat.h>
#include <paradiseo/mo/continuator/moSecondMomentNeighborStat.h>
#include <paradiseo/mo/continuator/moNbInfNeighborStat.h>
#include <paradiseo/mo/continuator/moNbSupNeighborStat.h>
#include <paradiseo/mo/continuator/moNeutralDegreeNeighborStat.h>
#include <paradiseo/mo/continuator/moSizeNeighborStat.h>

// REPRESENTATION
//-----------------------------------------------------------------------------
typedef eoBit<unsigned> Indi;
typedef moBitNeighbor<unsigned int> Neighbor ; // incremental evaluation
typedef moRndWithoutReplNeighborhood<Neighbor> Neighborhood ;

void main_function(int argc, char **argv)
{
    /* =========================================================
    *
    * Parameters
    *
    * ========================================================= */

    // First define a parser from the command-line arguments
    eoParser parser(argc, argv);

    // For each parameter, define Parameter, read it through the parser,
    // and assign the value to the variable

    eoValueParam<uint32_t> seedParam(time(0), "seed", "Random number seed", 'S');
    parser.processParam( seedParam );
    unsigned seed = seedParam.value();

    // description of genotype
    eoValueParam<unsigned int> vecSizeParam(8, "vecSize", "Genotype size", 'V');
    parser.processParam( vecSizeParam, "Representation" );
    unsigned vecSize = vecSizeParam.value();

    eoValueParam<unsigned int> blockSizeParam(2, "blockSize", "Size of block in the royal road", 'k');
    parser.processParam( blockSizeParam, "Representation" );
    unsigned blockSize = blockSizeParam.value();

    eoValueParam<unsigned int> stepParam(10, "nbStep", "Number of steps of the random walk", 'n');
    parser.processParam( stepParam, "Representation" );
    unsigned nbStep = stepParam.value();

    // the name of the "status" file where all actual parameter values will be saved
    string str_status = parser.ProgramName() + ".status"; // default value
    eoValueParam<string> statusParam(str_status.c_str(), "status", "Status file");
    parser.processParam( statusParam, "Persistence" );

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

    RoyalRoadEval<Indi> eval(blockSize);


    /* =========================================================
     *
     * Initilisazor of the solution
     *
     * ========================================================= */

    // a Indi random initializer
    eoUniformGenerator<bool> uGen;
    eoInitFixedLength<Indi> random(vecSize, uGen);


    /* =========================================================
     *
     * Evaluation of a neighbor solution
     *
     * ========================================================= */

    moFullEvalByModif<Neighbor> nhEval(eval);

    //An eval by copy can be used instead of the eval by modif
    //moFullEvalByCopy<Neighbor> nhEval(eval);


    /* =========================================================
     *
     * Comparator of neighbors
     *
     * ========================================================= */

    moNeighborComparator<Neighbor> comparator;
    moSolNeighborComparator<Neighbor> solComparator;


    /* =========================================================
     *
     * the neighborhood of a solution
     *
     * ========================================================= */

    Neighborhood neighborhood(vecSize);


    /* =========================================================
     *
     * a neighborhood explorer solution
     *
     * ========================================================= */

    moRandomNeutralWalkExplorer<Neighbor> explorer(neighborhood, nhEval, solComparator, nbStep);


    /* =========================================================
     *
     * initial random solution
     *
     * ========================================================= */

    Indi solution;

    random(solution);

    //Can be eval here, else it will be done at the beginning of the localSearch
    eval(solution);

    /* =========================================================
     *
     * the continuator and the checkpoint
     *
     * ========================================================= */

    moTrueContinuator<Neighbor> continuator;//always continue

    moCheckpoint<Neighbor> checkpoint(continuator);

    moFitnessStat<Indi> fStat;

    eoHammingDistance<Indi> distance;
    moDistanceStat<Indi, unsigned> distStat(distance, solution);  // distance from the intial solution

    moOrderNeighborhood<Neighbor> nh(vecSize);
    moNeighborhoodStat< Neighbor > neighborhoodStat(nh, nhEval, comparator, solComparator);
    moMinNeighborStat< Neighbor > minStat(neighborhoodStat);
    moSecondMomentNeighborStat< Neighbor > secondMomentStat(neighborhoodStat);
    moMaxNeighborStat< Neighbor > maxStat(neighborhoodStat);

    moNbSupNeighborStat< Neighbor > nbSupStat(neighborhoodStat);
    moNbInfNeighborStat< Neighbor > nbInfStat(neighborhoodStat);
    moNeutralDegreeNeighborStat< Neighbor > ndStat(neighborhoodStat);
    moSizeNeighborStat< Neighbor > sizeStat(neighborhoodStat);

    eoValueParam<unsigned int> genCounter(-1,"Gen");
    eoIncrementor<unsigned int> increm(genCounter.value());

    checkpoint.add(fStat);
    checkpoint.add(distStat);
    checkpoint.add(neighborhoodStat);
    checkpoint.add(minStat);
    checkpoint.add(secondMomentStat);
    checkpoint.add(maxStat);
    checkpoint.add(nbInfStat);
    checkpoint.add(ndStat);
    checkpoint.add(nbSupStat);
    checkpoint.add(sizeStat);
    checkpoint.add(increm);

    eoFileMonitor outputfile("out.dat", " ");
    checkpoint.add(outputfile);

    outputfile.add(genCounter);
    outputfile.add(fStat);
    outputfile.add(distStat);
    outputfile.add(minStat);
    outputfile.add(secondMomentStat);
    outputfile.add(maxStat);
    outputfile.add(nbInfStat);
    outputfile.add(ndStat);
    outputfile.add(nbSupStat);
    outputfile.add(sizeStat);

    /* =========================================================
     *
     * the local search algorithm
     *
     * ========================================================= */

    moLocalSearch<Neighbor> localSearch(explorer, checkpoint, eval);

    /* =========================================================
     *
     * execute the local search from random sollution
     *
     * ========================================================= */

    std::cout << "initial: " << solution << std::endl ;

    localSearch(solution);

    std::cout << "final:   " << solution << std::endl ;

}

// A main that catches the exceptions

int main(int argc, char **argv)
{
    try {
        main_function(argc, argv);
    }
    catch (exception& e) {
        cout << "Exception: " << e.what() << '\n';
    }
    return 1;
}
