/*
<testPopLS.cpp>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  ue,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

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

//-----------------------------------------------------------------------------
//Representation and initializer
#include <eoInt.h>
#include <eoInit.h>
#include <eoScalarFitness.h>
#include <ga/eoBit.h>
#include <eoPop.h>

// fitness function
#include <eval/oneMaxPopEval.h>
#include <problems/eval/moPopBitEval.h>

//Neighbors and Neighborhoods
#include <problems/bitString/moPopBitNeighbor.h>
#include <neighborhood/moOrderNeighborhood.h>

//Algorithm and its components
#include <coolingSchedule/moCoolingSchedule.h>
#include <algo/moSimpleHC.h>

//comparator
#include <comparator/moSolNeighborComparator.h>

//continuators
#include <continuator/moTrueContinuator.h>
#include <continuator/moCheckpoint.h>
#include <continuator/moFitnessStat.h>
#include <utils/eoFileMonitor.h>
#include <continuator/moCounterMonitorSaver.h>

#include "moPopFitContinuator.h"

#include <mo>


//-----------------------------------------------------------------------------
// Define types of the representation solution, different neighbors and neighborhoods
//-----------------------------------------------------------------------------
typedef moPopSol<eoBit<double> > Solution; //Permutation (Queen's problem representation)

typedef moPopBitNeighbor<double> Neighbor; //shift Neighbor
typedef moRndWithReplNeighborhood<Neighbor> Neighborhood; //rnd shift Neighborhood (Indexed)


class popInit: public eoInit<Solution>{

public:

	popInit( eoInitFixedLength<eoBit<double> > & _rnd, unsigned int _popSize):rnd(_rnd), popSize(_popSize){}

	void operator()(Solution & _sol){



	    eoBit<double> tmp;

	    for(unsigned int i=0; i<popSize; i++){
	    	rnd(tmp);
	    	_sol.push_back(tmp);
	    }
	}

private:
	eoInitFixedLength<eoBit<double> >& rnd;
	unsigned int popSize;

};

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

    // the number of steps of the random walk
    eoValueParam<unsigned int> stepParam(100, "nbStep", "Number of steps of the random walk", 'n');
    parser.processParam( stepParam, "Representation" );
    unsigned nbStep = stepParam.value();

    // description of genotype
    eoValueParam<unsigned int> vecSizeParam(8, "vecSize", "Genotype size", 'V');
    parser.processParam( vecSizeParam, "Representation" );
    unsigned vecSize = vecSizeParam.value();

    // description of genotype
    eoValueParam<unsigned int> popSizeParam(10, "popSize", "population size", 'P');
    parser.processParam( popSizeParam, "Representation" );
    unsigned popSize = popSizeParam.value();

    // the name of the "status" file where all actual parameter values will be saved
    string str_status = parser.ProgramName() + ".status"; // default value
    eoValueParam<string> statusParam(str_status.c_str(), "status", "Status file");
    parser.processParam( statusParam, "Persistence" );

    // the name of the output file
    string str_out = "out.dat"; // default value
    eoValueParam<string> outParam(str_out.c_str(), "out", "Output file of the sampling", 'o');
    parser.processParam(outParam, "Persistence" );

    // description of genotype
    eoValueParam<unsigned int> pparam(10, "p", "p", 'p');
    parser.processParam( pparam, "Representation" );
    unsigned p = pparam.value();

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
    // you'll always get the same result, NOT a random run
    rng.reseed(seed);


    /* =========================================================
     *
     * Eval fitness function
     *
     * ========================================================= */

    oneMaxEval< eoBit<double> > eval;
    oneMaxPopEval< eoBit<double> > popEval(eval, p);


    /* =========================================================
     *
     * Initilisation of the solution
     *
     * ========================================================= */

    eoUniformGenerator<bool> uGen;
    eoInitFixedLength<eoBit<double> > rnd(vecSize, uGen);

	popInit random(rnd, popSize);




    /* =========================================================
     *
     * evaluation of a neighbor solution
     *
     * ========================================================= */

    moPopBitEval<Neighbor> evalNeighbor(eval,p);

//	Neighbor n;
//
//	n.index(3);
//	moEval(sol, n);
//	n.move(sol);
//    popEval(sol);
//    sol.printOn(std::cout);
//	std::cout << "fit neighor: " << n.fitness() << std::endl;




    /* =========================================================
     *
     * the neighborhood of a solution
     *
     * ========================================================= */

    Neighborhood neighborhood(vecSize*popSize);

    moPopFitContinuator<Neighbor> cont(vecSize);

    /* =========================================================
     *
     * the local search algorithm
     *
     * ========================================================= */

    //moSimpleHC<Neighbor> ls(neighborhood, popEval, evalNeighbor, cont);

    /* =========================================================
     *
     * execute the local search from random solution
     *
     * ========================================================= */

//    ls(sol);
//
//    std::cout << "final solution:" << std::endl;
//    sol.printOn(std::cout);
//    std::cout << std::endl;




    /* =========================================================
     *
     * The sampling of the search space
     *
     * ========================================================= */

    // sampling object :
    //    - random initialization
    //    - neighborhood to compute the next step
    //    - fitness function
    //    - neighbor evaluation
    //    - number of steps of the walk
    moAutocorrelationSampling<Neighbor> sampling(random, neighborhood, popEval, evalNeighbor, nbStep);

    /* =========================================================
     *
     * execute the sampling
     *
     * ========================================================= */

    sampling();

    /* =========================================================
     *
     * export the sampling
     *
     * ========================================================= */

    // to export the statistics into file
    sampling.fileExport(str_out);

    // to get the values of statistics
    // so, you can compute some statistics in c++ from the data
    const std::vector<double> & fitnessValues = sampling.getValues(0);

    std::cout << "First values:" << std::endl;
    std::cout << "Fitness  " << fitnessValues[0] << std::endl;

    std::cout << "Last values:" << std::endl;
    std::cout << "Fitness  " << fitnessValues[fitnessValues.size() - 1] << std::endl;

    // more basic statistics on the distribution:
    moStatistics statistics;

    vector<double> rho, phi;

    statistics.autocorrelation(fitnessValues, 20, rho, phi);

    for (unsigned s = 0; s < rho.size(); s++)
        std::cout << s << " " << "rho=" << rho[s] << ", phi=" << phi[s] << std::endl;

//    Queen solution1, solution2;
//
//    init(solution1);
//
//    fullEval(solution1);
//
//    std::cout << "#########################################" << std::endl;
//    std::cout << "initial solution1: " << solution1 << std::endl ;
//
//    localSearch1(solution1);
//
//    std::cout << "final solution1: " << solution1 << std::endl ;
//    std::cout << "#########################################" << std::endl;


    /* =========================================================
     *
     * the cooling schedule of the process
     *
     * ========================================================= */

    // initial temp, factor of decrease, number of steps without decrease, final temp.


    /* =========================================================
     *
     * Comparator of neighbors
     *
     * ========================================================= */



    /* =========================================================
     *
     * Example of Checkpointing
     *
     * ========================================================= */

//    moTrueContinuator<shiftNeighbor> continuator;//always continue
//    moCheckpoint<shiftNeighbor> checkpoint(continuator);
//    moFitnessStat<Queen> fitStat;
//    checkpoint.add(fitStat);
//    eoFileMonitor monitor("fitness.out", "");
//    moCounterMonitorSaver countMon(100, monitor);
//    checkpoint.add(countMon);
//    monitor.add(fitStat);
//
//    moSA<shiftNeighbor> localSearch2(rndShiftNH, fullEval, shiftEval, coolingSchedule, solComparator, checkpoint);
//
//    init(solution2);
//
//    fullEval(solution2);
//
//    std::cout << "#########################################" << std::endl;
//    std::cout << "initial solution2: " << solution2 << std::endl ;
//
//    localSearch2(solution2);
//
//    std::cout << "final solution2: " << solution2 << std::endl ;
//    std::cout << "#########################################" << std::endl;
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

