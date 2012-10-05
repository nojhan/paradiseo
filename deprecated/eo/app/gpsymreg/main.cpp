/*

    This program is free software; you can redistribute it and/or modify

    it under the terms of the GNU General Public License as published by

    the Free Software Foundation; either version 2 of the License, or

    (at your option) any later version.



    This library is distributed in the hope that it will be useful,

    but WITHOUT ANY WARRANTY; without even the implied warranty of

    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU

    Lesser General Public License for more details.



    You should have received a copy of the GNU General Public License

    along with this library; if not, write to the Free Software

    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA



    Contact: todos@geneura.ugr.es, http://geneura.ugr.es

	     jeggermo@liacs.nl

*/



#ifdef _MSC_VER

#pragma warning(disable:4786)

#endif



#ifdef HAVE_CONFIG_H

#include <config.h>

#endif



#include <iostream>

#include "gp/eoParseTree.h"

#include "eo"



using namespace gp_parse_tree;

using namespace std;



//-----------------------------------------------------------------------------



#include "node.h"

#include "parameters.h"

#include "fitness.h"





// TYPE DECLARATIONS FOR GP





typedef eoParseTree<FitnessType, Node > EoType;

typedef eoPop<EoType> Pop;



//-----------------------------------------------------------------------------



int main(int argc, char *argv[])

{



	// the vector containing the possible nodes

	vector<Node> initSequence;



	// initialise parameters

	Parameters parameter(argc, argv);



	// set the randomseed

	rng.reseed(parameter.randomseed);



	 // Create a generation counter

    	eoValueParam<unsigned> generationCounter(0, "Gen.");



    	// Create an incrementor (sub-class of eoUpdater). Note that the

    	// parameter's value is passed by reference,

    	// so every time the incrementer is updated (every generation),

    	// the data in generationCounter will change.

    	eoIncrementor<unsigned> increment(generationCounter.value());





	// create an instantiation of the fitness/evaluation function

	// it initializes the initSequence vector

	// the parameters are passed on as well

	RegFitness eval(generationCounter, initSequence, parameter);



	// Depth Initializor, set for Ramped Half and Half Initialization

	eoParseTreeDepthInit<FitnessType, Node> initializer(parameter.InitMaxDepth, initSequence, true, true);



	// create the initial population

	Pop pop(parameter.population_size, initializer);



	// and evaluate the individuals

	apply<EoType>(eval, pop);



	generationCounter.value()++; // set the generationCounter to 1





    	// define X-OVER



	eoSubtreeXOver<FitnessType, Node>   xover(parameter.MaxSize);



	// define MUTATION

      eoBranchMutation<FitnessType, Node> mutation(initializer, parameter.MaxSize);

//      eoExpansionMutation<FitnessType, Node> mutation(initializer, parameter.MaxSize);

//	eoCollapseSubtreeMutation<FitnessType, Node> mutation(initializer, parameter.MaxSize);

//	eoPointMutation<FitnessType, Node> mutation(initSequence);

//	eoHoistMutation<FitnessType, Node> mutation;



	// The operators are  encapsulated into an eoTRansform object,

    	// that performs sequentially crossover and mutation

	eoSGATransform<EoType> transform(xover, parameter.xover_rate, mutation, parameter.mutation_rate);



	// The robust tournament selection

	// in our case 5-tournament selection

    	eoDetTournamentSelect<EoType> selectOne(parameter.tournamentsize);

	// is now encapsulated in a eoSelectMany

	eoSelectMany<EoType> select(selectOne, parameter.offspring_size, eo_is_an_integer);



	// and the generational replacement

    	//eoGenerationalReplacement<EoType> replace;

	// or the SteadtState replacment

	//eoSSGAWorseReplacement<EoType> replace;

	// or comma selection

	eoCommaReplacement<EoType> replace;



    	// Terminators

    	eoGenContinue<EoType> term(parameter.nGenerations);



    	eoCheckPoint<EoType> checkPoint(term);



	// STATISTICS

    	eoAverageStat<EoType>     avg;

    	eoBestFitnessStat<EoType> best;





    	// Add it to the checkpoint,

    	// so the counter is updated (here, incremented) every generation

    	checkPoint.add(increment);

	checkPoint.add(avg);

	checkPoint.add(best);



#ifdef HAVE_GNUPLOT

	eoGnuplot1DMonitor gnuplotmonitor("gnuplotBestStats");

  	gnuplotmonitor.add(generationCounter);

	gnuplotmonitor.add(best);

	// we need to add a empty string variable if we want to seed the second fitness value

	eoValueParam<string> dummy1("", "Smallest Tree Size");

	gnuplotmonitor.add(dummy1);



	eoGnuplot1DMonitor gnuplotAvgmonitor("gnuplotAvgStats");

	gnuplotAvgmonitor.add(generationCounter);

	gnuplotAvgmonitor.add(avg);

	// we need to add a empty string variable if we want to seed the second fitness value

	eoValueParam<string> dummy2("", "Average Tree Size");

	gnuplotAvgmonitor.add(dummy2);



	checkPoint.add(gnuplotmonitor);

  	checkPoint.add(gnuplotAvgmonitor);

#endif

	// GP Generation

	eoEasyEA<EoType> gp(checkPoint, eval, select, transform, replace);



    	cout << "Initialization done" << endl;





    	try

    	{

      		gp(pop);

    	}

    	catch (exception& e)

    	{

	    cout << "exception: " << e.what() << endl;;

	    exit(EXIT_FAILURE);

    	}



    	return 1;



}
