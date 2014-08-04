/*
    This library is free software; you can redistribute it and/or modify
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

#ifndef _PARAMETERS_FUNCTION_H
#define _PARAMETERS_FUNCTION_H

#include <gp/eoParseTree.h>
#include <eo>

using namespace gp_parse_tree;
using namespace std;

struct Parameters{
			unsigned int nGenerations;	// -G
			unsigned population_size;	// -P
			unsigned offspring_size;	// -O
			unsigned int MaxSize;		// -S
			unsigned int InitMaxDepth;      // -D
			unsigned int randomseed;	// -R
			double xover_rate;		// -x
			double mutation_rate;		// -y
			unsigned int tournamentsize;    // -t


			Parameters(int argc, char **argv)
			{
				eoParser parser(argc,argv);

				// generations
				eoValueParam<unsigned int> paramGenerations(1, "generations", "Generations", 'G', false);
				parser.processParam( paramGenerations );
				nGenerations = paramGenerations.value();
				cerr << "nGenerations= " << nGenerations << endl;

				// populationsize
				eoValueParam<unsigned int> paramPopulationSize(10, "populationsize", "PopulationSize", 'P', false);
				parser.processParam( paramPopulationSize );
				population_size = paramPopulationSize.value();
				cerr << "population_size= " << population_size << endl;

				// offspringsize
				eoValueParam<unsigned int> paramOffspringSize(population_size, "offspringsize", "OffspringSize", 'O', false);
				parser.processParam( paramOffspringSize );
				offspring_size = paramOffspringSize.value();
				cerr << "offspring_size= " << offspring_size << endl;

				// maxsize
				eoValueParam<unsigned int> paramMaxSize(15, "maxsize", "MaxSize", 'S', false);
				parser.processParam( paramMaxSize );
				MaxSize = paramMaxSize.value();
				cerr << "MaxSize= " << MaxSize << endl;

				// initialmaxdepth
				eoValueParam<unsigned int> paramInitialMaxDepth(4, "initialmaxdepth", "InitialMaxDepth", 'D', false);
				parser.processParam( paramInitialMaxDepth );
				InitMaxDepth = paramInitialMaxDepth.value();
				cerr << "InitMaxDepth= " << InitMaxDepth << endl;

				// randomseed
				eoValueParam<unsigned int> paramRandomSeed(1, "randomseed", "Random Seed", 'R', false);
				parser.processParam( paramRandomSeed );
				randomseed = paramRandomSeed.value();
				cerr << "randomseed= " << randomseed << endl;


				// crossover-rate
				eoValueParam<double> paramXover(0.75, "crossoverrate", "crossover rate", 'x', false);
				parser.processParam(paramXover );
				xover_rate = paramXover.value();
				cerr << "xover_rate= " << xover_rate << endl;

				//mutation-rate
				eoValueParam<double> paramMutation(0.25, "mutationrate", "mutation rate", 'm', false);
				parser.processParam(paramMutation );
				mutation_rate = paramMutation.value();
				cerr << "mutation_rate= " << mutation_rate << endl;

				//tournament size
				eoValueParam<unsigned int > paramTournamentSize(5, "tournamentsize", "tournament size", 't', false);
				parser.processParam(paramTournamentSize );
				tournamentsize = paramTournamentSize.value();
				cerr << "Tournament Size= " << tournamentsize << endl;


				if (parser.userNeedsHelp())
			      	{
					parser.printHelp(cout);
					exit(1);
			      	}

			};

			~Parameters(){};
};

#endif
