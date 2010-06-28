/*
  <simulated_annealing.cpp>
  Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
  (C) OPAC Team, LIFL, 2002-2008

  Sébastien Cahon, Jean-Charles Boisson

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

#include <eo>
#include <oldmo>
#include <tsp>

void manage_configuration_file(eoParser & _parser);

int
main (int _argc, char* _argv [])
{
  std::string instancePath, value;
  unsigned int seed, maxIterations;
  double threshold, geometricRatio, linearRatio, initialTemperature;

  eoParser parser(_argc, _argv); 

  manage_configuration_file(parser);

  seed=atoi( (parser.getParamWithLongName("seed")->getValue()).c_str() );
  instancePath=parser.getParamWithLongName("instancePath")->getValue();
  maxIterations=atoi( (parser.getParamWithLongName("maxIter")->getValue()).c_str() );
  initialTemperature=atof( (parser.getParamWithLongName("initialTemp")->getValue()).c_str() );
  threshold=atof( (parser.getParamWithLongName("threshold")->getValue()).c_str() );
  geometricRatio=atof( (parser.getParamWithLongName("geometricRatio")->getValue()).c_str() );
  linearRatio=atof( (parser.getParamWithLongName("lineaRatio")->getValue()).c_str() );
  value=parser.getParamWithLongName("coolSchedType")->getValue();

  srand (seed);
  Graph::load(instancePath.c_str());

  Route solution;

  RouteInit initializer;
  initializer (solution);

  RouteEval full_evaluation;
  full_evaluation (solution);

  std :: cout << "[From] " << solution << std :: endl;

  /* Tools for an efficient (? :-))
     local search ! */

  TwoOptRand two_opt_random_move_generator;

  TwoOptIncrEval two_opt_incremental_evaluation;

  TwoOpt move;

  moCoolingSchedule* coolingSchedule;

  if(value.compare("Geometric")==0)
    {
      coolingSchedule=new moGeometricCoolingSchedule(threshold, geometricRatio);
    }
  else if (value.compare("Linear")==0)
    {
      coolingSchedule=new moLinearCoolingSchedule(threshold, linearRatio);
    }
  else
    {
      throw std::runtime_error("[simulated_annealing.cpp]: the type of cooling schedule '"+value+"' is not correct.");
    }

  moGenSolContinue <Route> continu (maxIterations);

  moSA <TwoOpt> simulated_annealing (two_opt_random_move_generator, two_opt_incremental_evaluation,
				     continu, initialTemperature, *coolingSchedule, full_evaluation);
  simulated_annealing (solution);

  std :: cout << "[To] " << solution << std :: endl;

  delete(coolingSchedule);

  return EXIT_SUCCESS ;
}

void
manage_configuration_file(eoParser & _parser)
{
  std::ofstream os;

  _parser.createParam(std::string("../examples/tsp/benchs/berlin52.tsp"), "instancePath", "Path to the instance.",
		      0, "Configuration", false);

  _parser.getORcreateParam((unsigned int)time(0), "seed", "Seed for rand.", 0, "Configuration", false);

  _parser.getORcreateParam((unsigned int)1000, "maxIter", "Maximum number of iterations.", 0, "Configuration", false);

  _parser.getORcreateParam((double)1000, "initialTemp", "Initial temperature.", 0, "Configuration", false);

  _parser.getORcreateParam((double)0.1, "threshold", "Minimum temperature allowed.", 0, "Configuration", false);

  _parser.getORcreateParam((double)0.98, "geometricRatio", "Ratio used if exponential cooling schedule is chosen.", 0, "Configuration", false);

  _parser.getORcreateParam((double)0.5, "lineaRatio", "Ratio used if linear cooling schedule is chosen.", 0, "Configuration", false);

  _parser.getORcreateParam(std::string("Geometric"), "coolSchedType", "Type the cooling schedule: 'Geometric' or 'Linear'.", 
			   0, "Configuration", false);
  
  if (_parser.userNeedsHelp())
    {
      _parser.printHelp(std::cout);
      exit(EXIT_FAILURE);
    }
  
  os.open("current_param");
  if(!os.is_open())
    {
      throw std::runtime_error("[simulated_annealing.cpp]: the file current_param cannot be created.");
    }
  os <<_parser;
  os.close();
}
