/* 
* <tsp.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sébastien Cahon, Thomas Legrand
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

#include <eo>
#include <tsp>

void manage_configuration_file(eoParser & _parser);

int
main (int _argc, char* _argv [])
{
  std::string instancePath, crossoverType;
  unsigned int seed, populationSize, maxIterations, selectedParentNumber;
  double crossoverRate, mutationRate, elitismRate, tournamentRate;

  eoParser parser(_argc, _argv); 

  manage_configuration_file(parser);

  seed=atoi( (parser.getParamWithLongName("seed")->getValue()).c_str() );
  instancePath=parser.getParamWithLongName("instancePath")->getValue();
  populationSize=atoi( (parser.getParamWithLongName("popSize")->getValue()).c_str() );
  maxIterations=atoi( (parser.getParamWithLongName("maxIter")->getValue()).c_str() );
  crossoverRate=atof( (parser.getParamWithLongName("crossRate")->getValue()).c_str() );
  mutationRate=atof( (parser.getParamWithLongName("mutRate")->getValue()).c_str() );
  selectedParentNumber=atoi( (parser.getParamWithLongName("nbSelPar")->getValue()).c_str() );
  elitismRate=atof( (parser.getParamWithLongName("elitismRate")->getValue()).c_str() );
  tournamentRate=atof( (parser.getParamWithLongName("tournRate")->getValue()).c_str() );
  crossoverType=parser.getParamWithLongName("crossType")->getValue();

  srand (seed);
  Graph::load(instancePath.c_str());

  RouteInit init ;
  
  RouteEval full_eval ;
   
  eoPop <Route> pop (populationSize, init) ;
  apply <Route> (full_eval, pop) ;

  std :: cout << "[From] " << pop.best_element () << std :: endl ;
  
  eoGenContinue <Route> continu (maxIterations) ;
 
  eoStochTournamentSelect <Route> select_one ;

  eoSelectNumber <Route> select (select_one, selectedParentNumber) ;

  eoQuadOp <Route>*crossover;

  if(crossoverType.compare("Partial")==0)
    {
      crossover=new PartialMappedXover();
    }
  else if (crossoverType.compare("Order")==0)
    {
      crossover=new OrderXover();
    }
  else if (crossoverType.compare("Edge")==0)
    {
      crossover=new EdgeXover();
    }
  else
    {
      throw std::runtime_error("[tsp.cpp]: the crossover type '"+crossoverType+"' is not correct.");
    }
  
  CitySwap mutation ;
  
  eoSGATransform <Route> transform (*crossover, crossoverRate, mutation, mutationRate) ; 
  
  eoElitism <Route> merge (elitismRate) ;
  
  eoStochTournamentTruncate <Route> reduce (tournamentRate) ;
  
  eoEasyEA <Route> ea (continu, full_eval, select, transform, merge, reduce) ;
  
  ea (pop) ;
  
  std :: cout << "[To] " << pop.best_element () << std :: endl ;
    
  delete(crossover);

  return EXIT_SUCCESS;
}

void
manage_configuration_file(eoParser & _parser)
{
  std::ofstream os;

  _parser.getORcreateParam(std::string("benchs/berlin52.tsp"), "instancePath", "Path to the instance.", 
			   0, "Configuration", false);
  _parser.getORcreateParam((unsigned int)time(0), "seed", "Seed for rand.", 0, "Configuration", false);

  _parser.getORcreateParam((unsigned int)100, "popSize", "Size of the population.", 0, "Configuration", false);

  _parser.getORcreateParam((unsigned int)1000, "maxIter", "Maximum number of iterations.", 0, "Configuration", false);
  
  _parser.getORcreateParam((double)1.0, "crossRate", "Probability of crossover.", 0, "Configuration", false);
  
  _parser.getORcreateParam((double)0.01, "mutRate", "Probability of mutation.", 0, "Configuration", false);

  _parser.getORcreateParam((unsigned int)100, "nbSelPar", "Number of selected parents.", 0, "Configuration", false);

  _parser.getORcreateParam((double)1.0, "elitismRate", "Percentage of the best individuals kept.", 0, "Configuration", false);

  _parser.getORcreateParam((double)0.7, "tournRate", "Percentage of the individuals used during the tournament.", 
			   0, "Configuration", false);

  _parser.getORcreateParam(std::string("Partial"), "crossType", "Crossover to use, it can be 'Partial', 'Order' or 'Edge'.", 
			   0, "Configuration", false);

  if (_parser.userNeedsHelp())
    {
      _parser.printHelp(std::cout);
      exit(EXIT_FAILURE);
    }
  
  os.open("current_param");
  if(!os.is_open())
    {
      throw std::runtime_error("[tsp.cpp]: the file current_param cannot be created.");
    }
  os <<_parser;
  os.close();
}
