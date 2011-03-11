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

int main (int _argc, char* _argv [])
{

  eoParser parser(_argc, _argv); 

  manage_configuration_file(parser);

  unsigned int seed=atoi( (parser.getParamWithLongName("seed")->getValue()).c_str() );
  std::string instancePath=parser.getParamWithLongName("instancePath")->getValue();
  unsigned int populationSize=atoi( (parser.getParamWithLongName("popSize")->getValue()).c_str() );
  unsigned int maxGen=atoi( (parser.getParamWithLongName("maxGen")->getValue()).c_str() );
  double crossoverRate=atof( (parser.getParamWithLongName("crossRate")->getValue()).c_str() );
  double mutationRate=atof( (parser.getParamWithLongName("mutRate")->getValue()).c_str() );
  unsigned int nbOffspring=atoi( (parser.getParamWithLongName("nbOffspring")->getValue()).c_str() );
  std::string crossoverType=parser.getParamWithLongName("crossType")->getValue();

  // random number generator
  srand (seed);
  
  // load test instance
  Graph::load(instancePath.c_str());



  /*** the representation-dependent things ***/
  
  // the evaluation function
  RouteEval full_eval ;
  // the genotype (through a genotype initializer)
  RouteInit init ;
  // crossover
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
  // mutation 
  CitySwap mutation ;
  // variation operators
  eoSGATransform <Route> transform (*crossover, crossoverRate, mutation, mutationRate) ; 



   /*** the representation-independent things ***/
  
  // initialization of the population
   eoPop <Route> pop (populationSize, init) ;
   apply <Route> (full_eval, pop) ;
   // select
   eoDetTournamentSelect <Route> select_one ;
   eoSelectNumber <Route> select (select_one, nbOffspring) ;
   // replace
   eoGenerationalReplacement <Route> genReplace;
   eoWeakElitistReplacement <Route> replace(genReplace);
   // stopping criteria
   eoGenContinue <Route> continu (maxGen) ;
   // algorithm
   eoEasyEA <Route> ea (continu, full_eval, select, transform, replace) ;

  

   /*** Go ! ***/

   // initial solution
   std :: cout << "[From] " << pop.best_element () << std :: endl ;
  
   // run the algo
   ea(pop);

   // final solution
  std :: cout << "[To] " << pop.best_element () << std :: endl ;
  


  // delete pointer
  delete(crossover);

  // that's all
  return EXIT_SUCCESS;
}



void manage_configuration_file(eoParser & _parser)
{
  std::ofstream os;
  _parser.getORcreateParam(std::string("../tsp/benchs/berlin52.tsp"), "instancePath", "Path to the instance.", 0, "Configuration", false);
  _parser.getORcreateParam((unsigned int)time(0), "seed", "Seed for rand.", 0, "Configuration", false);
  _parser.getORcreateParam((unsigned int)100, "popSize", "Size of the population.", 0, "Configuration", false);
  _parser.getORcreateParam((unsigned int)1000, "maxGen", "Maximum number of generations.", 0, "Configuration", false);
  _parser.getORcreateParam((double)1.0, "crossRate", "Probability of crossover.", 0, "Configuration", false);
  _parser.getORcreateParam((double)0.01, "mutRate", "Probability of mutation.", 0, "Configuration", false);
  _parser.getORcreateParam((unsigned int)100, "nbOffspring", "Number of offspring.", 0, "Configuration", false);
  _parser.getORcreateParam(std::string("Partial"), "crossType", "Crossover to use, it can be 'Partial', 'Order' or 'Edge'.", 0, "Configuration", false);
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
