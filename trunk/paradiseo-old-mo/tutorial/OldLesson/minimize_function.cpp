/*
  <minimize_function.cpp>
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
#include <function>

void manage_configuration_file(eoParser & _parser);

int
main (int _argc, char* _argv [])
{
  std::string selectionType;
  double initialBound, searchBound, searchStep;
  eoParser parser(_argc, _argv); 

  manage_configuration_file(parser);

  initialBound=atof( (parser.getParamWithLongName("initialBound")->getValue()).c_str() );
  searchBound=atof( (parser.getParamWithLongName("searchBound")->getValue()).c_str() );
  searchStep=atof( (parser.getParamWithLongName("searchStep")->getValue()).c_str() );
  selectionType=parser.getParamWithLongName("selectionType")->getValue();

  Affectation solution;

  AffectationInit initialize(initialBound);
  initialize (solution);

  AffectationEval evaluation;
  evaluation (solution);

  std::cout << "Initial affectation : " << std::endl;
  std::cout << "\t x1 = " << solution.first << std::endl;
  std::cout << "\t x2 = " << solution.second << std::endl;
  std::cout << "\t f(x1,x2) = " << solution.fitness() << std::endl;

  DeviationInit deviation_initializer(searchBound);

  DeviationNext deviation_next_move_generator(searchBound, searchStep);

  DeviationIncrEval deviation_incremental_evaluation;

  moMoveSelect<Deviation>* deviation_selection;
  
  if(selectionType.compare("Best")==0)
    {
      deviation_selection= new moBestImprSelect<Deviation>();
    }
  else if (selectionType.compare("First")==0)
    {
      deviation_selection= new moFirstImprSelect<Deviation>();
    }
  else if (selectionType.compare("Random")==0)
    {
      deviation_selection= new moRandImprSelect<Deviation>();
    }
  else
    {
      throw std::runtime_error("[minimize_function.cpp]: the type of selection '"+selectionType+"' is not correct.");
    }

  moHC <Deviation> hill_climbing (deviation_initializer, deviation_next_move_generator, deviation_incremental_evaluation, 
				  *deviation_selection, evaluation);
  hill_climbing (solution) ;
  
  std::cout << "Final affectation : " << std::endl;
  std::cout << "\t x1 = " << solution.first << std::endl;
  std::cout << "\t x2 = " << solution.second << std::endl;
  std::cout << "\t f(x1,x2) = " << solution.fitness() << std::endl;
  
  delete(deviation_selection);

  return EXIT_SUCCESS;
}

void
manage_configuration_file(eoParser & _parser)
{
  std::ofstream os;

  _parser.createParam((double)1, "initialBound", "Bound for the initial affectation.", 0, "Configuration", false);
  
  _parser.createParam((double)1, "searchBound", "Bound for neighbourhood exploration.", 0, "Configuration", false);

  _parser.createParam((double)1, "searchStep", "Step between two values during the neighbourhood exploration.", 
		      0, "Configuration", false);

  _parser.createParam(std::string("First"), "selectionType", "Type of the selection: 'Best', 'First' or 'Random'.", 
			   0, "Configuration", false);
  
  if (_parser.userNeedsHelp())
    {
      _parser.printHelp(std::cout);
      exit(EXIT_FAILURE);
    }

  os.open("current_param");
  if(!os.is_open())
    {
      throw std::runtime_error("[minimize_function.cpp]: the file current_param cannot be created.");
    }
  os <<_parser;
  os.close();
}
