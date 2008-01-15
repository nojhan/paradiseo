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

#include <mo>
#include <tsp>

int
main (int _argc, char* _argv [])
{
  if (_argc != 2)
    {
      std :: cerr << "Usage : ./simulated_annealing [instance]" << std :: endl;
      return EXIT_FAILURE;
    }

  Graph::load (_argv [1]);

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

  moExponentialCoolingSchedule cooling_schedule (0.1, 0.98);
  //moLinearCoolingSchedule cooling_schedule (0.1, 0.5);

  moGenSolContinue <Route> continu (1000); /* Temperature Descreasing
					      will occur each 1000
					      iterations */

  moSA <TwoOpt> simulated_annealing (two_opt_random_move_generator, two_opt_incremental_evaluation,
				     continu, 1000, cooling_schedule, full_evaluation);
  simulated_annealing (solution);

  std :: cout << "[To] " << solution << std :: endl;

  return EXIT_SUCCESS ;
}

