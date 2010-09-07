/*
* <t-moSteadyFitSolContinue.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2008
*
* Sébastien Cahon, Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)
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
//-----------------------------------------------------------------------------
// t-moSteadyFitSolContinue.cpp
//-----------------------------------------------------------------------------

#include <eo>  // EO
#include <oldmo>  // MO

using std::cout;
using std::endl;

//-----------------------------------------------------------------------------

typedef EO<unsigned int> solution;

//-----------------------------------------------------------------------------

int
main()
{
  std::string test_result, test_1, test_2;
  int return_value, value_1, value_2;

  unsigned int i;
  solution solution_1, solution_2;

  moSteadyFitSolContinue<solution> continu(10,10);

  cout << "[ moSteadyFitSolContinue       ] ==> ";

  continu.init();

  i=0;
  solution_1.fitness(i);
  while( continu(solution_1) && i<50)
    {
      i++;

      if(i<11)
	{
	  solution_1.fitness(i);
	}
      
      if(i==11)
	{
	  test_1="KO";
	  try
	    {
	      continu(solution_2);
	    }
	  catch(std::runtime_error e)
	    {
	      test_1="OK";
	    }
	  value_1=((test_1.compare("KO")==0)?EXIT_FAILURE:EXIT_SUCCESS);
	}
      
      if(i==15)
	{
	  solution_1.fitness(i);
	}
    }

  test_2=((i!=25)?"KO":"OK");
  value_2=((test_2.compare("KO")==0)?EXIT_FAILURE:EXIT_SUCCESS);

  test_result=(((test_1.compare("OK")==0)&&(test_2.compare("OK")==0))?"OK":"KO");
  return_value=(((value_1==EXIT_SUCCESS)&&(value_2==EXIT_SUCCESS))?EXIT_SUCCESS:EXIT_FAILURE);

  cout << test_result << endl;
  return return_value;
}

//-----------------------------------------------------------------------------
