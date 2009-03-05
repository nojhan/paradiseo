/*
  <main.cpp>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille Nord Europe, 2006-2009
  (C) OPAC Team, LIFL, 2002-2009

  The Van LUONG,  (The-Van.Luong@inria.fr)
  Mahmoud FATENE, (mahmoud.fatene@inria.fr)

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
#include <peo>
#include <time.h>

#include <iostream>
#include <fstream>

#include "QAP.h"
#include "Move.h"
#include "QAPGA.h"
#include "qapPackUnpack.h"

using namespace std;

/** Set of parameters wrapped into a structure. We pass then the structure 
 * to a function which parses the parameters file. Doing so helps cleaning 
 * the code from the parts of reading the inputs.
 */
#include "parserStruct.h"
#include "utils.h"
/** The actual reading and parameters parsing is done inside this class utilities
 */ 


// Global variables
int n;                    // problem size
int** a;
int** b;         // flows and distances matrices

int bkv; //best known value

void main_function(int argc, char **argv)
{
 
 if (argc < 2){
    cout << "Please give a param file" << endl;
    exit(1);
  }

  
  // Declaration of useful variables to parse the parameters file and then 
  // put its elements into a structure
  eoParser parser(argc, argv);
  parameters param;
  parseFile(parser, param);

  string s (argv[1]);

  if ( (s.compare("-h") == 0) || (s.compare("--help") == 0 ) )
    ;//make help
  
  // Reading the a and b matrices of the QAP problem
  else
    loadInstances(param.inst.c_str(), n, bkv, a, b);
    
  MoveInit move_init;

  ProblemEval problem_eval;  

  MoveNext move_next;

  MoveIncrEval move_incr_eval;

  moBestImprSelect<Move> move_select;
 
  moSimpleMoveTabuList <Move> tabulist (param.tabuListSize);
  
  //moNoAspirCrit<Move> aspiration_criterion;
  moImprBestFitAspirCrit <Move> aspiration_criterion;
  
  moGenSolContinue <Problem> continu (param.TSmaxIter);

  moTS <Move> tabu_search (move_init, move_next, move_incr_eval, 
			   tabulist, aspiration_criterion, continu, problem_eval );

  ProblemInit chromInit;
  
  eoPop <Problem> pop (param.popSize, chromInit);
  
  // Initialisation of the parallel environment
  peo :: init (argc, argv);
  
  if (getNodeRank()==1)
  {
      std :: cout << "\nResult before the local search\n";
      //std :: cout << "size\t" << pop.size() << endl;
      for (int i=0;i<pop.size();i++)
        problem_eval(pop[i]);
      pop.sort();
      cout << pop;
  }
  // Local search
  
  peoMultiStart <Problem> initParallelTS (tabu_search);
  peoWrapper parallelTS (initParallelTS, pop);
  initParallelTS.setOwner(parallelTS);
  peo :: run( );
  peo :: finalize( );

  if (getNodeRank()==1)
  {
      cout << "\nInstance size = " << n << endl;
      cout << "Best known value in the litterature = " << bkv << endl;
      std :: cout << "\nResult after the local search\n";
      // sort and print the population
      pop.sort();
      cout << pop;
  }

 

  cout << endl;


}

// A main that catches the exceptions

int main(int argc, char **argv)
{

  try
    {  
      main_function(argc, argv);

    }
  catch(exception& e)
    {
      cout << "Exception: " << e.what() << '\n';
    }

	
  // desallocate memory
  for (int i=0; i<n; i++){
    delete[] a[i];
    delete[] b[i];
  }

  delete[] a;
  delete[] b;


  return 1;
}
