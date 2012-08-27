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

/**
 * Declaration of the necessary headers: In these are defined the class QAP,
 * redefinition of the crossover, mutation, initialisation of solution.
 * 
 */ 
#include <peo>
#include "QAP.h"
#include "QAPGA.h"
#include <string>
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
int** b;         // a and  matrices

int bkv; //best known value

void main_function(int argc, char **argv)
{
  // Declaration of useful variables to parse the parameters file and then 
  // its elements into a structure
  if (argc < 2){
    cout << "Please give a param file" << endl;
    exit(1);
  }
   
  eoParser parser(argc, argv);
  parameters param;
  parseFile(parser, param);
  rng.reseed(param.seed);
  
  
  string s (argv[1]);

  if ( (s.compare("-h") == 0) || (s.compare("--help") == 0 ) )
    ;//make help
  
  // Reading the a and b matrices of the QAP problem
  else
   loadInstances(param.inst.c_str(), n, bkv, a, b);

  // Declaration of class wrapping the evaluation function of the QAP
  ProblemEvalFunc plainEval;
  eoEvalFuncCounter<Problem> eval(plainEval);
 
  // Class involving a simple call to the function of initialisation of a solution
  ProblemInit chromInit;


  eoPop<Problem> pop1(param.popSize, chromInit);  // Initialise the population
  eoPop<Problem> pop2(param.popSize, chromInit);  // Initialise the population
  eoPop<Problem> pop3(param.popSize, chromInit);  // Initialise the population

  // The robust tournament selection
  eoDetTournamentSelect<Problem> selectOne(param.tSize);
  // is now encapsulated in a eoSelectPerc (entage)
  eoSelectPerc<Problem> select(selectOne);// by default rate==1

  ProblemXover Xover;  // CROSSOVER
  ProblemSwapMutation  mutationSwap;  // MUTATION
  
  // The operators are  encapsulated into an eoTRansform object
  eoSGATransform<Problem> transform(Xover, param.pCross, mutationSwap, param.pMut);

  // REPLACE
  eoPlusReplacement<Problem> replace;

  eoGenContinue<Problem> genCont(param.maxGen); // generation continuation
 
  eoEasyEA<Problem> gga1(genCont, plainEval, select, transform, replace);
  eoEasyEA<Problem> gga2(genCont, plainEval, select, transform, replace);
  eoEasyEA<Problem> gga3(genCont, plainEval, select, transform, replace);
  
  // Start the parallel EA
  peo :: init (argc, argv);
  if (getNodeRank()==1)
  {
  	  apply<Problem>(eval, pop1);
	  pop1.sort();
	  cout << "Initial Population\n" << pop1 << endl;
  }  
  if (getNodeRank()==2)
  {
	  apply<Problem>(eval, pop2);
	  pop2.sort();
	  cout << "Initial Population\n" << pop2 << endl;
  }
  if (getNodeRank()==3)
  {  
	  apply<Problem>(eval, pop3);
	  pop3.sort();
	  cout << "Initial Population\n" << pop3 << endl;
  }
  
  peoWrapper parallelEA1 (gga1, pop1);
  peoWrapper parallelEA2 (gga2, pop2);
  peoWrapper parallelEA3 (gga3, pop3);
  
  peo :: run( );
  peo :: finalize( );

  // Print (sorted) intial population
  if (getNodeRank()==1)
  {
	  pop1.sort();
	  cout << "FINAL Population\n" << pop1 << endl;
	  cout << "\nInstance size = " << n << endl;
	  cout << "Best known value in the litterature = " << bkv <<"\n"<< endl;
	  cout << "Best solution found\t" << pop1[0].fitness() << endl;
  }
  if (getNodeRank()==2)
  {
	  pop2.sort();
	  cout << "FINAL Population\n" << pop2 << endl;
	  cout << "\nInstance size = " << n << endl;
	  cout << "Best known value in the litterature = " << bkv <<"\n"<< endl;
	  cout << "Best solution found\t" << pop2[0].fitness() << endl;
  }
  if (getNodeRank()==3)
  {
	  pop3.sort();
	  cout << "FINAL Population\n" << pop3 << endl;
	  cout << "\nInstance size = " << n << endl;
	  cout << "Best known value in the litterature = " << bkv <<"\n"<< endl;
	  cout << "Best solution found\t" << pop3[0].fitness() << endl;
  }
  
  // GENERAL


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
