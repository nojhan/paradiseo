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
 * Declaration of the necessary headers: In these are defined the class Problem,
 * redefinition of the crossover, mutation, initialisation of solution.
 * 
 */ 
#include <peo>
#include "QAP.h"
#include "QAPGA.h"
#include "qapPackUnpack.h"
#include <string>
using namespace std;
//typedef Problem Problem;
/** Set of parameters wrapped into a structure. We pass then the structure 
 * to a function which parses the parameters file. Doing so helps cleaning 
 * the code from the parts of reading the inputs.
 */
#include "parserStruct.h"
#include "utils.h"
/** The actual reading and parameters parsing is done inside this class utilities
 */ 

// Global variables
int n;                    // Problem size
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
  
  // Reading the a and b matrices of the Problem Problem
  else
   loadInstances(param.inst.c_str(), n, bkv, a, b);

  // Declaration of class wrapping the evaluation function of the Problem
  ProblemEvalFunc plainEval;
  eoEvalFuncCounter<Problem> eval(plainEval);
 
  // Class involving a simple call to the function of initialisation of a solution
  ProblemInit chromInit;


  eoPop<Problem> pop_1(param.popSize, chromInit);  // Initialise the population
  eoPop<Problem> pop_2(param.popSize, chromInit);  // Initialise the population
  eoPop<Problem> pop_3(param.popSize, chromInit);  // Initialise the population
  
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
  
  eoCheckPoint< Problem > checkpoint_1( genCont );
  eoCheckPoint< Problem > checkpoint_2( genCont );
  eoCheckPoint< Problem > checkpoint_3( genCont );
  
  eoEasyEA<Problem> gga_1(checkpoint_1, plainEval, select, transform, replace);
  eoEasyEA<Problem> gga_2(checkpoint_2, plainEval, select, transform, replace);
  eoEasyEA<Problem> gga_3(checkpoint_3, plainEval, select, transform, replace);
  
  peo :: init (argc, argv);
  initDebugging();
  setDebugMode(true);
  
  // Start the parallel EA
  
  if (getNodeRank()==1)
  {
  	  apply<Problem>(eval, pop_1);
	  pop_1.sort();
	  cout << "Initial Population 1\n" << pop_1 << endl;
  } 
   
  if (getNodeRank()==2)
  {
	  apply<Problem>(eval, pop_2);
	  pop_2.sort();
	  cout << "Initial Population 2\n" << pop_2 << endl;
  }
  
  if (getNodeRank()==3)
  {  
	  apply<Problem>(eval, pop_3);
	  pop_3.sort();
	  cout << "Initial Population 3\n" << pop_3 << endl;
  }
  
  //Topolgy
  RingTopology ring,topology;
  
  eoPeriodicContinue< Problem > mig_conti_1(  param.manGeneration );
  eoContinuator<Problem> mig_cont_1(mig_conti_1,pop_1);
  eoRandomSelect<Problem> mig_select_one_1;
  eoSelector <Problem, eoPop<Problem> > mig_select_1 (mig_select_one_1,param.nbMigrants,pop_1);
  eoPlusReplacement<Problem> replace_one_1;
  eoReplace <Problem, eoPop<Problem> > mig_replace_1 (replace_one_1,pop_1);
  peoAsyncIslandMig< eoPop< Problem >, eoPop< Problem > > mig_1 (mig_cont_1,mig_select_1,mig_replace_1,topology);
  checkpoint_1.add( mig_1 );
  
  eoPeriodicContinue< Problem > mig_conti_2(  param.manGeneration );
  eoContinuator<Problem> mig_cont_2(mig_conti_2,pop_2);
  eoRandomSelect<Problem> mig_select_one_2;
  eoSelector <Problem, eoPop<Problem> > mig_select_2 (mig_select_one_2,param.nbMigrants,pop_2);
  eoPlusReplacement<Problem> replace_one_2;
  eoReplace <Problem, eoPop<Problem> > mig_replace_2 (replace_one_2,pop_2);
  peoAsyncIslandMig< eoPop< Problem >, eoPop< Problem > > mig_2 (mig_cont_2,mig_select_2,mig_replace_2,topology);
  checkpoint_2.add( mig_2 );
  
  eoPeriodicContinue< Problem > mig_conti_3(  param.manGeneration );
  eoContinuator<Problem> mig_cont_3(mig_conti_3,pop_3);
  eoRandomSelect<Problem> mig_select_one_3;
  eoSelector <Problem, eoPop<Problem> > mig_select_3 (mig_select_one_3,param.nbMigrants,pop_3);
  eoPlusReplacement<Problem> replace_one_3;
  eoReplace <Problem, eoPop<Problem> > mig_replace_3 (replace_one_3,pop_3);
  peoAsyncIslandMig< eoPop< Problem >, eoPop< Problem > > mig_3 (mig_cont_3,mig_select_3,mig_replace_3,topology);
  checkpoint_3.add( mig_3 );
   
  peoWrapper parallelEA_1( gga_1, pop_1 );
  mig_1.setOwner( parallelEA_1 );
  
  peoWrapper parallelEA_2( gga_2, pop_2 );
  mig_2.setOwner( parallelEA_2 );

  peoWrapper parallelEA_3( gga_3, pop_3 );
  mig_3.setOwner( parallelEA_3 );

  peo :: run( );
  peo :: finalize( );
  endDebugging();
  // Print (sorted) intial population
  
  if (getNodeRank()==1)
  {
	  pop_1.sort();
	  cout << "FINAL Population 1\n" << pop_1 << endl;
	  cout << "Best solution found\t" << pop_1[0].fitness() << endl;
  }
  
  if (getNodeRank()==2)
  {
	  pop_2.sort();
	  cout << "FINAL Population 2\n" << pop_2 << endl;
	  cout << "Best solution found\t" << pop_2[0].fitness() << endl;
  }
  
  if (getNodeRank()==3)
  {
	  pop_3.sort();
	  cout << "FINAL Population 3\n" << pop_3 << endl;
	  cout << "Best solution found\t" << pop_3[0].fitness() << endl;
  }
  
  if (getNodeRank()==0)
  {
	  cout << "\nInstance size = " << n << endl;
	  cout << "Best known value in the litterature = " << bkv <<"\n"<< endl;
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
