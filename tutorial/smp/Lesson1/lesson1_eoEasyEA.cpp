/*
<lesson1_eoEasyEA.cpp>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Alexandre Quemy, Thibault Lasnier - INSA Rouen

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  ue,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

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
#include <paradiseo/smp.h>
#include "QAP.h"
#include "QAPGA.h"
#include <string>
using namespace std;
using namespace paradiseo::smp;

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

int main(int argc, char **argv)
{
    if (argc < 2){
        cout << "Please give a param file" << endl;
        exit(1);
    }
       
    eoParser parser(argc, argv);
    parameters param;
    parseFile(parser, param);
    rng.reseed(param.seed);
      
    // Reading the a and b matrices of the QAP problem
    loadInstances(param.inst.c_str(), n, bkv, a, b);

    // Declaration of class wrapping the evaluation function of the QAP
    ProblemEvalFunc plainEval;
    eoEvalFuncCounter<Problem> eval(plainEval);
     
    // Class involving a simple call to the function of initialisation of a solution
    ProblemInit chromInit;

    eoPop<Problem> pop(param.popSize, chromInit);  // Initialise the population

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
      
    try
    {
        // Create the algorithm
        MWModel<eoEasyEA,Problem> mw(genCont, plainEval, select, transform, replace);
        // Start a parallel evaluation on the population
        mw.evaluate(pop);
        std::cout << "Initial population :" << std::endl;
        pop.sort();
        std::cout << pop << std::endl;
        // Start the algorithm on the population
        mw(pop);
        std::cout << "Final population :" << std::endl;
        pop.sort();
        std::cout << pop << std::endl;
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
