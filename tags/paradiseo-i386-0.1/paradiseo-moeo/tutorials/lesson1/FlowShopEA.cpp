// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "FlowShopEA.cpp"

// (c) OPAC Team, LIFL, October 2006

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: Arnaud.Liefooghe@lifl.fr
*/


// Miscilaneous include and declaration 
using namespace std;


/* EO */
// eo general include
#include "eo"
// for the creation of an evaluator
#include "make_eval_FlowShop.h"
// for the creation of an initializer
#include "make_genotype_FlowShop.h"
// for the creation of the variation operators
#include "make_op_FlowShop.h"
// how to initialize the population
#include <do/make_pop.h>
// the stopping criterion
#include <do/make_continue_pareto.h>
// outputs (stats, population dumps, ...)
#include <do/make_checkpoint_pareto.h>
// simple call to the algo
#include <do/make_run.h>

// checks for help demand, and writes the status file and make_help; in libutils
void make_help (eoParser & _parser);


/* MOEO */
#include <moeoArchive.h>
#include <moeoArchiveUpdater.h>
#include <moeoArchiveFitnessSavingUpdater.h>
#include <metric/moeoContributionMetric.h>
#include <metric/moeoEntropyMetric.h>
#include <metric/moeoBinaryMetricSavingUpdater.h>
// evolution engine (selection and replacement)
#include <old/make_algo_MOEO.h>

/* FLOW-SHOP */
// definition of representation
#include "FlowShop.h"
// definition of fitness
#include "FlowShopFitness.h"



int
main (int argc, char *argv[])
{
  try
  {

    eoParser parser (argc, argv);	// for user-parameter reading
    eoState state;		// to keep all things allocated





    /*** the representation-dependent things ***/

    // The fitness evaluation
    eoEvalFuncCounter < FlowShop > &eval = do_make_eval (parser, state);
    // the genotype (through a genotype initializer)
    eoInit < FlowShop > &init = do_make_genotype (parser, state);
    // the variation operators
    eoGenOp < FlowShop > &op = do_make_op (parser, state);





    /*** the representation-independent things ***/

    // initialization of the population   
    eoPop < FlowShop > &pop = do_make_pop (parser, state, init);
    // definition of the archive
    moeoArchive < FlowShop > arch;
    // stopping criteria
    eoContinue < FlowShop > &term =
      do_make_continue_pareto (parser, state, eval);
    // output
    eoCheckPoint < FlowShop > &checkpoint =
      do_make_checkpoint_pareto (parser, state, eval, term);
    // algorithm    
    eoAlgo < FlowShop > &algo =
      do_make_algo_MOEO (parser, state, eval, checkpoint, op, arch);





    /*** archive-related features ***/
    // update the archive every generation   
    moeoArchiveUpdater < FlowShop > updater (arch, pop);
    checkpoint.add (updater);
    // save the archive every generation in 'Res/Arch*'
    moeoArchiveFitnessSavingUpdater < FlowShop > save_updater (arch);
    checkpoint.add (save_updater);
    // save the contribution of the non-dominated solutions in 'Res/Contribution.txt'
    moeoVectorVsVectorBM < FlowShop, double >*contribution =
      new moeoContributionMetric < FlowShop > ();
    moeoBinaryMetricSavingUpdater < FlowShop >
      contribution_updater (*contribution, arch, "Res/Contribution.txt");
    checkpoint.add (contribution_updater);
    // save the entropy of the non-dominated solutions in 'Res/Entropy.txt'
    moeoVectorVsVectorBM < FlowShop, double >*entropy =
      new moeoEntropyMetric < FlowShop > ();
    moeoBinaryMetricSavingUpdater < FlowShop > entropy_updater (*entropy,
								arch,
								"Res/Entropy.txt");
    checkpoint.add (entropy_updater);




    /*** Go ! ***/

    // help ?
    make_help (parser);

    // first evalution
    apply < FlowShop > (eval, pop);

    // printing of the initial population
    cout << "Initial Population\n";
    pop.sortedPrintOn (cout);
    cout << endl;

    // run the algo
    do_run (algo, pop);

    // printing of the final population
    cout << "Final Population\n";
    pop.sortedPrintOn (cout);
    cout << endl;

    // printing of the final archive
    cout << "Final Archive\n";
    arch.sortedPrintOn (cout);
    cout << endl;

  } catch (exception & e)
  {
    cout << e.what () << endl;
  }
  return EXIT_SUCCESS;
}
