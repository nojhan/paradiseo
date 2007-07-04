// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopEA.cpp
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------


// moeo general include
#include <moeo>
// for the creation of an evaluator
#include <make_eval_FlowShop.h>
// for the creation of an initializer
#include <make_genotype_FlowShop.h>
// for the creation of the variation operators
#include <make_op_FlowShop.h>
// how to initialize the population
#include <do/make_pop.h>
// the stopping criterion
#include <do/make_continue_moeo.h>
// outputs (stats, population dumps, ...)
#include <do/make_checkpoint_moeo.h>
// evolution engine (selection and replacement)
#include <do/make_ea_moeo.h>
// simple call to the algo
#include <do/make_run.h>
// checks for help demand, and writes the status file and make_help; in libutils
void make_help(eoParser & _parser);
// definition of the representation
#include <FlowShop.h>


using namespace std;


int main(int argc, char* argv[])
{
    try
    {
    
        eoParser parser(argc, argv);  // for user-parameter reading
        eoState state;                // to keep all things allocated


        /*** the representation-dependent things ***/

        // The fitness evaluation
        eoEvalFuncCounter<FlowShop>& eval = do_make_eval(parser, state);
        // the genotype (through a genotype initializer)
        eoInit<FlowShop>& init = do_make_genotype(parser, state);
        // the variation operators
        eoGenOp<FlowShop>& op = do_make_op(parser, state);


        /*** the representation-independent things ***/

        // initialization of the population
        eoPop<FlowShop>& pop = do_make_pop(parser, state, init);
        // definition of the archive
        moeoArchive<FlowShop> arch;
        // stopping criteria
        eoContinue<FlowShop>& term = do_make_continue_moeo(parser, state, eval);
        // output
        eoCheckPoint<FlowShop>& checkpoint = do_make_checkpoint_moeo(parser, state, eval, term, pop, arch);
        // algorithm
        eoAlgo<FlowShop>& algo = do_make_ea_moeo(parser, state, eval, checkpoint, op, arch);


        /*** Go ! ***/

        // help ?
        make_help(parser);

        // first evalution
        apply<FlowShop>(eval, pop);

        // printing of the initial population
        cout << "Initial Population\n";
        pop.sortedPrintOn(cout);
        cout << endl;

        // run the algo
        do_run(algo, pop);

        // printing of the final population
        cout << "Final Population\n";
        pop.sortedPrintOn(cout);
        cout << endl;

        // printing of the final archive
        cout << "Final Archive\n";
        arch.sortedPrintOn(cout);
        cout << endl;


    }
    catch (exception& e)
    {
        cout << e.what() << endl;
    }
    return EXIT_SUCCESS;
}
