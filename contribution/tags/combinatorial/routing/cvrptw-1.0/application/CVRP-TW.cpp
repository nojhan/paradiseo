/*
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2007
 *
 * (c) Antonio LaTorre <atorre@fi.upm.es>, 2007
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

// Miscellaneous includes and declarations
#include <iostream>
#include <time.h>

// eo general include
#include "eo"
// the real bounds (not yet in general eo include)
#include "utils/eoRealVectorBounds.h"

// Include here whatever specific files for your representation.
// Basically, this should include at least the following:

/** definition of representation:
  * class eoVRP MUST derive from EO<FitT> for some fitness
  */
#include "eoVRP.h"

/** definition of initilizqtion:
  * class eoVRPInit MUST derive from eoInit<eoVRP>
  */
#include "eoVRPInit.h"

/** definition of evaluation:
  * class eoVRPEvalFunc MUST derive from eoEvalFunc<eoVRP>
  * and should test for validity before doing any computation
  * see tutorial/Templates/evalFunc.tmpl
  */
#include "eoVRPEvalFunc.h"

/** definitions of operators: write as many classes as types of operators
  * and include them here. In this simple example,
  * one crossover (2->2) and one mutation (1->1) operators are used
  */
#include "eoVRPQuadCrossover.h"
#include "eoVRPMutation.h"

/* And (possibly) your personal statistics */
#include "eoVRPStat.h"

#include "eoVRPUtils.h"


/* **********************************************************************************
   ********************************************************************************** */

// Use existing modules to define representation independent routines

// How to initialize the population
// It IS representation independent if an eoInit is given
#include "do/make_pop.h"

// The stopping criterion
#include "do/make_continue.h"

// Output (stats, population dumps, ...)
#include "do/make_checkpoint.h"

// Simply call to the algo. Stays there for consistency reasons
// No template for that one
#include "do/make_run.h"

// The instanciating fitnesses
#include <eoScalarFitness.h>

// Checks for help demand, and writes the status file
// and make_help; in libutils

void make_help (eoParser& _parser);


/* **********************************************************************************
   ********************************************************************************** */

/*
 * This function builds the algorithm (i.e. selection and replacement)
 *      from existing continue (or checkpoint) and operators
 *
 * It uses a parser (to get user parameters) and a state (to store the memory)
 * the last argument is an individual, needed for 2 reasons
 *     it disambiguates the call after instanciations
 *     some operator might need some private information about the indis
 *
 * This is why the template is the complete EOT even though only the fitness
 * is actually templatized here
*/


eoAlgo<eoVRP>& make_algo_scalar_transform (eoParser& _parser, eoState& _state, eoEvalFunc<eoVRP>& _eval, eoCheckPoint<eoVRP>& _checkpoint, eoTransform<eoVRP>& _transform, eoDistance<eoVRP>* _dist = NULL) {

    // the selection : help and comment depend on whether or not a distance is passed
    std::string comment;

    if (_dist == NULL)
        comment = "Selection: DetTour(T), StochTour(t), Roulette, Ranking(p,e) or Sequential(ordered/unordered)";
    else
        comment = "Selection: DetTour(T), StochTour(t), Roulette, Ranking(p,e), Sharing(sigma_share) or Sequential(ordered/unordered)";

    eoValueParam<eoParamParamType>& selectionParam = _parser.createParam (eoParamParamType ("DetTour(2)"), "selection", comment, 'S', "Evolution Engine");

    eoParamParamType& ppSelect = selectionParam.value (); // std::pair<std::string,std::vector<std::string> >

    eoSelectOne<eoVRP>* selectOne;

    if (ppSelect.first == std::string("DetTour")) {

        unsigned detSize;

        if (!ppSelect.second.size ()) { // no parameter added

            std::cerr << "WARNING, no parameter passed to DetTour, using 2" << std::endl;
            detSize = 2;
            // put back 2 in parameter for consistency (and status file)
            ppSelect.second.push_back (std::string ("2"));

        }
        else // parameter passed by user as DetTour(T)
            detSize = atoi (ppSelect.second [0].c_str ());

        selectOne = new eoDetTournamentSelect<eoVRP> (detSize);

    }
    else if (ppSelect.first == std::string ("Sharing")) {

        double nicheSize;

        if (!ppSelect.second.size ()) { // no parameter added

            std::cerr << "WARNING, no parameter passed to Sharing, using 0.5" << std::endl;
            nicheSize = 0.5;
            // put back 2 in parameter for consistency (and status file)
            ppSelect.second.push_back (std::string ("0.5"));

        }
        else // parameter passed by user as DetTour(T)
            nicheSize = atof (ppSelect.second [0].c_str ());

        if (_dist == NULL) // no distance
            throw std::runtime_error ("You didn't specify a distance when calling make_algo_scalar and using sharing");

        selectOne = new eoSharingSelect<eoVRP> (nicheSize, *_dist);

    }
    else if (ppSelect.first == std::string ("StochTour")) {

        double p;

        if (!ppSelect.second.size ()) { // no parameter added

            std::cerr << "WARNING, no parameter passed to StochTour, using 1" << std::endl;
            p = 1;
            // put back p in parameter for consistency (and status file)
            ppSelect.second.push_back (std::string ("1"));

        }
        else // parameter passed by user as DetTour(T)
            p = atof (ppSelect.second [0].c_str ());

        selectOne = new eoStochTournamentSelect<eoVRP> (p);

    }
    else if (ppSelect.first == std::string ("Ranking")) {

        double p,e;

        if (ppSelect.second.size () == 2) { // 2 parameters: pressure and exponent

            p = atof (ppSelect.second [0].c_str ());
            e = atof (ppSelect.second [1].c_str ());

        }
        else if (ppSelect.second.size () == 1) { // 1 parameter: pressure

            std::cerr << "WARNING, no exponent to Ranking, using 1" << std::endl;
            e = 1;
            ppSelect.second.push_back (std::string ("1"));
            p = atof (ppSelect.second [0].c_str ());

        }
        else { // no parameters ... or garbage

            std::cerr << "WARNING, no parameter to Ranking, using (2,1)" << std::endl;
            p = 2;
            e = 1;
            // put back in parameter for consistency (and status file)
            ppSelect.second.resize (2); // just in case
            ppSelect.second [0] = (std::string ("2"));
            ppSelect.second [1] = (std::string ("1"));

        }

        // check for authorized values
        // pressure in (0,1]
        if ((p <= 1) || (p > 2)) {

            std::cerr << "WARNING, selective pressure must be in (0,1] in Ranking, using 2\n";
            p = 2;
            ppSelect.second [0] = (std::string ("2"));

        }

        // exponent >0
        if (e <= 0) {

            std::cerr << "WARNING, exponent must be positive in Ranking, using 1\n";
            e = 1;
            ppSelect.second [1] = (std::string ("1"));

        }

        // now we're OK
        eoPerf2Worth<eoVRP>& p2w = _state.storeFunctor (new eoRanking<eoVRP> (p,e));
        selectOne = new eoRouletteWorthSelect<eoVRP> (p2w);

    }
    else if (ppSelect.first == std::string ("Sequential")) { // one after the other

        bool b;

        if (ppSelect.second.size () == 0) { // no argument -> default = ordered

            b = true;
            // put back in parameter for consistency (and status file)
            ppSelect.second.push_back (std::string ("ordered"));

        }
        else
            b = !(ppSelect.second [0] == std::string ("unordered"));

        selectOne = new eoSequentialSelect<eoVRP> (b);

    }
    else if (ppSelect.first == std::string ("Roulette")) { // no argument (yet)

        selectOne = new eoProportionalSelect <eoVRP>;

    }
    else if (ppSelect.first == std::string ("Random")) { // no argument

        selectOne = new eoRandomSelect<eoVRP>;

    }
    else {

        std::string stmp = std::string ("Invalid selection: ") + ppSelect.first;
        throw std::runtime_error (stmp.c_str ());

    }

    _state.storeFunctor (selectOne);

    // Modified from original
    eoSelectPerc<eoVRP>* select = new eoSelectPerc<eoVRP> (*selectOne);
    _state.storeFunctor (select);

    // the number of offspring
    eoValueParam<eoHowMany>& offspringRateParam =  _parser.createParam (eoHowMany (1.0), "nbOffspring", "Nb of offspring (percentage or absolute)", 'O', "Evolution Engine");

    // the replacement
    eoValueParam<eoParamParamType>& replacementParam = _parser.createParam (eoParamParamType ("Comma"), "replacement", "Replacement: Comma, Plus or EPTour(T), SSGAWorst, SSGADet(T), SSGAStoch(t)", 'R', "Evolution Engine");

    eoParamParamType& ppReplace = replacementParam.value (); // std::pair<std::string,std::vector<std::string> >

    eoReplacement<eoVRP>* replace;

    if (ppReplace.first == std::string ("Comma")) { // Comma == generational

        replace = new eoCommaReplacement<eoVRP>;

    }
    else if (ppReplace.first == std::string ("Plus")) {

        replace = new eoPlusReplacement<eoVRP>;

    }
    else if (ppReplace.first == std::string ("EPTour")) {

        unsigned detSize;

        if (!ppReplace.second.size ()) { // no parameter added

            std::cerr << "WARNING, no parameter passed to EPTour, using 6" << std::endl;
            detSize = 6;
            // put back in parameter for consistency (and status file)
            ppReplace.second.push_back (std::string ("6"));

        }
        else // parameter passed by user as EPTour(T)
            detSize = atoi (ppSelect.second [0].c_str ());

        replace = new eoEPReplacement<eoVRP> (detSize);

    }
    else if (ppReplace.first == std::string ("SSGAWorst")) {

        replace = new eoSSGAWorseReplacement<eoVRP>;

    }
    else if (ppReplace.first == std::string ("SSGADet")) {

        unsigned detSize;

        if (!ppReplace.second.size ()) { // no parameter added

            std::cerr << "WARNING, no parameter passed to SSGADet, using 2" << std::endl;
            detSize = 2;
            // put back in parameter for consistency (and status file)
            ppReplace.second.push_back (std::string ("2"));

        }
        else // parameter passed by user as SSGADet(T)
            detSize = atoi (ppSelect.second [0].c_str ());

        replace = new eoSSGADetTournamentReplacement<eoVRP> (detSize);

    }
    else if (ppReplace.first == std::string ("SSGAStoch")) {

        double p;

        if (!ppReplace.second.size ()) { // no parameter added

            std::cerr << "WARNING, no parameter passed to SSGAStoch, using 1" << std::endl;
            p = 1;
            // put back in parameter for consistency (and status file)
            ppReplace.second.push_back (std::string ("1"));

        }
        else // parameter passed by user as SSGADet(T)
            p = atof (ppSelect.second [0].c_str ());

        replace = new eoSSGAStochTournamentReplacement<eoVRP> (p);

    }
    else {

        std::string stmp = std::string ("Invalid replacement: ") + ppReplace.first;
        throw std::runtime_error (stmp.c_str ());

    }

    _state.storeFunctor (replace);

    // adding weak elitism
    eoValueParam<bool>& weakElitismParam =  _parser.createParam (false, "weakElitism", "Old best parent replaces new worst offspring *if necessary*", 'w', "Evolution Engine");

    if (weakElitismParam.value ()) {

        eoReplacement<eoVRP>* replaceTmp = replace;
        replace = new eoWeakElitistReplacement<eoVRP> (*replaceTmp);
        _state.storeFunctor (replace);

    }



    eoSelectTransform<eoVRP>* selectTransform = new eoSelectTransform<eoVRP> (*select, _transform);
    _state.storeFunctor (selectTransform);

    eoTimeVaryingLoopEval<eoVRP>* popEval = new eoTimeVaryingLoopEval<eoVRP> (_eval);
    _state.storeFunctor (popEval);

    // now the eoEasyEA (Modified)
    eoAlgo<eoVRP>* ga = new eoEasyEA<eoVRP> (_checkpoint, *popEval, *selectTransform, *replace);
    _state.storeFunctor (ga);

    // that's it!
    return *ga;

}


/* **********************************************************************************
   ********************************************************************************** */

// Now use all of the above, + representation dependent things
int main (int argc, char* argv []) {

    try {

        // ////////////////////// //
        // User parameter reading //
        // ////////////////////// //

        eoParser parser (argc, argv);

        // Parameter for loading a problem instance
        eoValueParam<std::string> instanceParam ("", "instance", "Instance to be loaded");
        parser.processParam (instanceParam, "Problem params");
        std::string instance = instanceParam.value ();

        // We try to load an instance of the VRP problem
        eoVRPUtils::load (instance.c_str ());

        // Initialization of random seed
        rng.reseed (time (0));

        // ////////////////////////// //
        // Keeps all things allocated //
        // ////////////////////////// //

        eoState state;


        // ///////////////////// //
        // The fitness evaluator //
        // ///////////////////// //

        eoVRPEvalFunc plainEval;

        // Turn that object into an evaluation counter
        eoEvalFuncCounter<eoVRP> eval (plainEval);


        // ////////////////////// //
        // A genotype initializer //
        // ////////////////////// //

        eoVRPInit init;


        // ///////////////////////////////////////////////////// //
        // Build the variation operator (any seq/prop construct) //
        // ///////////////////////////////////////////////////// //

        // A (first) crossover
        eoVRPGenericCrossover cross;

        // A (first) mutation
        eoVRPMutation mut;

        // First read the individual level parameters
        double pCross = parser.createParam (0.6, "pCross", "Probability of Crossover", 'C', "Variation Operators" ).value ();

        // Minimum check
        if ((pCross < 0) || (pCross > 1))
            throw std::runtime_error ("Invalid pCross");

        double pMut = parser.createParam (0.1, "pMut", "Probability of Mutation", 'M', "Variation Operators" ).value ();

        // Minimum check
        if ((pMut < 0) || (pMut > 1))
            throw std::runtime_error ("Invalid pMut");

        // Now create the transform operator
        eoPropCombinedQuadOp<eoVRP> xover (cross, 1.0);
        eoPropCombinedMonOp<eoVRP> mutation (mut, 1.0);
        eoSGATransform<eoVRP> transform (xover, pCross, mutation, pMut);


        // ////////////////////////////////////////////// //
        // Now some representation-independent things     //
        // (no need to modify anything beyond this point) //
        // ////////////////////////////////////////////// //

        // Initialize the population
        eoPop<eoVRP>& pop = do_make_pop (parser, state, init);

        // Stopping criteria
        eoContinue<eoVRP>& term = do_make_continue (parser, state, eval);

        // Output
        eoCheckPoint<eoVRP>& checkpoint = do_make_checkpoint (parser, state, eval, term);


        // ////////// //
        // Statistics //
        // ////////// //

        eoVRPStat myStat;
        checkpoint.add (myStat);

        // This one is probably redundant with the one in make_checkpoint, but w.t.h.
        eoIncrementorParam<unsigned> generationCounter ("Gen.");
        checkpoint.add (generationCounter);

        // Need to get the name of the redDir param (if any)
        std::string dirName = parser.getORcreateParam (std::string ("Res"), "resDir", "Directory to store DISK outputs", '\0', "Output - Disk").value () + "/";

        // Those need to be pointers because of the if's
        eoStdoutMonitor* myStdOutMonitor;
        eoFileMonitor*   myFileMonitor;

#ifdef HAVE_GNUPLOT
        eoGnuplot1DMonitor* myGnuMonitor;
#endif

        // Now check how you want to output the stat:
        bool printVRPStat = parser.createParam (false, "coutVRPStat", "Prints my stat to screen, one line per generation", '\0', "My application").value ();
        bool fileVRPStat = parser.createParam (false, "fileVRPStat", "Saves my stat to file (in resDir", '\0', "My application").value ();
        bool plotVRPStat = parser.createParam (false, "plotVRPStat", "On-line plots my stat using gnuplot", '\0', "My application").value ();

        // Should we write it on StdOut ?
        if (printVRPStat) {

            myStdOutMonitor = new eoStdoutMonitor (false);

            // Don't forget to store the memory in the state
            state.storeFunctor (myStdOutMonitor);

            // And of course to add the monitor to the checkpoint
            checkpoint.add (*myStdOutMonitor);

            // And the different fields to the monitor
            myStdOutMonitor->add (generationCounter);
            myStdOutMonitor->add (eval);
            myStdOutMonitor->add (myStat);

        }

        // First check the directory (and creates it if not exists already):
        if (fileVRPStat || plotVRPStat)
            if (!testDirRes (dirName, true))
                throw std::runtime_error ("Problem with resDir");

        // Should we write it to a file ?
        if (fileVRPStat) {

            // The file name is hard-coded - of course you can read
            // a string parameter in the parser if you prefer
            myFileMonitor = new eoFileMonitor (dirName + "myStat.xg");

            // Don't forget to store the memory in the state
            state.storeFunctor (myFileMonitor);

            // And of course to add the monitor to the checkpoint
            checkpoint.add (*myFileMonitor);

            // And the different fields to the monitor
            myFileMonitor->add (generationCounter);
            myFileMonitor->add (eval);
            myFileMonitor->add (myStat);

        }

#ifdef HAVE_GNUPLOT

        // Should we PLOT it on StdOut ? (one dot per generation, incremental plot)
        if (plotVRPStat) {

            myGnuMonitor = new eoGnuplot1DMonitor (dirName + "plot_myStat.xg", minimizing_fitness<RouteSet> ());
            // NOTE: you can send commands to gnuplot at any time with the method
            // myGnuMonitor->gnuplotCommand(string)
            // par exemple, gnuplotCommand("set logscale y")

            // Don't forget to store the memory in the state
            state.storeFunctor (myGnuMonitor);

            // And of course to add the monitor to the checkpoint
            checkpoint.add (*myGnuMonitor);

            // And the different fields to the monitor (X = eval, Y = myStat)
            myGnuMonitor->add (eval);
            myGnuMonitor->add (myStat);

        }

#endif


        // ///////////////////////////// //
        // Construction of the algorithm //
        // ///////////////////////////// //

        // Algorithm (need the operator!)
        eoAlgo<eoVRP>& ga = make_algo_scalar_transform (parser, state, eval, checkpoint, transform);


        // /////////////////////////////////////////////////// //
        // To be called AFTER all parameters have been read!!! //
        // /////////////////////////////////////////////////// //

        make_help (parser);


        // //////////////////// //
        // Launch the algorithm //
        // /////////////////// //

        // Evaluate intial population AFTER help and status in case it takes time
        apply<eoVRP> (eval, pop);

        // Run the GA
        do_run (ga, pop);

        std::cout << "Solution:" << std::endl;
        pop.best_element ().printAllOn (std::cout);
        std::cout << std::endl;


    }
    catch (std::exception& e) {

        std::cerr << e.what () << std::endl;

    }

    return 0;

}
