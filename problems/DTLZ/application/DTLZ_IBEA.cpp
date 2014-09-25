// moeo general include

#include <iostream>
#include <fstream>
#include <moeo>
#include <es/eoRealInitBounded.h>
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
#include <DTLZ.h>
#include <DTLZ1Eval.h>
#include <DTLZ2Eval.h>
#include <DTLZ3Eval.h>
#include <DTLZ4Eval.h>
#include <DTLZ5Eval.h>
#include <DTLZ6Eval.h>
#include <DTLZ7Eval.h>
#include <SBXCrossover.h>
#include <PolynomialMutation.h>

using namespace std;

int main(int argc, char* argv[])
{
    try
    {

        eoParser parser(argc, argv);  // for user-parameter reading
        eoState state;                // to keep all things allocated

        unsigned int MAX_GEN = parser.createParam((unsigned int)(10000), "maxGen", "Maximum number of generations",'G',"Param").value();
        double P_CROSS = parser.createParam(1.0, "pCross", "Crossover probability",'C',"Param").value();
        double EXT_P_MUT = parser.createParam(1.0, "extPMut", "External Mutation probability",'E',"Param").value();
        double INT_P_MUT = parser.createParam(0.083, "intPMut", "Internal Mutation probability",'I',"Param").value();
        unsigned int VEC_SIZE = parser.createParam((unsigned int)(12), "vecSize", "Genotype Size",'V',"Param").value();
        unsigned int NB_OBJ= parser.createParam((unsigned int)(3), "nbObj", "Number of Objective",'N',"Param").value();
        std::string OUTPUT_FILE = parser.createParam(std::string("dtlz_ibea"), "outputFile", "Path of the output file",'o',"Output").value();
        unsigned int EVAL = parser.createParam((unsigned int)(1), "eval", "Number of the DTLZ evaluation fonction",'F',"Param").value();
        unsigned int DTLZ4_PARAM = parser.createParam((unsigned int)(100), "dtlz4_param", "Parameter of the DTLZ4 evaluation fonction",'P',"Param").value();
        unsigned int NB_EVAL = parser.createParam((unsigned int)(0), "nbEval", "Number of evaluation before Stop",'P',"Param").value();
        unsigned int TIME = parser.createParam((unsigned int)(0), "time", "Time(seconds) before Stop",'T',"Param").value();



        /*** the representation-dependent things ***/
        std::vector <bool> bObjectives(NB_OBJ);
        for (unsigned int i=0; i<NB_OBJ ; i++)
            bObjectives[i]=true;
        moeoObjectiveVectorTraits::setup(NB_OBJ,bObjectives);

        // The fitness evaluation
        eoEvalFunc <DTLZ> * eval;

        if (EVAL == 1)
            eval= new DTLZ1Eval;
        else if (EVAL == 2)
            eval= new DTLZ2Eval;
        else if (EVAL == 3)
            eval= new DTLZ3Eval;
        else if (EVAL == 4)
            eval= new DTLZ4Eval(DTLZ4_PARAM);
        else if (EVAL == 5)
            eval= new DTLZ5Eval;
        else if (EVAL == 6)
            eval= new DTLZ6Eval;
        else if (EVAL == 7)
            eval= new DTLZ7Eval;

        // the genotype (through a genotype initializer)
        eoRealVectorBounds bounds(VEC_SIZE, 0.0, 1.0);

        eoRealInitBounded <DTLZ> init (bounds);
        // the variation operators
        SBXCrossover < DTLZ > xover(bounds, 15);

        PolynomialMutation < DTLZ > mutation (bounds, INT_P_MUT, 20);

        /*** the representation-independent things ***/

        // initialization of the population

        // definition of the archive
        // stopping criteria

        eoGenContinue<DTLZ> term(MAX_GEN);

        eoEvalFuncCounter<DTLZ> evalFunc(*eval);

        /*eoTimeContinue<DTLZ> timeContinuator(TIME);
        eoCheckPoint<DTLZ> checkpoint(timeContinuator);*/

        eoCheckPoint<DTLZ>* checkpoint;

        if (TIME > 0)
            checkpoint = new eoCheckPoint<DTLZ>(*(new eoTimeContinue<DTLZ>(TIME)));
        else if (NB_EVAL > 0)
            checkpoint = new eoCheckPoint<DTLZ>(*(new eoEvalContinue<DTLZ>(evalFunc, NB_EVAL)));
        else {
            cout << "ERROR!!! : TIME or NB_EVAL must be > 0 : used option --time or --nbEval\n";
            return EXIT_FAILURE;
        }

        checkpoint->add(term);

        /*moeoArchiveObjectiveVectorSavingUpdater < DTLZ > updater(arch, OUTPUT_FILE);
        checkpoint->add(updater);*/

        // algorithm

        eoSGAGenOp < DTLZ > op(xover, P_CROSS, mutation, EXT_P_MUT);

        /*  moeoArchiveUpdater < DTLZ > up(arch, pop);
                checkpoint.add(up);*/

        //moeoNSGAII<DTLZ> algo(*checkpoint, *eval ,op);

        moeoAdditiveEpsilonBinaryMetric < DTLZObjectiveVector > metric;
        moeoIBEA<DTLZ> algo(*checkpoint, evalFunc ,op, metric);


        /*** Go ! ***/

        // help ?


        eoPop<DTLZ>& pop = do_make_pop(parser, state, init);

        make_help(parser);

        // run the algo
        do_run(algo, pop);

        moeoUnboundedArchive<DTLZ> finalArchive;
        finalArchive(pop);

        // printing of the final population
        //cout << "Final Archive \n";
        //finalArchive.sortedPrintOn(outfile);

        ofstream outfile(OUTPUT_FILE.c_str(), ios::app);
        if ((unsigned int)outfile.tellp() != 0)
            outfile << endl;

        for (unsigned int i=0 ; i < finalArchive.size(); i++) {
            for (unsigned int j=0 ; j<NB_OBJ; j++) {
                outfile << finalArchive[i].objectiveVector()[j];
                if (j != NB_OBJ -1)
                    outfile << " ";
            }
            outfile << endl;
        }

        outfile.close();

    }
    catch (exception& e)
    {
        cout << e.what() << endl;
    }
    return EXIT_SUCCESS;
}
