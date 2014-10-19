#include <iostream>

#include <paradiseo/eo.h>
/*
#include <paradiseo/eo/eoScalarFitness.h>
#include <paradiseo/eo/eoVector.h>
#include <paradiseo/eo/eoPop.h>
#include <paradiseo/eo/utils/eoParser.h>
#include <paradiseo/eo/utils/eoCheckPoint.h>
#include <paradiseo/eo/eoEvalFuncPtr.h>

#include <paradiseo/eo/eoGenContinue.h>
#include <paradiseo/eo/eoFitContinue.h>
#include <paradiseo/eo/utils/eoStdoutMonitor.h>
#include <paradiseo/eo/utils/eoStat.h>
#include <paradiseo/eo/utils/eoTimedMonitor.h>

#include <paradiseo/eo/eoMergeReduce.h>
#include <paradiseo/eo/eoEasyEA.h>

#include <paradiseo/eo/es/CMAState.h>
#include <paradiseo/eo/es/CMAParams.h>
#include <paradiseo/eo/es/eoCMAInit.h>
#include <paradiseo/eo/es/eoCMABreed.h>
*/

using namespace eo;
using namespace std;

typedef eoMinimizingFitness FitT;
typedef eoVector<FitT, double> EoType;

double sqr(double x) { return x*x; }

eoValueParam<int> evals(0,"Function Evals","Number of Evaluations");

double f_sphere(const vector<double>& values) {
    double sum = 0.0;
    for (unsigned i = 0; i < values.size(); ++i) {
	sum += values[i] * values[i];
    }
    ++evals.value();
    return sum;
}

double f_rosen(const vector<double>& x) {
    double sum =0.0;

    for (unsigned i = 0; i < x.size()-1; ++i) {
	sum += 100 * sqr(sqr(x[i])-x[i+1]) + sqr(1.-x[i]);
    }
    ++evals.value();
    return sum;
}



int main(int argc, char* argv[]) {

    // make sure we have a dimensionality parameter (for testing)
    char** rargv = new char*[argc+1];
    rargv[0] = argv[0];
    rargv[1] = (char*)"-N10";
    for (int i = 2; i < argc; ++i) {
	rargv[i] = argv[i-1];
    }

    eoParser parser(argc+1, rargv);

    CMAParams params(parser);

    vector<double> initial_point(params.n, 0.0);

    CMAState state(params, initial_point);

    if (parser.userNeedsHelp())
    {
	parser.printHelp(std::cout);
	return 1;
    }

    eoCMAInit<FitT> init(state);

    eoPop<EoType> pop(params.mu, init);

    eoEvalFuncPtr<EoType, double, const vector<double>&> eval(  f_rosen );

    eoCMABreed<FitT> breed(state, params.lambda);

    for (unsigned i = 0; i < pop.size(); ++i) {
	eval(pop[i]);
    }

    eoCommaReplacement<EoType> comma;

    eoGenContinue<EoType> gen(params.maxgen);
    eoFitContinue<EoType> fit(1e-10);

    eoCheckPoint<EoType> checkpoint(gen);
    checkpoint.add(fit);

    eoBestFitnessStat<EoType> stat;

    eoStdoutMonitor mon;
    mon.add(stat);
    mon.add(evals);

    eoTimedMonitor timed(1);// 1 seconds
    timed.add(mon); // wrap it

    checkpoint.add(timed);
    checkpoint.add(stat);

    eoEasyEA<EoType> algo(
	    checkpoint,
	    eval,
	    breed,
	    comma);


    algo(pop);
    pop.sort();

    cout << pop[0] << endl;
    cout << "Fitness achieved = " << pop[0].fitness() << endl;
    cout << "Function evaluations = " << evals.value() << endl;
}
