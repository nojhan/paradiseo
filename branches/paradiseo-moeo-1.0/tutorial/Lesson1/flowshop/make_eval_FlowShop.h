// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_eval_FlowShop.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MAKE_EVAL_FLOWSHOP_H_
#define MAKE_EVAL_FLOWSHOP_H_


#include <utils/eoParser.h>
#include <utils/eoState.h>
#include <eoEvalFuncCounter.h>
#include <FlowShop.h>
#include <FlowShopBenchmarkParser.h>
#include <FlowShopEval.h>

/*
 * This function creates an eoEvalFuncCounter<eoFlowShop> that can later be used to evaluate an individual.
 * @param eoParser& _parser  to get user parameters
 * @param eoState& _state  to store the memory
 */
eoEvalFuncCounter<FlowShop> & do_make_eval(eoParser& _parser, eoState& _state)
{
    // benchmark file name
    std::string benchmarkFileName = _parser.getORcreateParam(std::string(), "BenchmarkFile", "Benchmark file name (benchmarks are available at www.lifl.fr/~liefooga/benchmarks)", 'B',"Representation", true).value();
    if (benchmarkFileName == "") {
        std::string stmp = "*** Missing name of the benchmark file\n";
        stmp += "    Type '-B=the_benchmark_file_name' or '--BenchmarkFile=the_benchmark_file_name'\n";
        stmp += "    Benchmarks files are available at www.lifl.fr/~liefooga/benchmarks";
        throw std::runtime_error(stmp.c_str());
    }
    // reading of the parameters contained in the benchmark file
    FlowShopBenchmarkParser fParser(benchmarkFileName);
    unsigned int M = fParser.getM();
    unsigned int N = fParser.getN();
    std::vector< std::vector<unsigned int> > p = fParser.getP();
    std::vector<unsigned int> d = fParser.getD();
    // build of the initializer (a pointer, stored in the eoState)
    FlowShopEval* plainEval = new FlowShopEval(M, N, p, d);
    // turn that object into an evaluation counter
    eoEvalFuncCounter<FlowShop>* eval = new eoEvalFuncCounter<FlowShop> (* plainEval);
    // store in state
    _state.storeFunctor(eval);
    // and return a reference
    return *eval;
}

#endif /*MAKE_EVAL_FLOWSHOP_H_*/
