// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_genotype_FlowShop.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MAKE_GENOTYPE_FLOWSHOP_H_
#define MAKE_GENOTYPE_FLOWSHOP_H_

#include <utils/eoParser.h>
#include <utils/eoState.h>
#include "FlowShop.h"
#include "FlowShopInit.h"
#include "FlowShopBenchmarkParser.h"

/*
 * This function creates an eoInit<eoFlowShop> that can later be used to initialize the population (see make_pop.h).
 * @param eoParser& _parser  to get user parameters
 * @param eoState& _state  to store the memory
 */
eoInit<FlowShop> & do_make_genotype(eoParser& _parser, eoState& _state) {
 
  // benchmark file name
  string benchmarkFileName = _parser.getORcreateParam(string(), "BenchmarkFile", "Benchmark file name (benchmarks are available at " + BENCHMARKS_WEB_SITE + ")", 'B',"Representation", true).value();
  if (benchmarkFileName == "") {
    std::string stmp = "*** Missing name of the benchmark file\n";
    stmp += "   Type '-B=the_benchmark_file_name' or '--BenchmarkFile=the_benchmark_file_name'\n";
    stmp += "   Benchmarks files are available at " + BENCHMARKS_WEB_SITE;
    throw std::runtime_error(stmp.c_str());
  }
  // reading of number of jobs to schedule contained in the benchmark file   
  FlowShopBenchmarkParser fParser(benchmarkFileName);
  unsigned N = fParser.getN();

  // build of the initializer (a pointer, stored in the eoState)
  eoInit<FlowShop>* init = new FlowShopInit(N);
  // store in state
  _state.storeFunctor(init);
  // and return a reference
  return *init;
}

#endif /*MAKE_GENOTYPE_FLOWSHOP_H_*/
