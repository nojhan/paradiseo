// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "make_eval_FlowShop.h"

// (c) OPAC Team, LIFL, March 2006

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

#ifndef _make_eval_FlowShop_h
#define _make_eval_FlowShop_h

#include "FlowShop.h"
#include "FlowShopBenchmarkParser.h"
#include "FlowShopEval.h"
// also need the parser and param includes
#include <utils/eoParser.h>
#include <utils/eoState.h>


/*
 * This function creates an eoEvalFuncCounter<eoFlowShop> that can later be used to evaluate an individual.
 * @param eoParser& _parser  to get user parameters
 * @param eoState& _state  to store the memory 
 */
eoEvalFuncCounter < FlowShop > &do_make_eval (eoParser & _parser,
					      eoState & _state)
{

  // benchmark file name
  string benchmarkFileName =
    _parser.getORcreateParam (string (), "BenchmarkFile",
			      "Benchmark file name (benchmarks are available at "
			      + BENCHMARKS_WEB_SITE + ")", 'B',
			      "Representation", true).value ();
  if (benchmarkFileName == "")
    {
      std::string stmp = "*** Missing name of the benchmark file\n";
      stmp +=
	"    Type '-B=the_benchmark_file_name' or '--BenchmarkFile=the_benchmark_file_name'\n";
      stmp += "    Benchmarks files are available at " + BENCHMARKS_WEB_SITE;
      throw std::runtime_error (stmp.c_str ());
    }
  // reading of the parameters contained in the benchmark file
  FlowShopBenchmarkParser fParser (benchmarkFileName);
  unsigned M = fParser.getM ();
  unsigned N = fParser.getN ();
  std::vector < std::vector < unsigned > > p = fParser.getP ();
  std::vector < unsigned >d = fParser.getD ();

  // build of the initializer (a pointer, stored in the eoState)
  FlowShopEval *plainEval = new FlowShopEval (M, N, p, d);
  // turn that object into an evaluation counter
  eoEvalFuncCounter < FlowShop > *eval =
    new eoEvalFuncCounter < FlowShop > (*plainEval);
  // store in state
  _state.storeFunctor (eval);
  // and return a reference
  return *eval;
}

#endif
