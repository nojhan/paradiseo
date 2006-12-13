// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "make_genotype_FlowShop.h"

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

#ifndef _make_genotype_FlowShop_h
#define _make_genotype_FlowShop_h

#include "FlowShop.h"
#include "FlowShopInit.h"
#include "FlowShopBenchmarkParser.h"
// also need the parser and param includes
#include <utils/eoParser.h>
#include <utils/eoState.h>


/*
 * This function creates an eoInit<eoFlowShop> that can later be used to initialize the population (see make_pop.h).
 * @param eoParser& _parser  to get user parameters
 * @param eoState& _state  to store the memory
 */
eoInit < FlowShop > &do_make_genotype (eoParser & _parser, eoState & _state)
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
	"   Type '-B=the_benchmark_file_name' or '--BenchmarkFile=the_benchmark_file_name'\n";
      stmp += "   Benchmarks files are available at " + BENCHMARKS_WEB_SITE;
      throw std::runtime_error (stmp.c_str ());
    }
  // reading of number of jobs to schedule contained in the benchmark file   
  FlowShopBenchmarkParser fParser (benchmarkFileName);
  unsigned N = fParser.getN ();

  // build of the initializer (a pointer, stored in the eoState)
  eoInit < FlowShop > *init = new FlowShopInit (N);
  // store in state
  _state.storeFunctor (init);
  // and return a reference
  return *init;
}

#endif
