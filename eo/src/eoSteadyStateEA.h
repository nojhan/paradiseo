// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSteadyStateEA.h
// (c) GeNeura Team, 2000
/* 
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
 */
//-----------------------------------------------------------------------------

#ifndef _eoSteadyStateEA_h
#define _eoSteadyStateEA_h

//-----------------------------------------------------------------------------

#include "eoSteadyStateGeneration.h"     // eoPop
#include <eoTerm.h>

/** EOSteadyStateEA:
    An easy-to-use evolutionary algorithm, just supply
    a general operator selector, a selector for choosing the ones
    to reproduce and an eoSteadyStateInserter that takes care of evaluating
    and inserter the guy/girl in the steady state population.
*/
template<class EOT> class eoSteadyStateEA: public eoAlgo<EOT>
{
 public:
  /// Constructor.
  eoSteadyStateEA(
      eoGOpSelector<EOT>& _opSelector, 
      eoPopIndiSelector<EOT>& _selector,
      eoSteadyStateInserter<EOT>& _inserter, 
      eoTerm<EOT>&     _terminator,
      unsigned _steps = 0 )   
      : step(_opSelector, _selector, _inserter), 
      terminator( _terminator)
        {};

  /// Constructor from an already created generation
  eoSteadyStateEA(eoSteadyStateGeneration<EOT>& _gen,
	   eoTerm<EOT>&     _terminator):
    step(_gen), 
    terminator( _terminator){};
  
  /// Apply one generation of evolution to the population.
  virtual void operator()(eoPop<EOT>& pop) {
    do {
      try
	{
	  step(pop);
	}
    catch (exception& e)
	{
	  string s = e.what();
	  s.append( " in eoSteadyStateEA ");
	  throw runtime_error( s );
	}
    } while ( terminator( pop ) );
  
  }
  
  /// Class name.
  string className() const { return "eoSteadyStateEA"; }
  
 private:
  eoSteadyStateGeneration<EOT>  step;
  eoTerm<EOT>& terminator;
};

//-----------------------------------------------------------------------------

#endif eoEasyEA_h

