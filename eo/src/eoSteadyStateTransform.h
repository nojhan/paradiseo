// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSteadyStateTransform.h
// (c) Maarten Keijzer 2000, GeNeura Team, 1998
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

#ifndef eoSteadyStateTransform_h
#define eoSteadyStateTransform_h

//-----------------------------------------------------------------------------

#include <eoAlgo.h>     // eoPop
#include <eoEvalFunc.h> 

#include <eoGOpSelector.h>
#include <eoIndiSelector.h>
#include <eoSteadyStateInserter.h>

//-----------------------------------------------------------------------------

/** eoSteadyStateTransform
 * Single step of a steady state evolutionary algorithm. 
 * Proceeds by updating one individual at a time, by first selecting parents, 
 * creating one or more children and subsequently overwrite (a) bad individual(s)
*/
template<class EOT> class eoSteadyStateTransform: public eoTransform<EOT>
{
 public:
  /// Constructor.
  eoSteadyStateTransform(
      eoGOpSelector<EOT>& _opSelector, 
      eoSelectOne<EOT>& _selector,
      eoSteadyStateInserter<EOT>& _inserter, 
      unsigned _steps = 0) :
            opSelector(_opSelector), 
            selector(_selector), 
            inserter(_inserter) , 
            steps(_steps) {};


  /// Apply one generation of evolution to the population.
  virtual void operator()(eoPop<EOT>& pop) 
  {
      unsigned nSteps = steps;
      if (nSteps == 0)
      {
          nSteps = pop.size(); // make a 'generation equivalent'
      }

      for (unsigned i = 0; i < nSteps; ++i)
      {
         selector.bind(pop);
         inserter.bind(pop);

         opSelector.selectOp()(selector, inserter);
      }

  }
  
 private:
  eoGOpSelector<EOT>&         opSelector;
  eoSelectOneIndiSelector<EOT>  selector;
  eoSteadyStateInserter<EOT>& inserter;
  unsigned steps;
};

//-----------------------------------------------------------------------------

#endif eoGeneration_h

