/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

-----------------------------------------------------------------------------
    eoPopEvalFunc.h
    Abstract class for global evaluation of the population

    (c) GeNeura Team, 2000

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

#ifndef eoPopEvalFunc_H
#define eoPopEvalFunc_H

#include <eoEvalFunc.h>
#include <apply.h>

/** eoPopEvalFunc: This abstract class is for GLOBAL evaluators
 *    of a population after variation.
 *    It takes 2 populations (typically the parents and the offspring)
 *    and is suppposed to evaluate them alltogether
 *
 *  Basic use: apply an embedded eoEvalFunc to the offspring
 *
 *  Time-varying fitness: apply the embedded eoEvalFunc to both
 *     offspring and parents
 *
 *  Advanced uses: Co-evolution or "parisian" approach, or ...
 *
 *  Basic parallelization (synchronous standard evolution engine):
 *    call the slaves and wait for the results
 *
 *    @ingroup Evaluation
 */
template<class EOT>
class eoPopEvalFunc : public eoBF<eoPop<EOT> & , eoPop<EOT> &, void>
{};

/////////////////////////////////////////////////////////////
//           eoPopLoopEval
/////////////////////////////////////////////////////////////

/** eoPopLoopEval: an instance of eoPopEvalFunc that simply applies
 *     a private eoEvalFunc to all offspring
 *
 *    @ingroup Evaluation
 */
template<class EOT>
class eoPopLoopEval : public eoPopEvalFunc<EOT> {
public:
  /** Ctor: set value of embedded eoEvalFunc */
  eoPopLoopEval(eoEvalFunc<EOT> & _eval) : eval(_eval) {}

  /** Do the job: simple loop over the offspring */
  void operator()(eoPop<EOT> & _parents, eoPop<EOT> & _offspring)
  {
      (void)_parents;
      apply<EOT>(eval, _offspring);
  }

private:
  eoEvalFunc<EOT> & eval;
};

/////////////////////////////////////////////////////////////
//           eoTimeVaryingLoopEval
/////////////////////////////////////////////////////////////

/** eoPopLoopEval: an instance of eoPopEvalFunc that simply applies
 *     a private eoEvalFunc to all offspring AND ALL PARENTS
 *     as the fitness is supposed here to vary
 *
 *    @ingroup Evaluation
 */
template<class EOT>
class eoTimeVaryingLoopEval : public eoPopEvalFunc<EOT> {
public:
  /** Ctor: set value of embedded eoEvalFunc */
  eoTimeVaryingLoopEval(eoEvalFunc<EOT> & _eval) : eval(_eval) {}

  /** Do the job: simple loop over the offspring */
  void operator()(eoPop<EOT> & _parents, eoPop<EOT> & _offspring)
  {
    apply<EOT>(eval, _parents);
    apply<EOT>(eval, _offspring);
  }

private:
  eoEvalFunc<EOT> & eval;
};

#endif
