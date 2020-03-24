/*
The Evolving Objects framework is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own evolutionary algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2010 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
*/

#ifndef _EOALGORESTART_H_
#define _EOALGORESTART_H_

#include "eoPop.h"
#include "eoAlgo.h"

/** An algo that restart the given algorithm on a freshly init pop.
 *
 * @note: The fresh population size is set to the size of the last population at previous run.
 *
 *  @ingroup Algorithms
 */
template<class EOT>
class eoAlgoRestart : public eoAlgo<EOT>
{
public:
    /** Constructor with an eoPopEvalFunc
     *
     * @param init the initialization operator
     * @param popeval an evaluator for populations
     * @param algo the algorithm to restart
     * @param continuator a stopping criterion to manage the number of restarts
     */
    eoAlgoRestart(
            eoInit<EOT>& init,
            eoPopEvalFunc<EOT>& pop_eval,
            eoAlgo<EOT>& algo,
            eoContinue<EOT>& continuator
        ) :
            eoAlgo<EOT>(),
            _init(init),
            _eval(_dummy_eval),
            _loop_eval(_dummy_eval),
            _pop_eval(pop_eval),
            _algo(algo),
            _continue(continuator)
    {}

    /** Constructor with an eoEvalFunc
     *
     * @param init the initialization operator
     * @param popeval an evaluator for populations
     * @param algo the algorithm to restart
     * @param continuator a stopping criterion to manage the number of restarts
     */
    eoAlgoRestart(
            eoInit<EOT>& init,
            eoEvalFunc<EOT>& eval,
            eoAlgo<EOT>& algo,
            eoContinue<EOT>& continuator
        ) :
            eoAlgo<EOT>(),
            _init(init),
            _eval(eval),
            _loop_eval(_eval),
            _pop_eval(_loop_eval),
            _algo(algo),
            _continue(continuator)
    {}

    virtual void operator()(eoPop<EOT> & pop)
    {
        do {
            size_t pop_size = pop.size();
            pop.clear();
            pop.append(pop_size, _init);
            _pop_eval(pop,pop);

            _algo(pop);

        } while( _continue(pop) );
    }

protected:
    eoInit<EOT>& _init;

    eoEvalFunc<EOT>& _eval; 
    eoPopLoopEval<EOT> _loop_eval;
    eoPopEvalFunc<EOT>& _pop_eval;

    eoAlgo<EOT>& _algo;
    eoContinue<EOT>& _continue;

    class eoDummyEval : public eoEvalFunc<EOT>
    {
        public:
            void operator()(EOT &)
            {}
    };
    eoDummyEval _dummy_eval;
};


#endif // _EOALGORESTART_H_
