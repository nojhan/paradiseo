/*
The Evolving Objects framework is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own evolutionary algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation;
version 2.1 of the License.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2020 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
*/

#ifndef _EOALGORESET_H_
#define _EOALGORESET_H_

#include "eoContinue.h"

/** @defgroup Reset 
 *
 * Algorithms operating a reset of some sort.
 * (e.g. on the population).
 */

/** A semantic layer indicating that the derived operator operates a reset of some sort.
 *
 * @see eoAlgoRestart
 *
 * @ingroup Algorithms
 * @ingroup Reset
 */
template<class EOT>
class eoAlgoReset : public eoAlgo<EOT>
{ };

/** Reset the given population when called.
 *
 * i.e. Remove all its content, then re-generate individuals
 * with the given eoInit.
 *
 * The reinitialized pop will have either
 * the same size than the previous population
 * (if no pop_size is passed to the constructor),
 * either the previous (given) pop size.
 *
 * @see eoAlgoRestart
 *
 * @ingroup Reset
 */
template<class EOT>
class eoAlgoPopReset : public eoAlgoReset<EOT>
{
    public:
        /** Constructor for fixed-size populations. */
        eoAlgoPopReset( eoInit<EOT>& init, eoPopEvalFunc<EOT>& pop_eval ) :
            _init(init),
            _pop_eval(pop_eval),
            _has_pop_size(false),
            _pop_size(0)
        { }

        /** Constructor for resets to the given population size. */
        eoAlgoPopReset( eoInit<EOT>& init, eoPopEvalFunc<EOT>& pop_eval, size_t pop_size ) :
            _init(init),
            _pop_eval(pop_eval),
            _has_pop_size(true),
            _pop_size(pop_size)
        { }

        virtual void operator()(eoPop<EOT>& pop)
        {
            if(not _has_pop_size) {
                _pop_size = pop.size();
            }
            pop.clear();
            pop.append(_pop_size, _init);
            _pop_eval(pop,pop);
        }

    protected:
        eoInit<EOT>& _init;
        eoPopEvalFunc<EOT>& _pop_eval;
        bool _has_pop_size;
        size_t _pop_size;
};

/** Combine several eoAlgoReset in one.
 *
 * Useful if you want to perform several different resets.
 *
 * @ingroup Reset
 */
template<class EOT>
class eoAlgoResetCombine : public eoAlgoReset<EOT>
{
    public:
        eoAlgoResetCombine( eoAlgoReset<EOT>& reseter ) :
            _reseters(1, &reseter)
        { }

        eoAlgoResetCombine( std::vector<eoAlgoReset<EOT>*> reseters ) :
            _reseters(reseters)
        { }

        void add( eoAlgoReset<EOT>& reseter )
        {
            _reseters.push_back(&reseter);
        }

        virtual void operator()(eoPop<EOT>& pop)
        {
            for(auto& reseter : _reseters) {
                (*reseter)(pop);
            }
        }

    protected:
        std::vector<eoAlgoReset<EOT>*> _reseters;
};

#endif // _EOALGORESET_H_
