/*
The Evolving Distribution Objects framework (EDO) is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own estimation of distribution algorithms.

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
    Pierre Savéant <pierre.saveant@thalesgroup.com>
*/

#ifndef _edoAlgoStateless_h
#define _edoAlgoStateless_h

#include "edoAlgoAdaptive.h"

//! edoEDA< D >

/** A generic stochastic search template for algorithms that need a distribution parameter.
 */
template < typename EOD >
class edoAlgoStateless : public edoAlgoAdaptive< EOD >
{
public:
    //! Alias for the type EOT
    typedef typename EOD::EOType EOType;

    //! Alias for the atom type
    typedef typename EOType::AtomType AtomType;

    //! Alias for the fitness
    typedef typename EOType::Fitness Fitness;

public:


    /*!
      Without a distribution

      \param evaluation Evaluate a population
      \param selector Selection of the best candidate solutions in the population
      \param estimator Estimation of the distribution parameters
      \param sampler Generate feasible solutions using the distribution
      \param replacor Replace old solutions by new ones
      \param pop_continuator Stopping criterion based on the population features
      \param distribution_continuator Stopping criterion based on the distribution features
    */
    edoAlgoStateless(
        eoPopEvalFunc < EOType > & evaluator,
        eoSelect< EOType > & selector,
        edoEstimator< EOD > & estimator,
        edoSampler< EOD > & sampler,
        eoReplacement< EOType > & replacor,
        eoContinue< EOType > & pop_continuator,
        edoContinue< EOD > & distribution_continuator,
        EOD* tmp_distrib = (new EOD())
    ) :
        edoAlgoAdaptive<EOD>( *tmp_distrib, evaluator, selector, estimator, sampler, replacor, pop_continuator, distribution_continuator),
        _tmp_distrib( tmp_distrib ),
        _dummy_distrib()
    {
        this->_distrib = _dummy_distrib;
    }

    //! constructor without an edoContinue nor a distribution
    /*!
      \param evaluation Evaluate a population
      \param selector Selection of the best candidate solutions in the population
      \param estimator Estimation of the distribution parameters
      \param sampler Generate feasible solutions using the distribution
      \param replacor Replace old solutions by new ones
      \param pop_continuator Stopping criterion based on the population features
    */
    edoAlgoStateless (
        eoPopEvalFunc < EOType > & evaluator,
        eoSelect< EOType > & selector,
        edoEstimator< EOD > & estimator,
        edoSampler< EOD > & sampler,
        eoReplacement< EOType > & replacor,
        eoContinue< EOType > & pop_continuator,
        EOD* tmp_distrib = (new EOD())
    ) :
        edoAlgoAdaptive<EOD>( *tmp_distrib, evaluator, selector, estimator, sampler, replacor, pop_continuator),
        _tmp_distrib( tmp_distrib ),
        _dummy_distrib()
    {
        this->_distrib = _dummy_distrib;
    }

    ~edoAlgoStateless()
    {
        delete _tmp_distrib;
    }


protected:

    EOD* _tmp_distrib;

    /** A dummy distribution, for algorithms willing to replace it instead of updating
     *
     * Thus we can instanciate _distrib on this and replace it at the first iteration with an estimator.
     * This is why an edoDistrib must have an empty constructor.
     */
    EOD _dummy_distrib;

};

#endif // !_edoAlgoStateless_h

