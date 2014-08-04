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

#ifndef _edoAlgoAdaptive_h
#define _edoAlgoAdaptive_h

#include "../eo/utils/eoRNG.h"

#include "edoAlgo.h"
#include "edoEstimator.h"
#include "edoModifierMass.h"
#include "edoSampler.h"
#include "edoContinue.h"

/** A generic stochastic search template for algorithms that need a distribution parameter.
 *
 * An adaptive algorithm will directly updates a distribution, it must thus be instanciated
 * with an edoDistrib at hand. Thus, this distribution object should be instanciated appart.
 * The reference to this distribution is generally also needed by at least one of the
 * algorithm's operator, generally for algorithms that shares the same algorithms across
 * operators and/or iterations.
 *
 * If you no operator needs to update the distribution, then it is simpler to use an
 * edoAlgoStateless .
 *
 * @ingroup Algorithms
 */
template < typename D >
class edoAlgoAdaptive : public edoAlgo< D >
{
public:
    //! Alias for the type EOT
    typedef typename D::EOType EOType;

    //! Alias for the atom type
    typedef typename EOType::AtomType AtomType;

    //! Alias for the fitness
    typedef typename EOType::Fitness Fitness;

public:

    /*!
      Takes algo operators, all are mandatory

      \param distrib A distribution to use, if you want to update this parameter (e.gMA-ES) instead of replacing it (e.g. an EDA)
      \param evaluator Evaluate a population
      \param selector Selection of the best candidate solutions in the population
      \param estimator Estimation of the distribution parameters
      \param sampler Generate feasible solutions using the distribution
      \param replacor Replace old solutions by new ones
      \param pop_continuator Stopping criterion based on the population features
      \param distribution_continuator Stopping criterion based on the distribution features
    */
    edoAlgoAdaptive(
        D & distrib,
        eoPopEvalFunc < EOType > & evaluator,
        eoSelect< EOType > & selector,
        edoEstimator< D > & estimator,
        edoSampler< D > & sampler,
        eoReplacement< EOType > & replacor,
        eoContinue< EOType > & pop_continuator,
        edoContinue< D > & distribution_continuator
    ) :
        _distrib(distrib),
        _evaluator(evaluator),
        _selector(selector),
        _estimator(estimator),
        _sampler(sampler),
        _replacor(replacor),
        _pop_continuator(pop_continuator),
        _dummy_continue(),
        _distribution_continuator(distribution_continuator)
    {}


    //! constructor without an edoContinue
    /*!
      Takes algo operators, all are mandatory

      \param distrib A distribution to use, if you want to update this parameter (e.gMA-ES) instead of replacing it (e.g. an EDA)
      \param evaluator Evaluate a population
      \param selector Selection of the best candidate solutions in the population
      \param estimator Estimation of the distribution parameters
      \param sampler Generate feasible solutions using the distribution
      \param replacor Replace old solutions by new ones
      \param pop_continuator Stopping criterion based on the population features
    */
    edoAlgoAdaptive (
        D & distrib,
        eoPopEvalFunc < EOType > & evaluator,
        eoSelect< EOType > & selector,
        edoEstimator< D > & estimator,
        edoSampler< D > & sampler,
        eoReplacement< EOType > & replacor,
        eoContinue< EOType > & pop_continuator
    ) :
        _distrib( distrib ),
        _evaluator(evaluator),
        _selector(selector),
        _estimator(estimator),
        _sampler(sampler),
        _replacor(replacor),
        _pop_continuator(pop_continuator),
        _dummy_continue(),
        _distribution_continuator( _dummy_continue )
    {}

    /** Call the algorithm
     *
     * \param pop the population of candidate solutions
     * \return void
    */
    void operator ()(eoPop< EOType > & pop)
    {
        assert(pop.size() > 0);

        eoPop< EOType > current_pop;
        eoPop< EOType > selected_pop;

        // update the extern distribution passed to the estimator (cf. CMA-ES)
        // OR replace the dummy distribution for estimators that do not need extern distributions (cf. EDA)
        _distrib = _estimator(pop);

        // Evaluating a first time the candidate solutions
        // The first pop is not supposed to be evaluated (@see eoPopLoopEval).
        // _evaluator( current_pop, pop );

        do {
            // (1) Selection of the best points in the population
            _selector(pop, selected_pop);
            assert( selected_pop.size() > 0 );

            // (2) Estimation of the distribution parameters
            _distrib = _estimator(selected_pop);

            // (3) sampling
            // The sampler produces feasible solutions (@see edoSampler that
            // encapsulate an edoBounder)
            current_pop.clear();
            for( unsigned int i = 0; i < pop.size(); ++i ) {
                current_pop.push_back( _sampler(_distrib) );
            }

            // (4) Evaluate new solutions
            _evaluator( pop, current_pop );

            // (5) Replace old solutions by new ones
            _replacor(pop, current_pop); // e.g. copy current_pop in pop

        } while( _distribution_continuator( _distrib ) && _pop_continuator( pop ) );
    } // operator()


protected:

    //! The distribution that you want to update
    D & _distrib;

    //! A full evaluation function.
    eoPopEvalFunc<EOType> & _evaluator;

    //! A EOType selector
    eoSelect<EOType> & _selector;

    //! A EOType estimator. It is going to estimate distribution parameters.
    edoEstimator<D> & _estimator;

    //! A D sampler
    edoSampler<D> & _sampler;

    //! A EOType replacor
    eoReplacement<EOType> & _replacor;

    //! A EOType population continuator
    eoContinue<EOType> & _pop_continuator;

    //! A D continuator that always return true
    edoDummyContinue<D> _dummy_continue;

    //! A D continuator
    edoContinue<D> & _distribution_continuator;

};

#endif // !_edoAlgoAdaptive_h

