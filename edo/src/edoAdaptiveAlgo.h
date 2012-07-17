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

#ifndef _edoAdaptiveAlgo_h
#define _edoAdaptiveAlgo_h

#include <eo>

#include <utils/eoRNG.h>

#include "edoAlgo.h"
#include "edoEstimator.h"
#include "edoModifierMass.h"
#include "edoSampler.h"
#include "edoContinue.h"

//! edoEDA< D >

// FIXME factoriser edoAdaptiveAlgo et edoEDA, la seule différence est la référence _distrib !
template < typename EOD >
class edoAdaptiveAlgo : public edoAlgo< EOD >
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
      Takes algo operators, all are mandatory

      \param evaluation Evaluate a population
      \param selector Selection of the best candidate solutions in the population
      \param estimator Estimation of the distribution parameters
      \param sampler Generate feasible solutions using the distribution
      \param replacor Replace old solutions by new ones
      \param pop_continuator Stopping criterion based on the population features
      \param distribution_continuator Stopping criterion based on the distribution features
    */
    edoAdaptiveAlgo(
        EOD & distrib,
        eoPopEvalFunc < EOType > & evaluator,
        eoSelect< EOType > & selector,
        edoEstimator< EOD > & estimator,
        edoSampler< EOD > & sampler,
        eoReplacement< EOType > & replacor,
        eoContinue< EOType > & pop_continuator,
        edoContinue< EOD > & distribution_continuator
    ) :
        _dummy_distrib(),
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
    edoAdaptiveAlgo(
        eoPopEvalFunc < EOType > & evaluator,
        eoSelect< EOType > & selector,
        edoEstimator< EOD > & estimator,
        edoSampler< EOD > & sampler,
        eoReplacement< EOType > & replacor,
        eoContinue< EOType > & pop_continuator,
        edoContinue< EOD > & distribution_continuator
    ) :
        _dummy_distrib(),
        _distrib( _dummy_distrib ),
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

      \param evaluation Evaluate a population
      \param selector Selection of the best candidate solutions in the population
      \param estimator Estimation of the distribution parameters
      \param sampler Generate feasible solutions using the distribution
      \param replacor Replace old solutions by new ones
      \param pop_continuator Stopping criterion based on the population features
    */
    edoAdaptiveAlgo (
        EOD & distrib,
        eoPopEvalFunc < EOType > & evaluator,
        eoSelect< EOType > & selector,
        edoEstimator< EOD > & estimator,
        edoSampler< EOD > & sampler,
        eoReplacement< EOType > & replacor,
        eoContinue< EOType > & pop_continuator
    ) :
        _dummy_distrib(),
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

    //! constructor without an edoContinue nor a distribution
    /*!
      Takes algo operators, all are mandatory

      \param evaluation Evaluate a population
      \param selector Selection of the best candidate solutions in the population
      \param estimator Estimation of the distribution parameters
      \param sampler Generate feasible solutions using the distribution
      \param replacor Replace old solutions by new ones
      \param pop_continuator Stopping criterion based on the population features
    */
    edoAdaptiveAlgo (
        eoPopEvalFunc < EOType > & evaluator,
        eoSelect< EOType > & selector,
        edoEstimator< EOD > & estimator,
        edoSampler< EOD > & sampler,
        eoReplacement< EOType > & replacor,
        eoContinue< EOType > & pop_continuator
    ) :
        _dummy_distrib(),
        _distrib( _dummy_distrib ),
        _evaluator(evaluator),
        _selector(selector),
        _estimator(estimator),
        _sampler(sampler),
        _replacor(replacor),
        _pop_continuator(pop_continuator),
        _dummy_continue(),
        _distribution_continuator( _dummy_continue )
    {}




    /** Covariance Matrix Adaptation Evolution Strategies
     *
     * \param pop the population of candidate solutions
     * \return void
    */
    void operator ()(eoPop< EOType > & pop)
    {
        assert(pop.size() > 0);

        eoPop< EOType > current_pop;
        eoPop< EOType > selected_pop;

        // FIXME one must instanciate a first distrib here because there is no empty constructor, see if it is possible to instanciate Distributions without parameters
        _distrib = _estimator(pop);

        // Evaluating a first time the candidate solutions
        // The first pop is not supposed to be evaluated (@see eoPopLoopEval).
        // _evaluator( current_pop, pop );

        do {
            // (1) Selection of the best points in the population
            //selected_pop.clear(); // FIXME is it necessary to clear?
            _selector(pop, selected_pop);
            assert( selected_pop.size() > 0 );
            // TODO: utiliser selected_pop ou pop ???

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

    EOD _dummy_distrib;

    EOD & _distrib;

    //! A full evaluation function.
    eoPopEvalFunc<EOType> & _evaluator;

    //! A EOType selector
    eoSelect<EOType> & _selector;

    //! A EOType estimator. It is going to estimate distribution parameters.
    edoEstimator<EOD> & _estimator;

    //! A D sampler
    edoSampler<EOD> & _sampler;

    //! A EOType replacor
    eoReplacement<EOType> & _replacor;

    //! A EOType population continuator
    eoContinue<EOType> & _pop_continuator;

    //! A D continuator that always return true
    edoDummyContinue<EOD> _dummy_continue;

    //! A D continuator
    edoContinue<EOD> & _distribution_continuator;

};

#endif // !_edoAdaptiveAlgo_h

