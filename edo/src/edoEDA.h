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
    Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _edoEDA_h
#define _edoEDA_h

#include <eo>
#include <mo>

#include <utils/eoRNG.h>

#include "edoAlgo.h"
#include "edoEstimator.h"
#include "edoModifierMass.h"
#include "edoSampler.h"
#include "edoContinue.h"

//! edoEDA< D >

template < typename D >
class edoEDA : public edoAlgo< D >
{
public:
    //! Alias for the type EOT
    typedef typename D::EOType EOT;

    //! Alias for the atom type
    typedef typename EOT::AtomType AtomType;

    //! Alias for the fitness
    typedef typename EOT::Fitness Fitness;

public:

    //! edoEDA constructor
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
    edoEDA (
        eoPopEvalFunc < EOT > & evaluator,
        eoSelect< EOT > & selector,
        edoEstimator< D > & estimator,
        edoSampler< D > & sampler,
        eoReplacement< EOT > & replacor,
        eoContinue< EOT > & pop_continuator,
        edoContinue< D > & distribution_continuator
    ) :
        _evaluator(evaluator),
        _selector(selector),
        _estimator(estimator),
        _sampler(sampler),
        _replacor(replacor),
        _pop_continuator(pop_continuator),
        _dummy_continue(),
        _distribution_continuator(distribution_continuator)
    {}

    //! edoEDA constructor without an edoContinue
    /*!
      Takes algo operators, all are mandatory

      \param evaluation Evaluate a population
      \param selector Selection of the best candidate solutions in the population
      \param estimator Estimation of the distribution parameters
      \param sampler Generate feasible solutions using the distribution
      \param replacor Replace old solutions by new ones
      \param pop_continuator Stopping criterion based on the population features
    */
    edoEDA (
        eoPopEvalFunc < EOT > & evaluator,
        eoSelect< EOT > & selector,
        edoEstimator< D > & estimator,
        edoSampler< D > & sampler,
        eoReplacement< EOT > & replacor,
        eoContinue< EOT > & pop_continuator
    ) :
        _evaluator(evaluator),
        _selector(selector),
        _estimator(estimator),
        _sampler(sampler),
        _replacor(replacor),
        _pop_continuator(pop_continuator),
        _dummy_continue(),
        _distribution_continuator( _dummy_continue )
    {}


    /** A basic EDA algorithm that iterates over:
     * selection, estimation, sampling, bounding, evaluation, replacement
     *
     * \param pop the population of candidate solutions
     * \return void
    */
    void operator ()(eoPop< EOT > & pop)
    {
        assert(pop.size() > 0);

        eoPop< EOT > current_pop;
        eoPop< EOT > selected_pop;

        // FIXME one must instanciate a first distrib here because there is no empty constructor, see if it is possible to instanciate Distributions without parameters
        D distrib = _estimator(pop);

        // Evaluating a first time the candidate solutions
        // The first pop is not supposed to be evaluated (@see eoPopLoopEval).
        _evaluator( current_pop, pop );

        do {
            // (1) Selection of the best points in the population
            //selected_pop.clear(); // FIXME is it necessary to clear?
            _selector(pop, selected_pop);
            assert( selected_pop.size() > 0 );
            // TODO: utiliser selected_pop ou pop ???

            // (2) Estimation of the distribution parameters
            distrib = _estimator(selected_pop);

            // (3) sampling
            // The sampler produces feasible solutions (@see edoSampler that
            // encapsulate an edoBounder)
            current_pop.clear();
            for( unsigned int i = 0; i < pop.size(); ++i ) {
                current_pop.push_back( _sampler(distrib) );
            }

            // (4) Evaluate new solutions
            _evaluator( pop, current_pop );

            // (5) Replace old solutions by new ones
            _replacor(pop, current_pop); // e.g. copy current_pop in pop

        } while( _distribution_continuator( distrib ) && _pop_continuator( pop ) );
    } // operator()

private:

    //! A full evaluation function.
    eoPopEvalFunc < EOT > & _evaluator;

    //! A EOT selector
    eoSelect < EOT > & _selector;

    //! A EOT estimator. It is going to estimate distribution parameters.
    edoEstimator< D > & _estimator;

    //! A D sampler
    edoSampler< D > & _sampler;

    //! A EOT replacor
    eoReplacement < EOT > & _replacor;

    //! A EOT population continuator
    eoContinue < EOT > & _pop_continuator;

    //! A D continuator that always return true
    edoDummyContinue<D> _dummy_continue;

    //! A D continuator
    edoContinue < D > & _distribution_continuator;

};

#endif // !_edoEDA_h
