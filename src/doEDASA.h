// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doEDASA_h
#define _doEDASA_h

#include <eo>
#include <mo>

#include <utils/eoRNG.h>

#include "doAlgo.h"
#include "doEstimator.h"
#include "doModifierMass.h"
#include "doSampler.h"
#include "doContinue.h"

template < typename D >
class doEDASA : public doAlgo< D >
{
public:
    //! Alias for the type EOT
    typedef typename D::EOType EOT;

    //! Alias for the atom type
    typedef typename EOT::AtomType AtomType;

    //! Alias for the fitness
    typedef typename EOT::Fitness Fitness;

public:

    //! doEDASA constructor
    /*!
      All the boxes used by a EDASA need to be given.

      \param selector Population Selector
      \param estimator Distribution Estimator
      \param selectone SelectOne
      \param modifier Distribution Modifier
      \param sampler Distribution Sampler
      \param pop_continue Population Continuator
      \param distribution_continue Distribution Continuator
      \param evaluation Evaluation function.
      \param sa_continue Stopping criterion.
      \param cooling_schedule Cooling schedule, describes how the temperature is modified.
      \param initial_temperature The initial temperature.
      \param replacor Population replacor
    */
    doEDASA (eoSelect< EOT > & selector,
	     doEstimator< D > & estimator,
	     eoSelectOne< EOT > & selectone,
	     doModifierMass< D > & modifier,
	     doSampler< D > & sampler,
	     eoContinue< EOT > & pop_continue,
	     doContinue< D > & distribution_continue,
	     eoEvalFunc < EOT > & evaluation,
	     moContinuator< moDummyNeighbor<EOT> > & sa_continue,
	     moCoolingSchedule<EOT> & cooling_schedule,
	     double initial_temperature,
	     eoReplacement< EOT > & replacor
	     )
	: _selector(selector),
	  _estimator(estimator),
	  _selectone(selectone),
	  _modifier(modifier),
	  _sampler(sampler),
	  _pop_continue(pop_continue),
	  _distribution_continue(distribution_continue),
	  _evaluation(evaluation),
	  _sa_continue(sa_continue),
	  _cooling_schedule(cooling_schedule),
	  _initial_temperature(initial_temperature),
	  _replacor(replacor)
    {}

    //! function that launches the EDASA algorithm.
    /*!
      As a moTS or a moHC, the EDASA can be used for HYBRIDATION in an evolutionary algorithm.

      \param pop A population to improve.
      \return TRUE.
    */
    void operator ()(eoPop< EOT > & pop)
    {
        assert(pop.size() > 0);

	double temperature = _initial_temperature;

	eoPop< EOT > current_pop;

	eoPop< EOT > selected_pop;


	//-------------------------------------------------------------
	// Estimating a first time the distribution parameter thanks
	// to population.
	//-------------------------------------------------------------

	D distrib = _estimator(pop);

	double size = distrib.size();
	assert(size > 0);

	//-------------------------------------------------------------


	do
	    {
		//-------------------------------------------------------------
		// (3) Selection of the best points in the population
		//-------------------------------------------------------------

		selected_pop.clear();

		_selector(pop, selected_pop);

		assert( selected_pop.size() > 0 );

		//-------------------------------------------------------------


		_sa_continue.init();


		//-------------------------------------------------------------
		// (4) Estimation of the distribution parameters
		//-------------------------------------------------------------

		distrib = _estimator(selected_pop);

		//-------------------------------------------------------------


		// TODO: utiliser selected_pop ou pop ???

		assert(selected_pop.size() > 0);


		//-------------------------------------------------------------
		// Init of a variable contening a point with the bestest fitnesses
		//-------------------------------------------------------------

		EOT current_solution = _selectone(selected_pop);

		//-------------------------------------------------------------


		//-------------------------------------------------------------
		// Fit the current solution with the distribution parameters (bounds)
		//-------------------------------------------------------------

		// FIXME: si besoin de modifier la dispersion de la distribution
		// _modifier_dispersion(distribution, selected_pop);
		_modifier(distrib, current_solution);

		//-------------------------------------------------------------


		//-------------------------------------------------------------
		// Building of the sampler in current_pop
		//-------------------------------------------------------------

		current_pop.clear();

		do
		    {
			EOT candidate_solution = _sampler(distrib);

			EOT& e1 = candidate_solution;
			_evaluation( e1 );
			EOT& e2 = current_solution;
			_evaluation( e2 );

			// TODO: verifier le critere d'acceptation
			if ( e1.fitness() < e2.fitness() ||
			     rng.uniform() < exp( ::fabs(e1.fitness() - e2.fitness()) / temperature ) )
			    {
				current_pop.push_back(candidate_solution);
				current_solution = candidate_solution;
			    }
		    }
 		while ( _sa_continue( current_solution) );

		_replacor(pop, current_pop); // copy current_pop in pop

		pop.sort();

		if ( ! _cooling_schedule( temperature ) ){ eo::log << eo::debug << "_cooling_schedule" << std::endl; break; }

		if ( ! _distribution_continue( distrib ) ){ eo::log << eo::debug << "_distribution_continue" << std::endl; break; }

		if ( ! _pop_continue( pop ) ){ eo::log << eo::debug << "_pop_continue" << std::endl; break; }

	    }
	while ( 1 );
    }

private:

    //! A EOT selector
    eoSelect < EOT > & _selector;

    //! A EOT estimator. It is going to estimate distribution parameters.
    doEstimator< D > & _estimator;

    //! SelectOne
    eoSelectOne< EOT > & _selectone;

    //! A D modifier
    doModifierMass< D > & _modifier;

    //! A D sampler
    doSampler< D > & _sampler;

    //! A EOT population continuator
    eoContinue < EOT > & _pop_continue;

    //! A D continuator
    doContinue < D > & _distribution_continue;

    //! A full evaluation function.
    eoEvalFunc < EOT > & _evaluation;

    //! Stopping criterion before temperature update
    moContinuator< moDummyNeighbor<EOT> > & _sa_continue;

    //! The cooling schedule
    moCoolingSchedule<EOT> & _cooling_schedule;

    //! Initial temperature
    double  _initial_temperature;

    //! A EOT replacor
    eoReplacement < EOT > & _replacor;

    //-------------------------------------------------------------
    // Temporary solution to store populations state at each
    // iteration for plotting.
    //-------------------------------------------------------------

    // std::ofstream _ofs_params;
    // std::ofstream _ofs_params_var;

    //-------------------------------------------------------------

    //-------------------------------------------------------------
    // Temporary solution to store bounds values for each distribution.
    //-------------------------------------------------------------

    // std::string _bounds_results_destination;

    //-------------------------------------------------------------

};

#endif // !_doEDASA_h
