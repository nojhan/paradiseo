#ifndef _doCMASA_h
#define _doCMASA_h

//-----------------------------------------------------------------------------
// Temporary solution to store populations state at each iteration for plotting.
//-----------------------------------------------------------------------------

// system header inclusion
#include <cstdlib>
// end system header inclusion

// mkdir headers inclusion
#include <sys/stat.h>
#include <sys/types.h>
// end mkdir headers inclusion

#include <fstream>
#include <numeric>

#include <limits>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

//-----------------------------------------------------------------------------

#include <eo>
#include <mo>

#include <utils/eoRNG.h>

#include "doAlgo.h"
#include "doEstimator.h"
#include "doModifierMass.h"
#include "doSampler.h"
#include "doStats.h"

using namespace boost::numeric::ublas;

template < typename D >
class doCMASA : public doAlgo< D >
{
public:
    //! Alias for the type
    typedef typename D::EOType EOT;

    //! Alias for the fitness
    typedef typename EOT::Fitness Fitness;

public:

    //! doCMASA constructor
    /*!
      All the boxes used by a CMASA need to be given.

      \param selector The EOT selector
      \param estomator The EOT selector
      \param selectone SelectOne
      \param modifier The D modifier
      \param sampler The D sampler
      \param evaluation The evaluation function.
      \param continue The stopping criterion.
      \param cooling_schedule The cooling schedule, describes how the temperature is modified.
      \param initial_temperature The initial temperature.
      \param replacor The EOT replacor
    */
    doCMASA (eoSelect< EOT > & selector,
	     doEstimator< D > & estimator,
	     eoSelectOne< EOT > & selectone,
	     doModifierMass< D > & modifier,
	     doSampler< D > & sampler,
	     eoContinue< EOT > & monitoring_continue,
	     eoEvalFunc < EOT > & evaluation,
	     moSolContinue < EOT > & continuator,
	     moCoolingSchedule & cooling_schedule,
	     double initial_temperature,
	     eoReplacement< EOT > & replacor
	     )
	: _selector(selector),
	  _estimator(estimator),
	  _selectone(selectone),
	  _modifier(modifier),
	  _sampler(sampler),
	  _monitoring_continue(monitoring_continue),
	  _evaluation(evaluation),
	  _continuator(continuator),
	  _cooling_schedule(cooling_schedule),
	  _initial_temperature(initial_temperature),
	  _replacor(replacor)
    {}

    //! function that launches the CMASA algorithm.
    /*!
      As a moTS or a moHC, the CMASA can be used for HYBRIDATION in an evolutionary algorithm.

      \param pop A population to improve.
      \return TRUE.
    */
    //bool operator ()(eoPop< EOT > & pop)
    void operator ()(eoPop< EOT > & pop)
    {
        assert(pop.size() > 0);

	double temperature = _initial_temperature;

	eoPop< EOT > current_pop = pop;

	eoPop< EOT > selected_pop;


	//-------------------------------------------------------------
	// Temporary solution used by plot to enumerate iterations in
	// several files.
	//-------------------------------------------------------------

	int number_of_iterations = 0;

	//-------------------------------------------------------------


	//-------------------------------------------------------------
	// Temporary solution to store populations state at each
	// iteration for plotting.
	//-------------------------------------------------------------

	std::string pop_results_destination("ResPop");

	{
	    std::stringstream ss;
	    ss << "rm -rf " << pop_results_destination;
	    ::system(ss.str().c_str());
	}

	::mkdir(pop_results_destination.c_str(), 0755); // create a first time the
	// directory where populations state are going to be stored.
	std::ofstream ofs_params("ResParams.txt");
	std::ofstream ofs_params_var("ResVars.txt");

	//-------------------------------------------------------------


	//-------------------------------------------------------------
	// Temporary solution to store bounds valeur for each distribution.
	//-------------------------------------------------------------

	std::string bounds_results_destination("ResBounds");

	{
	    std::stringstream ss;
	    ss << "rm -rf " << bounds_results_destination;
	    ::system(ss.str().c_str());
	}

	::mkdir(bounds_results_destination.c_str(), 0755); // create a first time the

	//-------------------------------------------------------------


	{
	    D distrib = _estimator(pop);

	    double size = distrib.size();
	    assert(size > 0);

	    doHyperVolume hv;

	    for (int i = 0; i < size; ++i)
		{
		    hv.update( distrib.variance()[i] );
		}

	    ofs_params_var << hv << std::endl;
	}

	do
	    {
		if (pop != current_pop)
		    {
			_replacor(pop, current_pop);
		    }

		current_pop.clear();
		selected_pop.clear();


		//-------------------------------------------------------------
		// (3) Selection of the best points in the population
		//-------------------------------------------------------------

		_selector(pop, selected_pop);

		//-------------------------------------------------------------


		_continuator.init();


		//-------------------------------------------------------------
		// (4) Estimation of the distribution parameters
		//-------------------------------------------------------------

		D distrib = _estimator(selected_pop);

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
		while ( _continuator( current_solution) );


		//-------------------------------------------------------------
		// Temporary solution to store populations state
		// at each iteration for plotting.
		//-------------------------------------------------------------

		{
		    std::stringstream ss;
		    ss << pop_results_destination << "/" << number_of_iterations;
		    std::ofstream ofs(ss.str().c_str());
		    ofs << current_pop;
		}

		//-------------------------------------------------------------


		//-------------------------------------------------------------
		// Temporary solution used by plot to enumerate iterations in
		// several files.
		//-------------------------------------------------------------

		{
		    double size = distrib.size();

		    assert(size > 0);

		    std::stringstream ss;
		    ss << bounds_results_destination << "/" << number_of_iterations;
		    std::ofstream ofs(ss.str().c_str());

		    ofs << size << " ";
		    std::copy(distrib.param(0).begin(), distrib.param(0).end(), std::ostream_iterator< double >(ofs, " "));
		    std::copy(distrib.param(1).begin(), distrib.param(1).end(), std::ostream_iterator< double >(ofs, " "));
		    ofs << std::endl;
		}

		//-------------------------------------------------------------


		//-------------------------------------------------------------
		// Temporary saving to get a proof for distribution bounds
		// dicreasement
		//-------------------------------------------------------------

		// {
		//     double size = distrib.size();
		//     assert(size > 0);

		//     vector< double > vmin(size);
		//     vector< double > vmax(size);

		//     std::copy(distrib.param(0).begin(), distrib.param(0).end(), vmin.begin());
		//     std::copy(distrib.param(1).begin(), distrib.param(1).end(), vmax.begin());

		//     vector< double > vrange = vmax - vmin;

		//     doHyperVolume hv;

		//     for (int i = 0, size = vrange.size(); i < size; ++i)
		//     	{
		// 	    hv.update( vrange(i) );
		//     	}

		//     ofs_params << hv << std::endl;
		// }

		//-------------------------------------------------------------


		//-------------------------------------------------------------
		// Temporary saving to get a proof for distribution bounds
		// dicreasement
		//-------------------------------------------------------------

		{
		    double size = distrib.size();
		    assert(size > 0);

		    doHyperVolume hv;

		    for (int i = 0; i < size; ++i)
			{
			    hv.update( distrib.variance()[i] );
			}

		    ofs_params_var << hv << std::endl;
		}

		//-------------------------------------------------------------

		++number_of_iterations;

	    }
	while ( _cooling_schedule( temperature ) &&
		_monitoring_continue( selected_pop ) );
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

    //! A EOT monitoring continuator
    eoContinue < EOT > & _monitoring_continue;

    //! A full evaluation function.
    eoEvalFunc < EOT > & _evaluation;

    //! Stopping criterion before temperature update
    moSolContinue < EOT > & _continuator;

    //! The cooling schedule
    moCoolingSchedule & _cooling_schedule;

    //! Initial temperature
    double  _initial_temperature;

    //! A EOT replacor
    eoReplacement < EOT > & _replacor;
};

#endif // !_doCMASA_h
