// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoIBEA.h"

// (c) OPAC Team, LIFL, June 2006

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: Arnaud.Liefooghe@lifl.fr
*/

#ifndef _eoIBEASorting_h
#define _eoIBEASorting_h

#include <math.h>
#include <list>
#include <eoPop.h>
#include <eoPerf2Worth.h>
#include "eoBinaryQualityIndicator.h"


/**
 * Functor
 * The sorting phase of IBEA (Indicator-Based Evolutionary Algorithm)
 */
template < class EOT, class Fitness > class eoIBEA:public eoPerf2WorthCached < EOT,
  double >
{

  public:
  /** values */
  using eoIBEA < EOT, Fitness >::value;

  eoIBEA (eoBinaryQualityIndicator < Fitness > *_I)
  {
    I = _I;
  }


  /**
   * mapping
   * @param const eoPop<EOT>& _pop  the population
   */
  void calculate_worths (const eoPop < EOT > &_pop)
  {
    /* resizing the worths beforehand */
    value ().resize (_pop.size ());

    /* computation and setting of the bounds for each objective */
    setBounds (_pop);

    /* computation of the fitness for each individual */
    fitnesses (_pop);

    // higher is better, so invert the value
    double max = *std::max_element (value ().begin (), value ().end ());
    for (unsigned i = 0; i < value ().size (); i++)
      value ()[i] = max - value ()[i];
  }


protected:

  /** binary quality indicator to use in the selection process */
  eoBinaryQualityIndicator < Fitness > *I;

  virtual void setBounds (const eoPop < EOT > &_pop) = 0;
  virtual void fitnesses (const eoPop < EOT > &_pop) = 0;

};





/**
 * Functor
 * The sorting phase of IBEA (Indicator-Based Evolutionary Algorithm) without uncertainty
 * Adapted from the Zitzler and Künzli paper "Indicator-Based Selection in Multiobjective Search" (2004) 
 * Of course, Fitness needs to be an eoParetoFitness object
 */
template < class EOT, class Fitness = typename EOT::Fitness > class eoIBEASorting:public eoIBEA < EOT,
  Fitness
  >
{

public:

  /**
   * constructor
   * @param eoBinaryQualityIndicator<EOT>* _I  the binary quality indicator to use in the selection process
   * @param double _kappa  scaling factor kappa
   */
  eoIBEASorting (eoBinaryQualityIndicator < Fitness > *_I,
		 const double _kappa):
    eoIBEA <
    EOT,
  Fitness > (_I)
  {
    kappa = _kappa;
  }


private:
  /** quality indicator */
  using eoIBEASorting < EOT, Fitness >::I;
  /** values */
  using
    eoIBEA <
    EOT,
    Fitness >::value;
  /** scaling factor kappa */
  double
    kappa;


  /**
   * computation and setting of the bounds for each objective
   * @param const eoPop<EOT>& _pop  the population
   */
  void
  setBounds (const eoPop < EOT > &_pop)
  {
    typedef typename
      EOT::Fitness::fitness_traits
      traits;
    double
      min,
      max;
    for (unsigned i = 0; i < traits::nObjectives (); i++)
      {
	min = _pop[0].fitness ()[i];
	max = _pop[0].fitness ()[i];
	for (unsigned j = 1; j < _pop.size (); j++)
	  {
	    min = std::min (min, _pop[j].fitness ()[i]);
	    max = std::max (max, _pop[j].fitness ()[i]);
	  }
	// setting of the bounds for the objective i
	I->setBounds (i, min, max);
      }
  }


  /**
   * computation and setting of the fitness for each individual of the population
   * @param const eoPop<EOT>& _pop  the population
   */
  void
  fitnesses (const eoPop < EOT > &_pop)
  {
    // reprsentation of the fitness components
    std::vector < std::vector < double > >
    fitComponents (_pop.size (), _pop.size ());
    // the maximum absolute indicator value
    double
      maxAbsoluteIndicatorValue = 0;

    // computation of the indicator values and of the maximum absolute indicator value
    for (unsigned i = 0; i < _pop.size (); i++)
      for (unsigned j = 0; j < _pop.size (); j++)
	if (i != j)
	  {
	    fitComponents[i][j] =
	      (*I) (_pop[i].fitness (), _pop[j].fitness ());
	    maxAbsoluteIndicatorValue =
	      std::max (maxAbsoluteIndicatorValue,
			fabs (fitComponents[i][j]));
	  }

    // computation of the fitness components for each pair of individuals
    // if maxAbsoluteIndicatorValue==0, every individuals have the same fitness values for all objectives (already = 0)
    if (maxAbsoluteIndicatorValue != 0)
      for (unsigned i = 0; i < _pop.size (); i++)
	for (unsigned j = 0; j < _pop.size (); j++)
	  if (i != j)
	    fitComponents[i][j] =
	      exp (-fitComponents[i][j] /
		   (maxAbsoluteIndicatorValue * kappa));

    // computation of the fitness for each individual
    for (unsigned i = 0; i < _pop.size (); i++)
      {
	value ()[i] = 0;
	for (unsigned j = 0; j < _pop.size (); j++)
	  if (i != j)
	    value ()[i] += fitComponents[j][i];
      }
  }

};





/**
 * Functor
 * The sorting phase of IBEA (Indicator-Based Evolutionary Algorithm) under uncertainty
 * Adapted from the Basseur and Zitzler paper "Handling Uncertainty in Indicator-Based Multiobjective Optimization" (2006) 
 * Of course, the fitness of an individual needs to be an eoStochasticParetoFitness object
 */
template < class EOT, class FitnessEval = typename EOT::Fitness::FitnessEval > class eoIBEAStochSorting:public eoIBEA < EOT,
  FitnessEval
  >
{

public:

  /**
   * constructor
   * @param eoBinaryQualityIndicator<EOT>* _I  the binary quality indicator to use in the selection process
   */
eoIBEAStochSorting (eoBinaryQualityIndicator < FitnessEval > *_I):eoIBEA < EOT,
    FitnessEval >
    (_I)
  {
  }


private:
  /** quality indicator */
  using eoIBEAStochSorting < EOT, FitnessEval >::I;
  /** values */
  using
    eoIBEAStochSorting <
    EOT,
    FitnessEval >::value;


  /**
   * approximated zero value
   */
  static double
  zero ()
  {
    return 1e-7;
  }


  /**
   * computation and setting of the bounds for each objective
   * @param const eoPop<EOT>& _pop  the population
   */
  void
  setBounds (const eoPop < EOT > &_pop)
  {
    typedef typename
      EOT::Fitness::FitnessTraits
      traits;
    double
      min,
      max;
    for (unsigned i = 0; i < traits::nObjectives (); i++)
      {
	min = _pop[0].fitness ().minimum (i);
	max = _pop[0].fitness ().maximum (i);
	for (unsigned j = 1; j < _pop.size (); j++)
	  {
	    min = std::min (min, _pop[j].fitness ().minimum (i));
	    max = std::max (max, _pop[j].fitness ().maximum (i));
	  }
	// setting of the bounds for the ith objective
	I->setBounds (i, min, max);
      }
  }


  /**
   * computation and setting of the fitness for each individual of the population
   * @param const eoPop<EOT>& _pop  the population
   */
  void
  fitnesses (const eoPop < EOT > &_pop)
  {
    typedef typename
      EOT::Fitness::FitnessTraits
      traits;
    unsigned
      nEval = traits::nEvaluations ();
    unsigned
      index;
    double
      eiv,
      p,
      sumP,
      iValue;
    std::list < std::pair < double, unsigned > > l;
    std::vector < unsigned >
    n (_pop.size ());

    for (unsigned ind = 0; ind < _pop.size (); ind++)
      {
	value ()[ind] = 0.0;	// fitness value for the individual ind
	for (unsigned eval = 0; eval < nEval; eval++)
	  {

	    // I-values computation for the evaluation eval of the individual ind
	    l.clear ();
	    for (unsigned i = 0; i < _pop.size (); i++)
	      {
		if (i != ind)
		  {
		    for (unsigned j = 0; j < nEval; j++)
		      {
			std::pair < double, unsigned >
			  pa;
			// I-value
			pa.first =
			  (*I) (_pop[ind].fitness ()[eval],
				_pop[i].fitness ()[j]);
			// index of the individual
			pa.second = i;
			// append this to the list
			l.push_back (pa);
		      }
		  }
	      }

	    // sorting of the I-values (in decreasing order)
	    l.sort ();

	    // computation of the Expected Indicator Value (eiv) for the evaluation eval of the individual ind
	    eiv = 0.0;
	    n.assign (n.size (), 0);	// n[i]==0 for all i
	    sumP = 0.0;
	    while (((1 - sumP) > zero ()) && (l.size () > 0))
	      {
		// we use the last element of the list (the greatest one)
		iValue = l.back ().first;
		index = l.back ().second;
		// computation of the probability to appear
		p = (1.0 / (nEval - n[index])) * (1.0 - sumP);
		// eiv update
		eiv += p * iValue;
		// update of the number of elements for individual index
		n[index]++;
		// removing the last element of the list
		l.pop_back ();
		// sum of p update
		sumP += p;
	      }
	    value ()[ind] += eiv / nEval;
	  }
      }

  }

};





/**
 * Functor
 * The sorting phase of IBEA (Indicator-Based Evolutionary Algorithm) under uncertainty using averaged values for each objective
 * Follow the idea presented in the Deb & Gupta paper "Searching for Robust Pareto-Optimal Solutions in Multi-Objective Optimization", 2005
 * Of course, the fitness of an individual needs to be an eoStochasticParetoFitness object
 */
template < class EOT, class FitnessEval = typename EOT::Fitness::FitnessEval > class eoIBEAAvgSorting:public eoIBEA < EOT,
  FitnessEval
  >
{

public:

  /**
   * constructor
   * @param eoBinaryQualityIndicator<EOT>* _I  the binary quality indicator to use in the selection process
   * @param double _kappa  scaling factor kappa
   */
  eoIBEAAvgSorting (eoBinaryQualityIndicator < FitnessEval > *_I,
		    const double _kappa):
    eoIBEA <
    EOT,
  FitnessEval > (_I)
  {
    kappa = _kappa;
  }


private:
  /** quality indicator */
  using eoIBEAAvgSorting < EOT, FitnessEval >::I;
  /** values */
  using
    eoIBEAAvgSorting <
    EOT,
    FitnessEval >::value;
  /** scaling factor kappa */
  double
    kappa;


  /**
   * computation and setting of the bounds for each objective
   * @param const eoPop<EOT>& _pop  the population
   */
  void
  setBounds (const eoPop < EOT > &_pop)
  {
    typedef typename
      EOT::Fitness::FitnessTraits
      traits;
    double
      min,
      max;
    for (unsigned i = 0; i < traits::nObjectives (); i++)
      {
	min = _pop[0].fitness ().averagedParetoFitnessObject ()[i];
	max = _pop[0].fitness ().averagedParetoFitnessObject ()[i];
	for (unsigned j = 1; j < _pop.size (); j++)
	  {
	    min =
	      std::min (min,
			_pop[j].fitness ().averagedParetoFitnessObject ()[i]);
	    max =
	      std::max (max,
			_pop[j].fitness ().averagedParetoFitnessObject ()[i]);
	  }
	// setting of the bounds for the objective i
	I->setBounds (i, min, max);
      }
  }


  /**
   * computation and setting of the fitness for each individual of the population
   * @param const eoPop<EOT>& _pop  the population
   */
  void
  fitnesses (const eoPop < EOT > &_pop)
  {
    // reprsentation of the fitness components
    std::vector < std::vector < double > >
    fitComponents (_pop.size (), _pop.size ());
    // the maximum absolute indicator value
    double
      maxAbsoluteIndicatorValue = 0;

    // computation of the indicator values and of the maximum absolute indicator value
    for (unsigned i = 0; i < _pop.size (); i++)
      for (unsigned j = 0; j < _pop.size (); j++)
	if (i != j)
	  {
	    fitComponents[i][j] =
	      (*I) (_pop[i].fitness ().averagedParetoFitnessObject (),
		    _pop[j].fitness ().averagedParetoFitnessObject ());
	    maxAbsoluteIndicatorValue =
	      std::max (maxAbsoluteIndicatorValue,
			fabs (fitComponents[i][j]));
	  }

    // computation of the fitness components for each pair of individuals
    // if maxAbsoluteIndicatorValue==0, every individuals have the same fitness values for all objectives (already = 0)
    if (maxAbsoluteIndicatorValue != 0)
      for (unsigned i = 0; i < _pop.size (); i++)
	for (unsigned j = 0; j < _pop.size (); j++)
	  if (i != j)
	    fitComponents[i][j] =
	      exp (-fitComponents[i][j] /
		   (maxAbsoluteIndicatorValue * kappa));

    // computation of the fitness for each individual
    for (unsigned i = 0; i < _pop.size (); i++)
      {
	value ()[i] = 0;
	for (unsigned j = 0; j < _pop.size (); j++)
	  if (i != j)
	    value ()[i] += fitComponents[j][i];
      }
  }

};


#endif
