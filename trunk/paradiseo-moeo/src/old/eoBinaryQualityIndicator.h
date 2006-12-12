// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoBinaryQualityIndicator.h"

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

#ifndef _eoBinaryQualityIndicator_h
#define _eoBinaryQualityIndicator_h

// for std::exceptions
#include <stdexcept>
// for eoBF
#include <eoFunctor.h>


/**
 * Functor
 * Binary quality indicator
 * Binary performance measure to use in the replacement selection process of IBEA (Indicator-Based Evolutionary Algorithm)
 * Of course, EOFitness needs to be an eoParetoFitness object
 */
template < class EOFitness > class eoBinaryQualityIndicator:public eoBF < const EOFitness &, const EOFitness &,
  double >
{

public:

  /**
   * constructor  
   */
  eoBinaryQualityIndicator ():eoBF < const EOFitness &, const EOFitness &,
    double >()
  {
    bounds.reserve (traits::nObjectives ());
    bounds.resize (traits::nObjectives ());
  }


  /**
   * set the bounds for objective _iObj
   * @param unsigned _iObj  the index of the objective
   * @param double _min  the minimum value
   * @param double _max  the maximum value
   */
  void setBounds (const unsigned _iObj, const double _min, const double _max)
  {
    bounds[_iObj] = Range (_min, _max);
  }


protected:

  /**
   * Private class to represent the bounds
   */
  class Range
  {
  public:
    Range ()
    {
      min = 0;
      max = 0;
      r = 0;
    }
    Range (const double _min, const double _max)
    {
      min = _min;
      max = _max;
      r = max - min;
      if (r < 0)
	throw std::logic_error ("Negative range in eoBinaryQualityIndicator");
    }
    double minimum ()
    {
      return min;
    }
    double maximum ()
    {
      return max;
    }
    double range ()
    {
      return r;
    }
  private:
    double min, max, r;
  };


  /** range (min and max double value) for each objective */
  std::vector < Range > bounds;


private:

  /** fitness traits */
  typedef typename EOFitness::fitness_traits traits;

};





/**
 * Functor
 * Additive binary epsilon indicator for eoParetoFitness
 */
template < class EOFitness > class eoAdditiveBinaryEpsilonIndicator:public eoBinaryQualityIndicator <
  EOFitness
  >
{

public:

  /**
   * constructor  
   */
eoAdditiveBinaryEpsilonIndicator ():eoBinaryQualityIndicator < EOFitness >
    ()
  {
  }


  /** 
   * computation of the maximum epsilon value by which individual _eo1 must be
   * decreased in all objectives such that individual _eo2 is weakly dominated
   * (do not forget to set the bounds before the call of this function)
   * @param EOFitness & _fitness_eo1  the fitness of the first individual
   * @param EOFitness & _fitness_eo2  the fitness of the second individual
   */
  double operator () (const EOFitness & _fitness_eo1,
		      const EOFitness & _fitness_eo2)
  {
    double epsilon, tmp;
    // computation of the epsilon value for the first objective
    epsilon = epsilonValue (_fitness_eo1, _fitness_eo2, 0);
    // computation of the epsilon value for other objectives
    for (unsigned i = 1; i < traits::nObjectives (); i++)
      {
	tmp = epsilonValue (_fitness_eo1, _fitness_eo2, i);
	epsilon = std::max (epsilon, tmp);
      }
    // the maximum epsilon value
    return epsilon;
  }


private:

  /** fitness traits */
  typedef typename EOFitness::fitness_traits traits;
  /** bounds */
  using eoAdditiveBinaryEpsilonIndicator < EOFitness >::bounds;


  /**
   * computation of the epsilon value by which individual _eo1 must be
   * decreased in the objective _iObj such that individual _eo2 is weakly dominated
   * @param EOFitness & _fitness_eo1  the fitness of the first individual
   * @param EOFitness & _fitness_eo2  the fitness of the second individual
   * @param unsigned _iObj  the index of the objective
   */
  double epsilonValue (const EOFitness & _fitness_eo1,
		       const EOFitness & _fitness_eo2, const unsigned _iObj)
  {
    double result;
    if (bounds[_iObj].range () == 0)
      {
	// min==max => every individuals has the same value for this objective      
	result = 0;
      }
    else
      {
	// computation of the epsilon value for the objective _iObj (in case of a minimization)
	result =
	  (_fitness_eo1[_iObj] -
	   bounds[_iObj].minimum ()) / bounds[_iObj].range ();
	result -=
	  (_fitness_eo2[_iObj] -
	   bounds[_iObj].minimum ()) / bounds[_iObj].range ();
	// if we are maximizing, invert the value
	if (traits::maximizing (_iObj))
	  result = -result;
      }
    // the espilon value
    return result;
  }

};





/**
 * Functor
 * Binary hypervolume indicator for eoParetoFitness
 */
template < class EOFitness > class eoBinaryHypervolumeIndicator:public eoBinaryQualityIndicator <
  EOFitness >
{

public:

  /**
   * constructor
   * @param double _rho  reference point for the hypervolume calculation (rho must not be smaller than 1)
   */
eoBinaryHypervolumeIndicator (double _rho):eoBinaryQualityIndicator < EOFitness >
    ()
  {
    rho = _rho;
    // consistency check
    if (rho < 1)
      {
	cout <<
	  "Warning, reference point rho for the hypervolume calculation must not be smaller than 1"
	  << endl;
	cout << "Adjusted to 1" << endl;
	rho = 1;
      }
  }


  /**
   * indicator value of the hypervolume of the portion of the objective space
   * that is dominated by individual _eo1 but not by individual _eo2
   * (don't forget to set the bounds before the call of this function)
   * @param EOFitness & _fitness_eo1  the fitness of the first individual
   * @param EOFitness & _fitness_eo2  the fitness of the second individual
   */
  double operator () (const EOFitness & _fitness_eo1,
		      const EOFitness & _fitness_eo2)
  {
    double result;
    if (_fitness_eo1.dominates (_fitness_eo2))
      result =
	-hypervolumeIndicatorValue (_fitness_eo1, _fitness_eo2,
				    traits::nObjectives ());
    else
      result =
	hypervolumeIndicatorValue (_fitness_eo2, _fitness_eo1,
				   traits::nObjectives ());
    return result;
  }


private:

  /** fitness traits */
  typedef typename EOFitness::fitness_traits traits;
  /** bounds */
  using eoBinaryHypervolumeIndicator < EOFitness >::bounds;

  /** reference point for the hypervolume calculation */
  double rho;


  /**
   * computation of the hypervolume of the portion of the objective space
   * that is dominated by individual _eo1 but not by individual _eo2
   * @param EOFitness & _fitness_eo1  the fitness of the first individual
   * @param EOFitness & _fitness_eo2  the fitness of the second individual
   * @param unsigned _iObj  number of objectives (used for iteration)
   * @param bool _flag = false  (only used for iteration)
   */
  double hypervolumeIndicatorValue (const EOFitness & _fitness_eo1,
				    const EOFitness & _fitness_eo2,
				    const unsigned _iObj, const bool _flag =
				    false)
  {
    double result;
    if (bounds[_iObj - 1].range () == 0)
      {
	// min==max => every individuals as the same value for this objective      
	result = 0;
      }
    else
      {
	if (traits::maximizing (_iObj - 1))	// maximizing
	  result =
	    hypervolumeIndicatorValueMax (_fitness_eo1, _fitness_eo2, _iObj,
					  _flag);
	else			// minimizing
	  result =
	    hypervolumeIndicatorValueMin (_fitness_eo1, _fitness_eo2, _iObj,
					  _flag);
      }
    return result;
  }


  /**
   * computation of the hypervolume of the portion of the objective space
   * that is dominated by individual _eo1 but not by individual _eo2
   * in case of a minimization on the objective _iObj
   * @param EOFitness & _fitness_eo1  the fitness of the first individual
   * @param EOFitness & _fitness_eo2  the fitness of the second individual
   * @param unsigned _iObj  index of the objective
   * @param bool _flag  (only used for iteration)
   */
  double hypervolumeIndicatorValueMin (const EOFitness & _fitness_eo1,
				       const EOFitness & _fitness_eo2,
				       const unsigned _iObj, const bool _flag)
  {
    double result;
    double r = rho * bounds[_iObj - 1].range ();
    double max = bounds[_iObj - 1].minimum () + r;
    // fitness of individuals _eo1 and _eo2 for the objective _iObj (if flag==true, _eo2 is not taken into account)
    double fitness_eo1 = _fitness_eo1[_iObj - 1];
    double fitness_eo2;
    if (_flag)
      fitness_eo2 = max;
    else
      fitness_eo2 = _fitness_eo2[_iObj - 1];
    // computation of the volume
    if (_iObj == 1)
      {
	if (fitness_eo1 < fitness_eo2)
	  result = (fitness_eo2 - fitness_eo1) / r;
	else
	  result = 0;
      }
    else
      {
	if (fitness_eo1 < fitness_eo2)
	  {
	    result =
	      hypervolumeIndicatorValue (_fitness_eo1, _fitness_eo2,
					 _iObj - 1) * (max - fitness_eo2) / r;
	    result +=
	      hypervolumeIndicatorValue (_fitness_eo1, _fitness_eo2,
					 _iObj - 1,
					 true) * (fitness_eo2 -
						  fitness_eo1) / r;
	  }
	else
	  result =
	    hypervolumeIndicatorValue (_fitness_eo1, _fitness_eo2,
				       _iObj - 1) * (max - fitness_eo2) / r;
      }
    // the volume
    return result;
  }


  /**
   * computation of the hypervolume of the portion of the objective space
   * that is dominated by individual _eo1 but not by individual _eo2
   * in case of a maximization on the objective _iObj
   * @param EOFitness & _fitness_eo1  the fitness of the first individual
   * @param EOFitness & _fitness_eo2  the fitness of the second individual
   * @param unsigned _iObj  index of the objective
   * @param bool _flag  (only used for iteration)
   */
  double hypervolumeIndicatorValueMax (const EOFitness & _fitness_eo1,
				       const EOFitness & _fitness_eo2,
				       const unsigned _iObj, const bool _flag)
  {
    double result;
    double r = rho * bounds[_iObj - 1].range ();
    double min = bounds[_iObj - 1].maximum () - r;
    // fitness of individuals _eo1 and _eo2 for the objective _iObj (if flag==true, _eo2 is not taken into account)
    double fitness_eo1 = _fitness_eo1[_iObj - 1];
    double fitness_eo2;
    if (_flag)
      fitness_eo2 = min;
    else
      fitness_eo2 = _fitness_eo2[_iObj - 1];
    // computation of the volume
    if (_iObj == 1)
      {
	if (fitness_eo1 > fitness_eo2)
	  result = (fitness_eo1 - fitness_eo2) / r;
	else
	  result = 0;
      }
    else
      {
	if (fitness_eo1 > fitness_eo2)
	  {
	    result =
	      hypervolumeIndicatorValue (_fitness_eo1, _fitness_eo2,
					 _iObj - 1) * (fitness_eo2 - min) / r;
	    result +=
	      hypervolumeIndicatorValue (_fitness_eo1, _fitness_eo2,
					 _iObj - 1,
					 true) * (fitness_eo1 -
						  fitness_eo2) / r;
	  }
	else
	  result =
	    hypervolumeIndicatorValue (_fitness_eo1, _fitness_eo2,
				       _iObj - 1) * (fitness_eo2 - min) / r;
      }
    // the volume
    return result;
  }

};

#endif
