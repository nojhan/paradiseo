// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRealOp.h
// (c) EEAAX 2000 - Maarten Keijzer 2000
/* 
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoRealOp_h
#define eoRealOp_h

//-----------------------------------------------------------------------------

#include <algorithm>    // swap_ranges
#include <utils/eoRNG.h>
#include <es/eoReal.h>
#include <es/eoRealBounds.h>

//-----------------------------------------------------------------------------

/** eoUniformMutation --> changes all values of the vector
                          by uniform choice with range epsilon
                          with probability p_change per variable
\class eoUniformMutation eoRealOp.h Tutorial/eoRealOp.h
\ingroup parameteric
*/

template<class EOT> class eoUniformMutation: public eoMonOp<EOT>
{
 public:
  /**
   * (Default) Constructor.
   * The bounds are initialized with the global object that says: no bounds.
   *
   * @param _epsilon the range for uniform nutation
   * @param _p_change the probability to change a given coordinate
   */
  eoUniformMutation(const double& _epsilon, const double& _p_change = 1.0):
    bounds(eoDummyVectorNoBounds), epsilon(_epsilon), p_change(_p_change) {}

  /**
   * Constructor with bounds
   * @param _bounds an eoRealVectorBounds that contains the bounds
   * @param _epsilon the range for uniform nutation
   * @param _p_change the probability to change a given coordinate
   */
  eoUniformMutation(eoRealVectorBounds & _bounds,
		    const double& _epsilon, const double& _p_change = 1.0):
    bounds(_bounds), epsilon(_epsilon), p_change(_p_change) {}

  /// The class name.
  string className() const { return "eoUniformMutation"; }

  /**
   * Do it!
   * @param _eo The cromosome undergoing the mutation
   */
  bool operator()(EOT& _eo)
    {
      bool hasChanged=false;
      for (unsigned lieu=0; lieu<_eo.size(); lieu++)
	{
	  if (rng.flip(p_change))
	    {
	      // check the bounds
	      double emin = _eo[lieu]-epsilon;
	      double emax = _eo[lieu]+epsilon;
	      if (bounds.isMinBounded(lieu))
		emin = max(bounds.minimum(lieu), emin);
	      if (bounds.isMaxBounded(lieu))
		emax = min(bounds.maximum(lieu), emax);
	      _eo[lieu] = emin + (emax-emin)*rng.uniform();
	      hasChanged = true;
	    }
	}
      return hasChanged;
    }

private:
  eoRealVectorBounds & bounds;
  double epsilon;
  double p_change;
};

/** eoDetUniformMutation --> changes exactly k values of the vector
                          by uniform choice with range epsilon
\class eoDetUniformMutation eoRealOp.h Tutorial/eoRealOp.h
\ingroup parameteric
*/

template<class EOT> class eoDetUniformMutation: public eoMonOp<EOT>
{
 public:
  /**
   * (Default) Constructor.
   * @param _epsilon the range for uniform nutation
   * @param number of coordinate to modify
   */
  eoDetUniformMutation(const double& _epsilon, const unsigned& _no = 1):
    bounds(eoDummyVectorNoBounds), epsilon(_epsilon), no(_no) {}

  /**
   * Constructor with bounds
   * @param _bounds an eoRealVectorBounds that contains the bounds
   * @param _epsilon the range for uniform nutation
   * @param number of coordinate to modify
   */
  eoDetUniformMutation(eoRealVectorBounds & _bounds, 
		       const double& _epsilon, const unsigned& _no = 1): 
    bounds(_bounds), epsilon(_epsilon), no(_no) {}

  /// The class name.
  string className() const { return "eoDetUniformMutation"; }
  
  /**
   * Do it!
   * @param _eo The cromosome undergoing the mutation
   */
  bool operator()(EOT& _eo)
    {
      for (unsigned i=0; i<no; i++)
	    {
	      unsigned lieu = rng.random(_eo.size());
	  // actually, we should test that we don't re-modify same variable!

	      // check the bounds
	      double emin = _eo[lieu]-epsilon;
	      double emax = _eo[lieu]+epsilon;
	      if (bounds.isMinBounded(lieu))
		emin = max(bounds.minimum(lieu), emin);
	      if (bounds.isMaxBounded(lieu))
		emax = min(bounds.maximum(lieu), emax);
	      _eo[lieu] = emin + (emax-emin)*rng.uniform();
	    }

      return true;
    }

private:
  eoRealVectorBounds & bounds;
  double epsilon;
  unsigned no;
};


// two arithmetical crossovers

/** eoSegmentCrossover --> uniform choice in segment
                 == arithmetical with same value along all coordinates
\class eoSegmentCrossover eoRealOp.h Tutorial/eoRealOp.h
\ingroup parameteric
*/

template<class EOT> class eoSegmentCrossover: public eoQuadOp<EOT>
{
 public:
  /**
   * (Default) Constructor.
   * The bounds are initialized with the global object that says: no bounds.
   *
   * @param _alphaMin the amount of exploration OUTSIDE the parents
   *               as in BLX-alpha notation (Eshelman and Schaffer)
   *               0 == contractive application
   *               Must be positive
   */
  eoSegmentCrossover(const double& _alpha = 0.0) :
    bounds(eoDummyVectorNoBounds), alpha(_alpha), range(1+2*_alpha) {}

  /**
   * Constructor with bounds
   * @param _bounds an eoRealVectorBounds that contains the bounds
   * @param _alphaMin the amount of exploration OUTSIDE the parents 
   *               as in BLX-alpha notation (Eshelman and Schaffer)
   *               0 == contractive application
   *               Must be positive
   */
  eoSegmentCrossover(eoRealVectorBounds & _bounds, 
		     const double& _alpha = 0.0) : 
    bounds(_bounds), alpha(_alpha), range(1+2*_alpha) {}

  /// The class name.
  string className() const { return "eoSegmentCrossover"; }

  /**
   * segment crossover - modifies both parents
   * @param _eo1 The first parent
   * @param _eo2 The first parent
   */
  bool operator()(EOT& _eo1, EOT& _eo2)
    {
      unsigned i;
      double r1, r2, fact;
      double alphaMin = -alpha;
      double alphaMax = 1+alpha;
      if (alpha == 0.0)		   // no check to perform
	fact = -alpha + rng.uniform(range); // in [-alpha,1+alpha)
      else			   // look for the bounds for fact
	{
	  for (i=0; i<_eo1.size(); i++)
	    {
	      r1=_eo1[i];
	      r2=_eo2[i];
	      if (r1 != r2) {	   // otherwise you'll get NAN's
		double rmin = min(r1, r2);
		double rmax = max(r1, r2);
		double length = rmax - rmin;
		if (bounds.isMinBounded(i))
		  {
		    alphaMin = max(alphaMin, (bounds.minimum(i)-rmin)/length);
		    alphaMin = max(alphaMin, (rmax-bounds.maximum(i))/length);
		  }
		if (bounds.isMaxBounded(i))
		  {
		    alphaMax = min(alphaMax, (bounds.maximum(i)-rmin)/length);
		    alphaMax = min(alphaMax, (rmax-bounds.minimum(i))/length);
		  }
	      }
	    }
	  fact = alphaMin + (alphaMax-alphaMin)*rng.uniform();
	}

      for (i=0; i<_eo1.size(); i++)
	{
	  r1=_eo1[i];
	  r2=_eo2[i];
	  _eo1[i] = fact * r1 + (1-fact) * r2;
	  _eo2[i] = (1-fact) * r1 + fact * r2;
	}
      return true;	   // shoudl test if fact was 0 or 1 :-)))
    }

protected:
  eoRealVectorBounds & bounds;
  double alpha;
  double range;			   // == 1+2*alpha
};

/** eoArithmeticCrossover --> uniform choice in hypercube
                 == arithmetical with different values for each coordinate
\class eoArithmeticCrossover eoRealOp.h Tutorial/eoRealOp.h
\ingroup parameteric
*/

template<class EOT> class eoArithmeticCrossover: public eoQuadOp<EOT>
{
 public:
  /**
   * (Default) Constructor.
   * The bounds are initialized with the global object that says: no bounds.
   *
   * @param _alphaMin the amount of exploration OUTSIDE the parents
   *               as in BLX-alpha notation (Eshelman and Schaffer)
   *               0 == contractive application
   *               Must be positive
   */
  eoArithmeticCrossover(const double& _alpha = 0.0):
    bounds(eoDummyVectorNoBounds), alpha(_alpha), range(1+2*_alpha)
  {
    if (_alpha < 0)
      throw runtime_error("BLX coefficient should be positive");
  }

  /**
   * Constructor with bounds
   * @param _bounds an eoRealVectorBounds that contains the bounds
   * @param _alphaMin the amount of exploration OUTSIDE the parents 
   *               as in BLX-alpha notation (Eshelman and Schaffer)
   *               0 == contractive application
   *               Must be positive
   */
  eoArithmeticCrossover(eoRealVectorBounds & _bounds, 
			const double& _alpha = 0.0): 
    bounds(_bounds), alpha(_alpha), range(1+2*_alpha) 
  {
    if (_alpha < 0)
      throw runtime_error("BLX coefficient should be positive");
  }

  /// The class name.
  string className() const { return "eoArithmeticCrossover"; }

  /**
   * arithmetical crossover - modifies both parents
   * @param _eo1 The first parent
   * @param _eo2 The first parent
   */
  bool operator()(EOT& _eo1, EOT& _eo2)
    {
      unsigned i;
      double r1, r2, fact;
      if (alpha == 0.0)		   // no check to perform
	  for (i=0; i<_eo1.size(); i++)
	      {
		r1=_eo1[i];
		r2=_eo2[i];
	    fact = -alpha + rng.uniform(range);	 // in [-alpha,1+alpha)
	    _eo1[i] = fact * r1 + (1-fact) * r2;
	    _eo2[i] = (1-fact) * r1 + fact * r2;
	  }
      else			   // check the bounds
	for (i=0; i<_eo1.size(); i++)
	  {
	    r1=_eo1[i];
	    r2=_eo2[i];
	    if (r1 != r2) {	   // otherwise you'll get NAN's
	      double rmin = min(r1, r2);
	      double rmax = max(r1, r2);
	      double length = rmax - rmin;
	      double alphaMin = -alpha;
	      double alphaMax = 1+alpha;
	      // first find the limits on the alpha's
	      if (bounds.isMinBounded(i))
		{
		  alphaMin = max(alphaMin, (bounds.minimum(i)-rmin)/length);
		  alphaMin = max(alphaMin, (rmax-bounds.maximum(i))/length);
		}
	      if (bounds.isMaxBounded(i))
		{
		  alphaMax = min(alphaMax, (bounds.maximum(i)-rmin)/length);
		  alphaMax = min(alphaMax, (rmax-bounds.minimum(i))/length);
		}
	      fact = alphaMin + rng.uniform(alphaMax-alphaMin);
	      _eo1[i] = fact * rmin + (1-fact) * rmax;
	      _eo2[i] = (1-fact) * rmin + fact * rmax;
	    }
	  }

    return true;
   }

protected:
  eoRealVectorBounds & bounds;
  double alpha;
  double range;			   // == 1+2*alphaMin
};


/** eoRealUxOver --> Uniform crossover, also termed intermediate crossover
\class eoRealUxOver eoRealOp.h Tutorial/eoRealOp.h
\ingroup parameteric
*/

template<class EOT> class eoRealUxOver: public eoQuadOp<EOT>
{
 public:
  /**
   * (Default) Constructor.
   * @param _preference bias in the choice (usually, no bias == 0.5)
   */
  eoRealUxOver(const float& _preference = 0.5): preference(_preference)
    { 
      if ( (_preference <= 0.0) || (_preference >= 1.0) )
	runtime_error("UxOver --> invalid preference");
    }

  /// The class name.
  string className() const { return "eoRealUxOver"; }

  /**
   * Uniform crossover for real vectors
   * @param _eo1 The first parent
   * @param _eo2 The second parent
   *    @runtime_error if sizes don't match
   */
  bool operator()(EOT& _eo1, EOT& _eo2)
    {
      if ( _eo1.size() != _eo2.size())
	    runtime_error("UxOver --> chromosomes sizes don't match" );
      bool changed = false;
      for (unsigned int i=0; i<_eo1.size(); i++)
	{
	  if (rng.flip(preference))
	    if (_eo1[i] == _eo2[i])
	      {
		double tmp = _eo1[i];
	      _eo1[i]=_eo2[i];
	      _eo2[i] = tmp;
	      changed = true;
	    }
	}
      return changed;
    }
    private:
      float preference;
};
  

//-----------------------------------------------------------------------------
//@}
#endif eoRealOp_h

