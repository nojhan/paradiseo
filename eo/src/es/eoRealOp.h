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
   * @param _epsilon the range for uniform nutation
   * @param _p_change the probability to change a given coordinate
   */
  eoUniformMutation(const double& _epsilon, const double& _p_change = 1.0): 
    epsilon(_epsilon), p_change(_p_change) {}

  /// The class name.
  string className() const { return "eoUniformMutation"; }
  
  /**
   * Do it!
   * @param _eo The cromosome undergoing the mutation
   */
  void operator()(EOT& _eo) 
    {
      bool hasChanged=false;
      for (unsigned lieu=0; lieu<_eo.size(); lieu++) 
	{
	  if (rng.flip(p_change)) 
	    {
	      _eo[lieu] += 2*epsilon*rng.uniform()-epsilon;
	      hasChanged = true;
	    }
	}
      if (hasChanged)
	_eo.invalidate();
    }
  
private:
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
    epsilon(_epsilon), no(_no) {}

  /// The class name.
  string className() const { return "eoDetUniformMutation"; }
  
  /**
   * Do it!
   * @param _eo The cromosome undergoing the mutation
   */
  void operator()(EOT& _eo) 
    {
      _eo.invalidate();
      for (unsigned i=0; i<no; i++)
	{
	  unsigned lieu = rng.random(_eo.size());
	  // actually, we should test that we don't re-modify same variable!
	  _eo[lieu] += 2*epsilon*rng.uniform()-epsilon;
	}
    }

private:
  double epsilon;
  unsigned no;
};


// two arithmetical crossovers

/** eoSegmentCrossover --> uniform choice in segment 
                 == arithmetical with same value along all coordinates
\class eoSegmentCrossover eoRealOp.h Tutorial/eoRealOp.h
\ingroup parameteric
*/

template<class EOT> class eoSegmentCrossover: public eoQuadraticOp<EOT>
{
 public:
  /**
   * (Default) Constructor.
   * @param _alpha the amount of exploration OUTSIDE the parents 
   *               as in BLX-alpha notation (Eshelman and Schaffer)
   *               0 == contractive application
   */
  eoSegmentCrossover(const double& _alpha = 0.0) : 
    alpha(_alpha), range(1+2*alpha) {}

  /// The class name.
  string className() const { return "eoSegmentCrossover"; }

  /**
   * segment crossover - modifies both parents
   * @param _eo1 The first parent
   * @param _eo2 The first parent
   */
  void operator()(EOT& _eo1, EOT& _eo2) 
    {
      unsigned i;
      double r1, r2, fact;
      fact = rng.uniform(range);	   // in [0,range)
      for (i=0; i<_eo1.size(); i++)
	{
	  r1=_eo1[i];
	  r2=_eo2[i];
	  _eo1[i] = fact * r1 + (1-fact) * r2;
	  _eo2[i] = (1-fact) * r1 + fact * r2;
	}
      _eo1.invalidate();	   // shoudl test if fact was 0 or 1 :-)))
      _eo2.invalidate();
    }

protected:
  double alpha;
  double range;			   // == 1+2*alpha
};
  
/** eoArithmeticCrossover --> uniform choice in hypercube  
                 == arithmetical with different values for each coordinate
\class eoArithmeticCrossover eoRealOp.h Tutorial/eoRealOp.h
\ingroup parameteric
*/

template<class EOT> class eoArithmeticCrossover: public eoQuadraticOp<EOT>
{
 public:
  /**
   * (Default) Constructor.
   * @param _alpha the amount of exploration OUTSIDE the parents 
   *               as in BLX-alpha notation (Eshelman and Schaffer)
   *               0 == contractive application
   */
  eoArithmeticCrossover(const double& _alpha = 0.0): 
    alpha(_alpha), range(1+2*alpha) {}

  /// The class name.
  string className() const { return "eoArithmeticCrossover"; }

  /**
   * arithmetical crossover - modifies both parents
   * @param _eo1 The first parent
   * @param _eo2 The first parent
   */
  void operator()(EOT& _eo1, EOT& _eo2) 
    {
      unsigned i;
      double r1, r2, fact;
      for (i=0; i<_eo1.size(); i++)
	{
	  r1=_eo1[i];
	  r2=_eo2[i];
	  fact = rng.uniform(range);	   // in [0,range)
	  _eo1[i] = fact * r1 + (1-fact) * r2;
	  _eo2[i] = (1-fact) * r1 + fact * r2;
	}
    }

protected:
  double alpha;
  double range;			   // == 1+2*alpha
};
  

/** eoRealUxOver --> Uniform crossover, also termed intermediate crossover
\class eoRealUxOver eoRealOp.h Tutorial/eoRealOp.h
\ingroup parameteric
*/

template<class EOT> class eoRealUxOver: public eoQuadraticOp<EOT>
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
  void operator()(EOT& _eo1, EOT& _eo2) 
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
      if (changed)
	  {
	    _eo1.invalidate();
	    _eo2.invalidate();
	  }
    }
    private:
      float preference;
};
  

//-----------------------------------------------------------------------------
//@}
#endif eoRealOp_h

