// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoNormalMutation.h
// (c) EEAAX 2001 - Maarten Keijzer 2000
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

#ifndef eoNormalMutation_h
#define eoNormalMutation_h

//-----------------------------------------------------------------------------

#include <algorithm>    // swap_ranges
#include <utils/eoRNG.h>
#include <utils/eoUpdatable.h>
#include <eoEvalFunc.h>
#include <es/eoReal.h>
#include <utils/eoRealBounds.h>
//-----------------------------------------------------------------------------

/** Simple normal mutation of a vector of real values.
 *  The stDev is fixed - but it is passed ans stored as a reference, 
 *  to enable dynamic mutations (see eoOenFithMutation below).
 *
 * As for the bounds, the values are here folded back into the bounds.
 * The other possiblity would be to iterate until we fall inside the bounds -
 *     but this sometimes takes a long time!!!
 */

template<class EOT> class eoNormalMutation: public eoMonOp<EOT>
{
 public:
  /**
   * (Default) Constructor.
   * The bounds are initialized with the global object that says: no bounds.
   *
   * @param _sigma the range for uniform nutation
   * @param _p_change the probability to change a given coordinate
   */
  eoNormalMutation(double & _sigma, const double& _p_change = 1.0):
    sigma(_sigma), bounds(eoDummyVectorNoBounds), p_change(_p_change) {}

  /**
   * Constructor with bounds
   * @param _bounds an eoRealVectorBounds that contains the bounds
   * @param _sigma the range for uniform nutation
   * @param _p_change the probability to change a given coordinate
   */
  eoNormalMutation(eoRealVectorBounds & _bounds,
		    double & _sigma, const double& _p_change = 1.0):
    sigma(_sigma), bounds(_bounds), p_change(_p_change) {}

  /// The class name.
  string className() const { return "eoNormalMutation"; }

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
	      _eo[lieu] += sigma*rng.normal();
	      bounds.foldsInBounds(lieu, _eo[lieu]);
	      hasChanged = true;
	    }
	}
      return hasChanged;
    }

protected:
  double & sigma;
private:
  eoRealVectorBounds & bounds;
  double p_change;
};

/** the dynamic version: just say it is updatable -
 *  and write the update() method!
 *  here the 1 fifth rule: count the proportion of successful mutations, and
 *  increase sigma if more than threshold (1/5 !)
 */

template<class EOT> class eoOneFifthMutation : 
  public eoNormalMutation<EOT>, public eoUpdatable
{
public:
  typedef typename EOT::Fitness Fitness; 

  /**
   * (Default) Constructor.
   *
   * @param eval the evaluation fuinction, needed to recompute the fitmess
   * @param _sigmaInit the initial value for uniform nutation
   * @param _windowSize the size of the window for statistics
   * @param _threshold the threshold (the 1/5 - 0.2)
   * @param _updateFactor multiplicative update factor for sigma 
   */
  eoOneFifthMutation(eoEvalFunc<EOT> & _eval, double & _sigmaInit, 
		     unsigned _windowSize = 10, 
		     double _threshold=0.2, double _updateFactor=0.83): 
    eoNormalMutation<EOT>(_sigmaInit), eval(_eval), 
    threshold(_threshold), updateFactor(_updateFactor), 
    nbMut(_windowSize, 0), nbSuccess(_windowSize, 0), genIndex(0) {}

  /**
   * Do it!
   * @param _eo The cromosome undergoing the mutation
   * calls the standard mutation, then checks for success
   */
  void operator()(EOT & _eo) 
    {
      Fitness oldFitness = _eo.fitness(); // save old fitness

      eoNormalMutation<EOT>::operator()(_eo); // normal mutation
      nbMut++;		   // assumes normal mutation always modifies _eo

      eval(_eo);		   // compute fitness of offspring

      if (_eo.fitness() > oldFitness)
	nbSuccess++;		    // update counter
    }
  
  // this will be called every generation
  void update()
  {
    unsigned totalMut = 0;
    unsigned totalSuccess = 0;
    // compute the average stats over the time window
    for ( unsigned i=0; i<nbMut.size(); i++)
      {
	totalMut += nbMut[i];
	totalSuccess += nbSuccess[i];
      }

    // update sigma accordingly
    double prop = (double) totalSuccess / totalMut;
    if (prop > threshold)
      sigma /= updateFactor;	   // increase sigma
    else
      sigma *= updateFactor;	   // decrease sigma

    // go to next generation
    genIndex = (genIndex+1) % nbMut.size() ;
    nbMut[genIndex] = nbSuccess[genIndex] = 0;
  }
private:
  eoEvalFunc<EOT> & eval;
  double threshold;		   // 1/5 !
  double updateFactor ;		   // the multiplicative factor
  vector<unsigned> nbMut;	   // total number of mutations per gen
  vector<unsigned> nbSuccess;	   // number of successful mutations per gen
  unsigned genIndex ;		   // current gen
};


//-----------------------------------------------------------------------------
//@}
#endif eoRealOp_h

