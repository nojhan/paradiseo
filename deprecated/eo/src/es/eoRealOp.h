// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRealOp.h
// (c) Maarten Keijzer 2000 - Marc Schoenauer 2001
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
#include <utils/eoRealVectorBounds.h>

//-----------------------------------------------------------------------------

/** eoUniformMutation --> changes all values of the std::vector
                          by uniform choice with range epsilon
                          with probability p_change per variable
\class eoUniformMutation eoRealOp.h Tutorial/eoRealOp.h
 *
 * @ingroup Real
 * @ingroup Variators
*/

template<class EOT> class eoUniformMutation: public eoMonOp<EOT>
{
 public:
  /**
   * Constructor without bounds == unbounded variables :-)
   * not very clean, but who's doing unbounded optimization anyway?
   * and it's there mostly for backward compatibility
   *
   * @param _epsilon the range for uniform nutation
   * @param _p_change the probability to change a given coordinate
   */
  eoUniformMutation(const double& _epsilon, const double& _p_change = 1.0):
    homogeneous(true), bounds(eoDummyVectorNoBounds), epsilon(1, _epsilon),
    p_change(1, _p_change) {}

  /**
   * Constructor with bounds
   * @param _bounds an eoRealVectorBounds that contains the bounds
   * @param _epsilon the range for uniform mutation - a double to be scaled
   * @param _p_change the one probability to change all coordinates
   */
  eoUniformMutation(eoRealVectorBounds & _bounds,
                    const double& _epsilon, const double& _p_change = 1.0):
    homogeneous(false), bounds(_bounds), epsilon(_bounds.size(), _epsilon),
    p_change(_bounds.size(), _p_change)
  {
    // scale to the range - if any
    for (unsigned i=0; i<bounds.size(); i++)
      if (bounds.isBounded(i))
          epsilon[i] *= _epsilon*bounds.range(i);
  }

  /**
   * Constructor with bounds
   * @param _bounds an eoRealVectorBounds that contains the bounds
   * @param _epsilon the VECTOR of ranges for uniform mutation
   * @param _p_change the VECTOR of probabilities for each coordinates
   */
  eoUniformMutation(eoRealVectorBounds & _bounds,
                    const std::vector<double>& _epsilon,
                    const std::vector<double>& _p_change):
    homogeneous(false), bounds(_bounds), epsilon(_epsilon),
    p_change(_p_change) {}

  /// The class name.
  virtual std::string className() const { return "eoUniformMutation"; }

  /**
   * Do it!
   * @param _eo The indi undergoing the mutation
   */
  bool operator()(EOT& _eo)
    {
      bool hasChanged=false;
      if (homogeneous)             // implies no bounds object
        for (unsigned lieu=0; lieu<_eo.size(); lieu++)
          {
            if (rng.flip(p_change[0]))
              {
                _eo[lieu] += 2*epsilon[0]*rng.uniform()-epsilon[0];
                hasChanged = true;
              }
          }
      else
        {
          // sanity check ?
          if (_eo.size() != bounds.size())
            throw std::runtime_error("Invalid size of indi in eoUniformMutation");

          for (unsigned lieu=0; lieu<_eo.size(); lieu++)
            if (rng.flip(p_change[lieu]))
              {
                // check the bounds
                double emin = _eo[lieu]-epsilon[lieu];
                double emax = _eo[lieu]+epsilon[lieu];
                if (bounds.isMinBounded(lieu))
                  emin = std::max(bounds.minimum(lieu), emin);
                if (bounds.isMaxBounded(lieu))
                  emax = std::min(bounds.maximum(lieu), emax);
                _eo[lieu] = emin + (emax-emin)*rng.uniform();
                hasChanged = true;
              }
        }
      return hasChanged;
    }

private:
  bool homogeneous;   // == no bounds passed in the ctor
  eoRealVectorBounds & bounds;
  std::vector<double> epsilon;     // the ranges for mutation
  std::vector<double> p_change;    // the proba that each variable is modified
};

/** eoDetUniformMutation --> changes exactly k values of the std::vector
                          by uniform choice with range epsilon
\class eoDetUniformMutation eoRealOp.h Tutorial/eoRealOp.h
 *
 * @ingroup Real
 * @ingroup Variators
*/

template<class EOT> class eoDetUniformMutation: public eoMonOp<EOT>
{
 public:
  /**
   * (Default) Constructor for homogeneous genotype
   * it's there mostly for backward compatibility
   *
   * @param _epsilon the range for uniform nutation
   * @param _no number of coordinate to modify
   */
  eoDetUniformMutation(const double& _epsilon, const unsigned& _no = 1):
    homogeneous(true), bounds(eoDummyVectorNoBounds),
    epsilon(1, _epsilon), no(_no) {}

  /**
   * Constructor with bounds
   * @param _bounds an eoRealVectorBounds that contains the bounds
   * @param _epsilon the range for uniform nutation (to be scaled if necessary)
   * @param _no number of coordinate to modify
   */
  eoDetUniformMutation(eoRealVectorBounds & _bounds,
                       const double& _epsilon, const unsigned& _no = 1):
    homogeneous(false), bounds(_bounds),
    epsilon(_bounds.size(), _epsilon), no(_no)
  {
    // scale to the range - if any
    for (unsigned i=0; i<bounds.size(); i++)
      if (bounds.isBounded(i))
          epsilon[i] *= _epsilon*bounds.range(i);
  }

  /**
   * Constructor with bounds and full std::vector of epsilon
   * @param _bounds an eoRealVectorBounds that contains the bounds
   * @param _epsilon the VECTOR of ranges for uniform mutation
   * @param _no number of coordinates to modify
   */
  eoDetUniformMutation(eoRealVectorBounds & _bounds,
                       const std::vector<double>& _epsilon,
                       const unsigned& _no = 1):
    homogeneous(false), bounds(_bounds), epsilon(_epsilon), no(_no)
  {
    // scale to the range - if any
    for (unsigned i=0; i<bounds.size(); i++)
      if (bounds.isBounded(i))
          epsilon[i] *= _epsilon[i]*bounds.range(i);
  }

  /// The class name.
  virtual std::string className() const { return "eoDetUniformMutation"; }

  /**
   * Do it!
   * @param _eo The indi undergoing the mutation
   */
  bool operator()(EOT& _eo)
    {
      if (homogeneous)
        for (unsigned i=0; i<no; i++)
          {
            unsigned lieu = rng.random(_eo.size());
            // actually, we should test that we don't re-modify same variable!
            _eo[lieu] = 2*epsilon[0]*rng.uniform()-epsilon[0];
          }
      else
        {
          // sanity check ?
          if (_eo.size() != bounds.size())
            throw std::runtime_error("Invalid size of indi in eoDetUniformMutation");
          for (unsigned i=0; i<no; i++)
            {
              unsigned lieu = rng.random(_eo.size());
              // actually, we should test that we don't re-modify same variable!

              // check the bounds
              double emin = _eo[lieu]-epsilon[lieu];
              double emax = _eo[lieu]+epsilon[lieu];
              if (bounds.isMinBounded(lieu))
                emin = std::max(bounds.minimum(lieu), emin);
              if (bounds.isMaxBounded(lieu))
                emax = std::min(bounds.maximum(lieu), emax);
              _eo[lieu] = emin + (emax-emin)*rng.uniform();
            }
        }
      return true;
    }

private:
  bool homogeneous;   //  == no bounds passed in the ctor
  eoRealVectorBounds & bounds;
  std::vector<double> epsilon;     // the ranges of mutation
  unsigned no;
};


// two arithmetical crossovers

/** eoSegmentCrossover --> uniform choice in segment
                 == arithmetical with same value along all coordinates
\class eoSegmentCrossover eoRealOp.h Tutorial/eoRealOp.h
 *
 * @ingroup Real
 * @ingroup Variators
*/

template<class EOT> class eoSegmentCrossover: public eoQuadOp<EOT>
{
 public:
  /**
   * (Default) Constructor.
   * The bounds are initialized with the global object that says: no bounds.
   *
   * @param _alpha the amount of exploration OUTSIDE the parents
   *               as in BLX-alpha notation (Eshelman and Schaffer)
   *               0 == contractive application
   *               Must be positive
   */
  eoSegmentCrossover(const double& _alpha = 0.0) :
    bounds(eoDummyVectorNoBounds), alpha(_alpha), range(1+2*_alpha) {}

  /**
   * Constructor with bounds
   * @param _bounds an eoRealVectorBounds that contains the bounds
   * @param _alpha the amount of exploration OUTSIDE the parents
   *               as in BLX-alpha notation (Eshelman and Schaffer)
   *               0 == contractive application
   *               Must be positive
   */
  eoSegmentCrossover(eoRealVectorBounds & _bounds,
                     const double& _alpha = 0.0) :
    bounds(_bounds), alpha(_alpha), range(1+2*_alpha) {}

  /// The class name.
  virtual std::string className() const { return "eoSegmentCrossover"; }

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
      if (alpha == 0.0)            // no check to perform
        fact = -alpha + rng.uniform(range); // in [-alpha,1+alpha)
      else                         // look for the bounds for fact
        {
          for (i=0; i<_eo1.size(); i++)
            {
              r1=_eo1[i];
              r2=_eo2[i];
              if (r1 != r2) {      // otherwise you'll get NAN's
                double rmin = std::min(r1, r2);
                double rmax = std::max(r1, r2);
                double length = rmax - rmin;
                if (bounds.isMinBounded(i))
                  {
                    alphaMin = std::max(alphaMin, (bounds.minimum(i)-rmin)/length);
                    alphaMax = std::min(alphaMax, (rmax-bounds.minimum(i))/length);
                  }
                if (bounds.isMaxBounded(i))
                  {
                    alphaMax = std::min(alphaMax, (bounds.maximum(i)-rmin)/length);
                    alphaMin = std::max(alphaMin, (rmax-bounds.maximum(i))/length);
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
      return true;         // shoudl test if fact was 0 or 1 :-)))
    }

protected:
  eoRealVectorBounds & bounds;
  double alpha;
  double range;                    // == 1+2*alpha
};

/** eoHypercubeCrossover --> uniform choice in hypercube
                 == arithmetical with different values for each coordinate
\class eoArithmeticCrossover eoRealOp.h Tutorial/eoRealOp.h
 *
 * @ingroup Real
 * @ingroup Variators
*/
template<class EOT> class eoHypercubeCrossover: public eoQuadOp<EOT>
{
 public:
  /**
   * (Default) Constructor.
   * The bounds are initialized with the global object that says: no bounds.
   *
   * @param _alpha the amount of exploration OUTSIDE the parents
   *               as in BLX-alpha notation (Eshelman and Schaffer)
   *               0 == contractive application
   *               Must be positive
   */
  eoHypercubeCrossover(const double& _alpha = 0.0):
    bounds(eoDummyVectorNoBounds), alpha(_alpha), range(1+2*_alpha)
  {
    if (_alpha < 0)
      throw std::runtime_error("BLX coefficient should be positive");
  }

  /**
   * Constructor with bounds
   * @param _bounds an eoRealVectorBounds that contains the bounds
   * @param _alpha the amount of exploration OUTSIDE the parents
   *               as in BLX-alpha notation (Eshelman and Schaffer)
   *               0 == contractive application
   *               Must be positive
   */
  eoHypercubeCrossover(eoRealVectorBounds & _bounds,
                        const double& _alpha = 0.0):
    bounds(_bounds), alpha(_alpha), range(1+2*_alpha)
  {
    if (_alpha < 0)
      throw std::runtime_error("BLX coefficient should be positive");
  }

  /// The class name.
  virtual std::string className() const { return "eoHypercubeCrossover"; }

  /**
   * hypercube crossover - modifies both parents
   * @param _eo1 The first parent
   * @param _eo2 The first parent
   */
  bool operator()(EOT& _eo1, EOT& _eo2)
    {
      bool hasChanged = false;
      unsigned i;
      double r1, r2, fact;
      if (alpha == 0.0)            // no check to perform
          for (i=0; i<_eo1.size(); i++)
            {
              r1=_eo1[i];
              r2=_eo2[i];
              if (r1 != r2) {      // otherwise do nothing
                fact = rng.uniform(range);       // in [0,1)
                _eo1[i] = fact * r1 + (1-fact) * r2;
                _eo2[i] = (1-fact) * r1 + fact * r2;
                hasChanged = true; // forget (im)possible alpha=0
              }
            }
      else         // check the bounds
        // do not try to get a bound on the linear factor, but rather
        // on the object variables themselves
        for (i=0; i<_eo1.size(); i++)
          {
            r1=_eo1[i];
            r2=_eo2[i];
            if (r1 != r2) {        // otherwise do nothing
              double rmin = std::min(r1, r2);
              double rmax = std::max(r1, r2);

              // compute min and max for object variables
              double objMin = -alpha * rmax + (1+alpha) * rmin;
              double objMax = -alpha * rmin + (1+alpha) * rmax;

              // first find the limits on the alpha's
              if (bounds.isMinBounded(i))
                {
                  objMin = std::max(objMin, bounds.minimum(i));
                }
              if (bounds.isMaxBounded(i))
                {
                  objMax = std::min(objMax, bounds.maximum(i));
                }
              // then draw variables
              double median = (objMin+objMax)/2.0; // uniform within bounds
              // double median = (rmin+rmax)/2.0;  // Bounce on bounds
              double valMin = objMin + (median-objMin)*rng.uniform();
              double valMax = median + (objMax-median)*rng.uniform();
              // don't always put large value in _eo1 - or what?
              if (rng.flip(0.5))
                {
                  _eo1[i] = valMin;
                  _eo2[i] = valMax;
                }
              else
                {
                  _eo1[i] = valMax;
                  _eo2[i] = valMin;
                }
              // seomthing has changed
              hasChanged = true; // forget (im)possible alpha=0
            }
          }

    return hasChanged;
   }

protected:
  eoRealVectorBounds & bounds;
  double alpha;
  double range;                    // == 1+2*alphaMin
};


/** eoRealUxOver --> Uniform crossover, also termed intermediate crossover
\class eoRealUxOver eoRealOp.h Tutorial/eoRealOp.h
 *
 * @ingroup Real
 * @ingroup Variators
*/

template<class EOT> class eoRealUXover: public eoQuadOp<EOT>
{
 public:
  /**
   * (Default) Constructor.
   * @param _preference bias in the choice (usually, no bias == 0.5)
   */
  eoRealUXover(const float& _preference = 0.5): preference(_preference)
    {
      if ( (_preference <= 0.0) || (_preference >= 1.0) )
        std::runtime_error("UxOver --> invalid preference");
    }

  /// The class name.
  virtual std::string className() const { return "eoRealUXover"; }

  /**
   * Uniform crossover for real std::vectors
   * @param _eo1 The first parent
   * @param _eo2 The second parent
   *    @exception std::runtime_error if sizes don't match
   */
  bool operator()(EOT& _eo1, EOT& _eo2)
    {
      if ( _eo1.size() != _eo2.size())
            std::runtime_error("UxOver --> chromosomes sizes don't match" );
      bool changed = false;
      for (unsigned int i=0; i<_eo1.size(); i++)
        {
          if (rng.flip(preference))
            if (_eo1[i] != _eo2[i])
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
#endif
