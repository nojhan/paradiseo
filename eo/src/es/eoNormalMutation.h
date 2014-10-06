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

/** Simple normal mutation of a std::vector of real values.
 *  The stDev is fixed - but it is passed ans stored as a reference,
 *  to enable dynamic mutations (see eoOenFithMutation below).
 *
 * As for the bounds, the values are here folded back into the bounds.
 * The other possiblity would be to iterate until we fall inside the bounds -
 *     but this sometimes takes a long time!!!
 *
 * @ingroup Real
 * @ingroup Variators
 */
template<class EOT> class eoNormalVecMutation: public eoMonOp<EOT>
{
 public:
  /**
   * (Default) Constructor.
   * The bounds are initialized with the global object that says: no bounds.
   *
   * @param _sigma the range for uniform nutation
   * @param _p_change the probability to change a given coordinate
   */
  eoNormalVecMutation(double _sigma, const double& _p_change = 1.0):
    sigma(_sigma), bounds(eoDummyVectorNoBounds), p_change(_p_change) {}

  /**
   * Constructor with bounds
   * @param _bounds an eoRealVectorBounds that contains the bounds
   * @param _sigma the range for uniform nutation
   * @param _p_change the probability to change a given coordinate
   *
   * for each component, the sigma is scaled to the range of the bound, if bounded
   */
  eoNormalVecMutation(eoRealVectorBounds & _bounds,
                    double _sigma, const double& _p_change = 1.0):
    sigma(_bounds.size(), _sigma), bounds(_bounds), p_change(_p_change)
  {
    // scale to the range - if any
    for (unsigned i=0; i<bounds.size(); i++)
      if (bounds.isBounded(i))
          sigma[i] *= _sigma*bounds.range(i);
  }

  /** The class name */
  virtual std::string className() const { return "eoNormalVecMutation"; }

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
              _eo[lieu] += sigma[lieu]*rng.normal();
              bounds.foldsInBounds(lieu, _eo[lieu]);
              hasChanged = true;
            }
        }
      return hasChanged;
    }

private:
  std::vector<double> sigma;
  eoRealVectorBounds & bounds;
  double p_change;
};

/** Simple normal mutation of a std::vector of real values.
 *  The stDev is fixed - but it is passed ans stored as a reference,
 *  to enable dynamic mutations (see eoOenFithMutation below).
 *
 * As for the bounds, the values are here folded back into the bounds.
 * The other possiblity would be to iterate until we fall inside the bounds -
 *     but this sometimes takes a long time!!!
 *
 * @ingroup Real
 * @ingroup Variators
 */
template<class EOT> class eoNormalMutation
    : public eoMonOp<EOT>
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

  /** The class name */
  virtual std::string className() const { return "eoNormalMutation"; }

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

  /** Accessor to ref to sigma - for update and monitor */
  double & Sigma() {return sigma;}

private:
  double & sigma;
  eoRealVectorBounds & bounds;
  double p_change;
};

/** the dynamic version: just say it is updatable -
 *  and write the update() method!
 *  here the 1 fifth rule: count the proportion of successful mutations, and
 *  increase sigma if more than threshold (1/5 !)
 *
 * @ingroup Real
 * @ingroup Variators
 */
template<class EOT> class eoOneFifthMutation :
  public eoNormalMutation<EOT>, public eoUpdatable
{
public:

    using eoNormalMutation< EOT >::Sigma;

    typedef typename EOT::Fitness Fitness;

  /**
   * (Default) Constructor.
   *
   * @param _eval the evaluation function, needed to recompute the fitmess
   * @param _sigmaInit the initial value for uniform mutation
   * @param _windowSize the size of the window for statistics
   * @param _updateFactor multiplicative update factor for sigma
   * @param _threshold the threshold (the 1/5 - 0.2)
   */
  eoOneFifthMutation(eoEvalFunc<EOT> & _eval, double & _sigmaInit,
                     unsigned _windowSize = 10, double _updateFactor=0.83,
                     double _threshold=0.2):
    eoNormalMutation<EOT>(_sigmaInit), eval(_eval),
    threshold(_threshold), updateFactor(_updateFactor),
    nbMut(_windowSize, 0), nbSuccess(_windowSize, 0), genIndex(0)
  {
    // minimal check
    if (updateFactor>=1)
      throw std::runtime_error("Update factor must be < 1 in eoOneFifthMutation");
  }

  /** The class name */
  virtual std::string className() const { return "eoOneFifthMutation"; }

  /**
   * Do it!
   * calls the standard mutation, then checks for success and updates stats
   *
   * @param _eo The chromosome undergoing the mutation
   */
  bool operator()(EOT & _eo)
    {
      if (_eo.invalid())           // due to some crossover???
        eval(_eo);
      Fitness oldFitness = _eo.fitness(); // save old fitness

      // call standard operator - then count the successes
      if (eoNormalMutation<EOT>::operator()(_eo)) // _eo has been modified
        {
          _eo.invalidate();        // don't forget!!!
          nbMut[genIndex]++;
          eval(_eo);               // compute fitness of offspring

          if (_eo.fitness() > oldFitness)
            nbSuccess[genIndex]++;          // update counter
        }
      return false;                // because eval has reset the validity flag
    }

  /** the method that will be called every generation
   *  if the object is added to the checkpoint
   */
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
    double prop = double(totalSuccess) / totalMut;
    if (prop > threshold) {
      Sigma() /= updateFactor;     // increase sigma
    }
    else
      {
        Sigma() *= updateFactor;           // decrease sigma
      }
    genIndex = (genIndex+1) % nbMut.size() ;
    nbMut[genIndex] = nbSuccess[genIndex] = 0;

  }

private:
  eoEvalFunc<EOT> & eval;
  double threshold;                // 1/5 !
  double updateFactor ;            // the multiplicative factor
  std::vector<unsigned> nbMut;     // total number of mutations per gen
  std::vector<unsigned> nbSuccess;         // number of successful mutations per gen
  unsigned genIndex ;              // current index in std::vectors (circular)
};


//-----------------------------------------------------------------------------
//@}
#endif
