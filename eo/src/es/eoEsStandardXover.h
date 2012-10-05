/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEsLocalXover.h : ES global crossover
// (c) Marc Schoenauer 2001

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

    Contact: marc.schoenauer@polytechnique.fr http://eeaax.cmap.polytchnique.fr/
 */
//-----------------------------------------------------------------------------


#ifndef _eoEsLocalXover_H
#define _eoEsLocalXover_H

#include <utils/eoRNG.h>

#include <es/eoEsSimple.h>
#include <es/eoEsStdev.h>
#include <es/eoEsFull.h>

#include <eoGenOp.h>
// needs a selector - here random
#include <eoRandomSelect.h>

/** Standard (i.e. eoBinOp) crossover operator for ES genotypes.
 *  Uses some Atom crossovers to handle both the object variables
 *  and the mutation strategy parameters
 *  It is an  eoBinOp and has to be wrapped into an eoGenOp before being used
 *  like the global version
 *
 *  @ingroup Real
 *  @ingroup Variators
 */
template<class EOT>
class eoEsStandardXover: public eoBinOp<EOT>
{
public:
  typedef typename EOT::Fitness FitT;

  /**
   * (Default) Constructor.
   */
  eoEsStandardXover(eoBinOp<double> & _crossObj, eoBinOp<double> & _crossMut) :
    crossObj(_crossObj), crossMut(_crossMut) {}

  /// The class name. Used to display statistics
  virtual std::string className() const { return "eoEsStandardXover"; }

  /**
   * modifies one parents in the populator
   *     using a second parent
   */
  bool operator()(EOT& _eo1, const EOT& _eo2)
    {
      bool bLoc=false;
    // first, the object variables
    for (unsigned i=0; i<_eo1.size(); i++)
      {
        bLoc |= crossObj(_eo1[i], _eo2[i]); // apply eoBinOp
      }
    // then the self-adaptation parameters
    bLoc |= cross_self_adapt(_eo1, _eo2);
    return bLoc;
  }

private:

  // the method to cross slef-adaptation parameters: need to specialize

  bool cross_self_adapt(eoEsSimple<FitT> & _parent1, const eoEsSimple<FitT> & _parent2)
  {
    return crossMut(_parent1.stdev, _parent2.stdev); // apply eoBinOp
  }

  bool cross_self_adapt(eoEsStdev<FitT> & _parent1, const eoEsStdev<FitT> & _parent2)
  {
    bool bLoc=false;
    for (unsigned i=0; i<_parent1.size(); i++)
      {
        bLoc |= crossMut(_parent1.stdevs[i], _parent2.stdevs[i]); // apply eoBinOp
      }
    return bLoc;
  }

  bool cross_self_adapt(eoEsFull<FitT> & _parent1, const eoEsFull<FitT> & _parent2)
  {
    bool bLoc=false;
    unsigned i;
    // the StDev
    for (i=0; i<_parent1.size(); i++)
      {
        bLoc |= crossMut(_parent1.stdevs[i], _parent2.stdevs[i]); // apply eoBinOp
      }
    // the roataion angles
    for (i=0; i<_parent1.correlations.size(); i++)
      {
        bLoc |= crossMut(_parent1.correlations[i], _parent2.correlations[i]); // apply eoBinOp
      }
    return bLoc;

  }

  // the data
  eoRandomSelect<EOT> sel;
  eoBinOp<double> & crossObj;
  eoBinOp<double> & crossMut;
};

#endif
