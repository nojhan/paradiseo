// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoOneToOneBreeder.h
// (c) Maarten Keijzer and Marc Schoenauer, 2001
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

    Contact: mkeijzer@dhi.dk
             Marc.Schoenauer@polytechnique.fr
 */
//-----------------------------------------------------------------------------

#ifndef eoOneToOneBreeder_h
#define eoOneToOneBreeder_h

//-----------------------------------------------------------------------------

#include <eoOp.h>
#include <eoGenOp.h>
#include <eoPopulator.h>
#include <eoSelectOne.h>
#include <eoSequentialSelect.h>
#include <eoBreed.h>
#include <eoEvalFunc.h>
#include <eoPopulator.h>
#include <utils/eoHowMany.h>

/** eoOneToOneBreeder: transforms a population using
 *   - an operator that MODIFIES only one parent from the populator
 *     (though it can use any number aside) and thus generates ONE offspring)
 *   - a local replacement between the parent and its offspring
 *
 * Typically, Differential Evolution (Storn and Price 94) and Deb et al's
 *   G3 can be built on this
 *
 *  @ingroup Combination
 */
template<class EOT>
class eoOneToOneBreeder: public eoBreed<EOT>
{
 public:
  /** Ctor:
   * @param _op       a general operator (must MODIFY only ONE parent)
   * @param _eval     an eoEvalFunc to evaluate the offspring
   * @param _pReplace probability that the best of parent/offspring wins [1]
   * @param _howMany  eoHowMany offpsring to generate [100%]
   */
  eoOneToOneBreeder(
          eoGenOp<EOT>& _op,
          eoEvalFunc<EOT> & _eval,
          double _pReplace = 1.0,
          eoHowMany _howMany = eoHowMany(1.0) ) :
      op(_op), eval(_eval), select( false ),
      pReplace(_pReplace), howMany(_howMany) {}


  /** The breeder: iteratively calls the genOp ONCE on a selective populator
   *   after having recorded the parent
   *   Then does the replacement
   *
   * @param _parents the initial population
   * @param _offspring the resulting population (content -if any- is lost)
   */
  void operator()(const eoPop<EOT>& _parents, eoPop<EOT>& _offspring)
    {
      unsigned target = howMany(_parents.size());

      _offspring.clear();
      eoSelectivePopulator<EOT> popit(_parents, _offspring, select);

      for (unsigned iParent=0; iParent<target; iParent++)
        {
          unsigned pos = popit.tellp(); // remember current position
          EOT theParent = *popit;  // remember the parent itself

          // now apply operator - will modify the parent
          op(popit);

          // replacement
          EOT & leOffspring = *popit;

          // check: only one offspring?
          unsigned posEnd = popit.tellp();
          if (posEnd != pos)
            throw std::runtime_error("Operator can only generate a SINGLE offspring in eoOneToOneBreeder");

          // do the tournament between parent and offspring
          eval(leOffspring);  // first need to evaluate the offspring
          if (theParent > leOffspring) // old parent better than offspring
            if (rng.uniform() < pReplace) // if probability
              leOffspring = theParent; // replace
          // finally, go to next guy to handle
          ++popit;
        }
    }

  /// The class name.
  virtual std::string className() const { return "eoOneToOneBreeder"; }

 private:
  eoGenOp<EOT>& op;
  eoEvalFunc<EOT> & eval;
  eoSequentialSelect<EOT> select;
  double pReplace;
  eoHowMany howMany;
};

#endif
