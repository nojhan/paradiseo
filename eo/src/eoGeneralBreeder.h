// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoGeneralBreeder.h 
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

#ifndef eoGeneralBreeder_h
#define eoGeneralBreeder_h

//-----------------------------------------------------------------------------

/*****************************************************************************
 * eoGeneralBreeder: transforms a population using the generalOp construct.
 *****************************************************************************/

#include <eoOp.h>
#include <eoPopulator.h>

/**
  Base class for breeders using generalized operators.
*/
template<class EOT> 
class eoGeneralBreeder: public eoBreed<EOT>
{
 public:
  /** Ctor:
   *
   * @param _select a selectoOne, to be used for all selections
   * @param _op a general operator (will generally be an eoOpContainer)
   * @param _rate               pour howMany, le nbre d'enfants a generer
   * @param _interpret_as_rate  <a href="../../tutorial/html/eoEngine.html#howmany">explanation</a>
   */
  eoGeneralBreeder( eoSelectOne<EOT>& _select, eoGenOp<EOT>& _op, 
	       double  _rate=1.0, bool _interpret_as_rate = true) : 
    select( _select ), op(_op),  howMany(_rate, _interpret_as_rate) {}
  
  /** The breeder: simply calls the genOp on a selective populator!
   *
   * @param _parents the initial population
   * @param _offspring the resulting population (content -if any- is lost)
   */
  void operator()(const eoPop<EOT>& _parents, eoPop<EOT>& _offspring)
    {
      unsigned target = howMany(_parents.size());

      eoSelectivePopulator<EOT> it(_parents, select);

      select.setup(_parents);

      while (it.size() < target)
	{
	  op(it);
	}
      
      swap(_offspring, it);
      _offspring.resize(target);   // you might have generated a few more
    }
  
  /// The class name.
  string className() const { return "eoGeneralBreeder"; }
  
 private:
  eoSelectOne<EOT>& select;
  eoGenOp<EOT>& op;
  eoHowMany howMany;
};

#endif eoBreeder_h

