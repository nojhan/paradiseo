/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    -----------------------------------------------------------------------------
    eoProportionalGOpSelector.h
      Proportional Generalized Operator Selector.

    (c) Maarten Keijzer (mkeijzer@mad.scientist.com), GeNeura Team 1998, 1999, 2000
 
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

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
*/

#ifndef eoProportionalGOpSelector_h
#define eoProportionalGOpSelector_h

//-----------------------------------------------------------------------------

#include "eoGOpSelector.h"

/** eoProportionalGOpSel: do proportional selection, returns one of the 
    operators 
*/
template <class EOT> 
class eoProportionalGOpSel : public eoGOpSelector<EOT>
{
public :
  eoProportionalGOpSel() : eoGOpSelector<EOT>() {}
  
  /** Returns the operator proportionally selected */
  virtual eoGeneralOp<EOT>& selectOp()
    {
      unsigned what = rng.roulette_wheel(getRates());
      return *operator[](what);
    }

   ///
  virtual string className() const { return "eoGOpSelector"; };
};

#endif

