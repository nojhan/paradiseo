/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    -----------------------------------------------------------------------------
    eoSequentialGOpSel.h
      Sequential Generalized Operator Selector.

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

#ifndef eoSequentialGOpSel_h
#define eoSequentialGOpSel_h

//-----------------------------------------------------------------------------

#include <eoGOpSelector.h>
/** eoSequentialGOpSel: return a sequence of
    operations to be applied one after the other. The order of the 
    operators is significant. If for instance you first add a 
    quadratic operator and then a mutation operator, 

  @see eoGeneralOp, eoCombinedOp
*/
template <class EOT> 
class eoSequentialGOpSel : public eoGOpSelector<EOT>
{
public :
  
  eoSequentialGOpSel(void) : combined(*this, getRates()) {}
  
  virtual eoGeneralOp<EOT>& selectOp()
  {    
    return combined;
  }		
  
private :
  
  eoCombinedOp<EOT> combined;
};

#endif

