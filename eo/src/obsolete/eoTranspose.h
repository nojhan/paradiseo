// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoTranspose.h
// (c) GeNeura Team, 1998 Maarten Keijzer 2000
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

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
 */
//-----------------------------------------------------------------------------

#ifndef _EOTRANSPOSE_h
#define _EOTRANSPOSE_h

#include <vector>
#include <list>

#include <utils/eoRNG.h>
#include <eoOp.h>
#include <eoFixedLength.h>
#include <eoVariableLength.h>
/** 
Transposition operator: interchanges the position of two genes
of an EO. 
*/
template <class EOT >
class eoTranspose: public eoMonOp<EOT>  
{
public:
  
  // Specialization for a vector
  void operator()(eoFixedLength<typename EOT::Fitness, typename EOT::AtomType>& _eo )
  {
    unsigned pos1 = rng.random(_eo.size()),
    pos2 = rng.random(_eo.size());
  
    if (pos1 != pos2)
        swap(_eo[pos1], _eo[pos2]);

    if (_eo[pos1] != _eo[pos2])
        _eo.invalidate();
  }

  // Specialization for a list
  void operator()(eoVariableLength<typename EOT::Fitness, typename EOT::AtomType>& _eo )
  {
    unsigned pos1 = rng.random(_eo.size()),
    pos2 = rng.random(_eo.size());
  
    if (pos1 == pos2)
        return;

    if (pos1 > pos2)
        swap(pos1,pos2);

    pos2 -= pos1;
    
    typename EOT::iterator it1 = _eo.begin();

    while (pos1--) {it1++;}

    typename EOT::iterator it2 = it1;
    
    while (pos2--) {it2++;}

    swap(*it1, *it2);

    if (*it1 != *it2)
        _eo.invalidate();
  }
  
  /** Inherited from eoObject 
      @see eoObject
  */
  virtual string className() const {return "eoTranspose";};
  //@}
    
};

#endif

